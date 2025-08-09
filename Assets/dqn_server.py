# dqn_server.py
import socket, threading, json, time, os, random, math
import numpy as np
from collections import deque
import torch, torch.nn as nn, torch.optim as optim

HOST = "0.0.0.0"
PORT = 5000

STATE_SIZE = 18
YAW_OPTIONS   = [-6.0, 0.0, 6.0]
PITCH_OPTIONS = [-4.0, 0.0, 4.0]
ROLL_OPTIONS  = [-3.0, 0.0, 3.0]
ROT_COMBOS = [(y, p, r) for y in YAW_OPTIONS for p in PITCH_OPTIONS for r in ROLL_OPTIONS]
NUM_ROT = len(ROT_COMBOS)
ACTION_SIZE = NUM_ROT * 2

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 20000
TARGET_UPDATE_EVERY = 200
SAVE_EVERY = 1000
MODEL_PATH = "bot_brain.pth"

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

if os.path.exists(MODEL_PATH):
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        print("[DQN] Loaded model.")
    except Exception as e:
        print("[DQN] Load failed:", e)

memory = deque(maxlen=MEMORY_SIZE)
memory_lock = threading.Lock()
model_lock = threading.Lock()
train_steps = 0
agents = {}

def encode_action(idx):
    shoot_idx = idx // NUM_ROT
    rot_idx = idx % NUM_ROT
    y,p,r = ROT_COMBOS[rot_idx]
    return float(y), float(p), float(r), bool(shoot_idx)

def choose_action(state, player_in_fov=False):
    global EPSILON
    if random.random() < EPSILON:
        if player_in_fov:
            shoot_idx = 1 if random.random() < 0.8 else 0
        else:
            shoot_idx = 1 if random.random() < 0.5 else 0
        rot_idx = random.randrange(NUM_ROT)
        return shoot_idx * NUM_ROT + rot_idx
    with model_lock:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

def remember(s,a,r,s_next,done):
    with memory_lock: memory.append((s,a,r,s_next,done))

def replay():
    global train_steps
    with memory_lock:
        if len(memory) < BATCH_SIZE: return
        batch = random.sample(memory, BATCH_SIZE)
    states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor([1.0 if b[4] else 0.0 for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
    with model_lock:
        q_values = policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + (1.0 - dones) * GAMMA * next_q
        loss = loss_fn(q_values, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    train_steps += 1
    if train_steps % TARGET_UPDATE_EVERY == 0:
        with model_lock: target_net.load_state_dict(policy_net.state_dict()); print("[DQN] Target synced")
    if train_steps % SAVE_EVERY == 0:
        try:
            torch.save(policy_net.state_dict(), MODEL_PATH); print("[DQN] Model saved")
        except Exception as e: print("[DQN] Save failed:", e)

def bot_forward_from_rot(rot):
    pitch_rad = math.radians(rot[0]); yaw_rad = math.radians(rot[1])
    fx = math.cos(pitch_rad) * math.sin(yaw_rad)
    fy = -math.sin(pitch_rad)
    fz = math.cos(pitch_rad) * math.cos(yaw_rad)
    return np.array([fx,fy,fz], dtype=np.float32)

def handle_client(conn, addr):
    global EPSILON
    print("[NET] Connected:", addr)
    conn.settimeout(1.0)
    buffer = ""
    try:
        while True:
            try:
                data = conn.recv(8192)
            except socket.timeout:
                data = b""
            if not data:
                # if you want persistent connections, continue loop; if connection closed, break
                # here treat empty bytes as continue; if socket closed, recv returns b''
                pass
            else:
                buffer += data.decode('utf-8')

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip(): continue
                try:
                    msg = json.loads(line)
                except Exception as e:
                    print("[NET] JSON parse error:", e); continue

                # Hit reports may include shooter_id and victim_agent
                if "hit" in msg:
                    shooter = msg.get("shooter_id", None)
                    victim = msg.get("victim_agent", None)
                    if shooter and shooter.startswith("bot"):
                        ad = agents.get(shooter)
                        if ad and ad.get('prev_state') is not None:
                            print(f"[DQN] Bot {shooter} HIT -> +50")
                            remember(ad['prev_state'], ad['prev_action_idx'], 50.0, ad['prev_state'], True)
                            replay()
                    if victim and victim.startswith("bot") and shooter and shooter.startswith("player"):
                        vd = agents.get(victim)
                        if vd and vd.get('prev_state') is not None:
                            print(f"[DQN] Bot {victim} got HIT by player -> -30")
                            remember(vd['prev_state'], vd['prev_action_idx'], -30.0, vd['prev_state'], True)
                            replay()
                    continue

                agent_id = msg.get("agent_id", "bot_1")
                if not all(k in msg for k in ("player_pos","player_vel","player_rot","bot_pos","bot_vel","bot_rot")):
                    print("[NET] missing keys"); continue

                player_pos = np.array(msg["player_pos"], dtype=np.float32)
                player_vel = np.array(msg["player_vel"], dtype=np.float32)
                player_rot = np.array(msg["player_rot"], dtype=np.float32)
                bot_pos = np.array(msg["bot_pos"], dtype=np.float32)
                bot_vel = np.array(msg["bot_vel"], dtype=np.float32)
                bot_rot = np.array(msg["bot_rot"], dtype=np.float32)

                state = np.concatenate([player_pos, player_vel, player_rot, bot_pos, bot_vel, bot_rot]).astype(np.float32)

                dir_vec = player_pos - bot_pos
                dist = float(np.linalg.norm(dir_vec) + 1e-8)
                dir_norm = dir_vec / (dist + 1e-8)
                bot_fwd = bot_forward_from_rot(bot_rot)
                align = float(np.dot(bot_fwd, dir_norm))  # -1..1

                # player_in_fov if dot product > cos(angle)
                player_in_fov = align > math.cos(math.radians(25))

                action_idx = choose_action(state, player_in_fov)
                yaw_delta, pitch_delta, roll_delta, shoot_bool = encode_action(action_idx)

                reward = -0.01
                if player_in_fov: reward += 0.5

                prev = agents.get(agent_id, {})
                prev_dist = prev.get('prev_dist', None)
                if prev_dist is not None:
                    reward += max(-1.0, (prev_dist - dist) * 0.5)

                if prev.get('prev_state') is not None:
                    remember(prev['prev_state'], prev['prev_action_idx'], reward, state, False)
                    replay()

                agents[agent_id] = {'prev_state': state, 'prev_action_idx': action_idx, 'prev_dist': dist}

                if EPSILON > EPSILON_MIN:
                    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

                resp = {"yaw_delta": yaw_delta, "pitch_delta": pitch_delta, "roll_delta": roll_delta, "shoot": shoot_bool}
                try:
                    conn.sendall((json.dumps(resp) + "\n").encode('utf-8'))
                except Exception as e:
                    print("[NET] send failed", e)
    except Exception as e:
        print("[NET] client handler exception", e)
    finally:
        try: conn.close()
        except: pass
        print("[NET] Disconnected", addr)

def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT)); sock.listen(8)
    print("[NET] DQN listening on", HOST, PORT)
    while True:
        conn, addr = sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
