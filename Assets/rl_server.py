import torch
import torch.nn as nn
import numpy as np
import socket
import json
import threading

# Define actions
actions = ["left", "right", "forward", "shift+forward", "backward", "fire"]

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define the PPO Policy Network
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)  # Action probabilities
        self.critic = nn.Linear(128, 1)  # Value estimation

    def forward(self, x):
        x = self.fc(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# Ask user for model selection
model_type = input("Select Model (DQN/PPO): ").strip().upper()
state_dim = 6  # AI spaceship (x, y, z), Player spaceship (x, y, z)
action_dim = len(actions)

# Load the selected model
model = None
if model_type == "DQN":
    model = DQN(state_dim, action_dim)
    model_path = "dqn_model.pth"
elif model_type == "PPO":
    model = PPOPolicy(state_dim, action_dim)
    model_path = "ppo_model.pth"
else:
    print("[ERROR] Invalid model type. Please restart and choose DQN or PPO.")
    exit(1)

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"[INFO] {model_type} Model Loaded Successfully")
except Exception as e:
    print(f"[ERROR] Failed to load {model_type} model:", e)
    exit(1)

# Server configuration
HOST = "127.0.0.1"
PORT = 5005

def handle_client(conn):
    global model, model_type
    print("[DEBUG] Client handler started")
    
    buffer = ""

    while True:
        try:
            print("[DEBUG] Waiting for data from Unity...")
            data = conn.recv(1024).decode()
            
            if not data:
                print("[ERROR] No data received. Closing connection.")
                break
            
            buffer += data  # Append received data to buffer
            
            while "\n" in buffer:  # Process complete JSON messages
                json_data, buffer = buffer.split("\n", 1)
                print("[DEBUG] Received JSON:", json_data)

                try:
                    state_json = json.loads(json_data)
                    state = np.array([
                        state_json["bot_x"], state_json["bot_y"], state_json["bot_z"],
                        state_json["player_x"], state_json["player_y"], state_json["player_z"]
                    ], dtype=np.float32)
                    
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)

                    if model_type == "DQN":
                        with torch.no_grad():
                            action = torch.argmax(model(state_tensor)).item()
                    else:  # PPO
                        with torch.no_grad():
                            action_probs, _ = model(state_tensor)
                            action = torch.multinomial(action_probs, 1).item()

                    print("[DEBUG] Sending Action to Unity:", action)
                    conn.sendall((str(action) + "\n").encode())

                except json.JSONDecodeError as e:
                    print("[ERROR] JSON Parsing Error:", e)
                    buffer = ""
                    break

        except Exception as e:
            print("[ERROR] Exception:", e)
            break

    conn.close()
    print("[DEBUG] Client Disconnected")

def start_server():
    """Starts the TCP server for Unity communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f"[INFO] {model_type} Server started on {HOST}:{PORT}")

        while True:
            conn, addr = server.accept()
            print(f"[INFO] New Connection from {addr}")
            client_thread = threading.Thread(target=handle_client, args=(conn,), daemon=True)
            client_thread.start()

# Run the server
if __name__ == "__main__":
    start_server()
