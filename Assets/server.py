import socket
import json
import threading
import torch
import os
from dqn_model import DQN
from queue import Queue

model = DQN()
if os.path.exists("dqn_model.pth"):
    model.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device("cpu")))
    print("[INFO] Loaded DQN model")
else:
    print("[WARN] Model not found, using random weights")

model.eval()

HOST = '127.0.0.1'
PORT_RECEIVE = 5005
PORT_SEND = 5006

send_queue = Queue()

def handle_receive():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT_RECEIVE))
        s.listen()
        print(f"[RECEIVE] Listening on {HOST}:{PORT_RECEIVE}")
        conn, addr = s.accept()
        print(f"[RECEIVE] Connected: {addr}")
        with conn:
            buffer = ""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        payload = json.loads(line)
                        player = payload["player"]
                        bot = payload["bot"]

                        rel_pos = [
                            player["position"]["x"] - bot["position"]["x"],
                            player["position"]["y"] - bot["position"]["y"],
                            player["position"]["z"] - bot["position"]["z"]
                        ]

                        bot_forward = [
                            bot["rotation"]["x"],
                            bot["rotation"]["y"],
                            bot["rotation"]["z"]
                        ]

                        input_tensor = torch.tensor(rel_pos + bot_forward, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            pred_rot = model(input_tensor).squeeze(0).tolist()

                        bot_state = {
                            "rotation": {
                                "x": pred_rot[0],
                                "y": pred_rot[1],
                                "z": pred_rot[2],
                                "w": pred_rot[3]
                            }
                        }

                        print("[SEND] Rotation Pred:", bot_state["rotation"])  # Debug log
                        send_queue.put(json.dumps(bot_state).encode('utf-8'))

                    except Exception as e:
                        print("[ERROR]", e)

def handle_send():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT_SEND))
        s.listen()
        print(f"[SEND] Listening on {HOST}:{PORT_SEND}")
        conn, addr = s.accept()
        print(f"[SEND] Connected: {addr}")
        with conn:
            while True:
                try:
                    data = send_queue.get(timeout=1)
                    conn.sendall(data)
                except:
                    continue

if __name__ == "__main__":
    threading.Thread(target=handle_receive, daemon=True).start()
    handle_send()
