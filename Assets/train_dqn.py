# train_dqn.py
import socket
import json
import torch
import torch.nn as nn
import torch.optim as optim
from dqn_model import DQNModel
from datetime import datetime

HOST = '127.0.0.1'
PORT = 5005

model = DQNModel(input_dim=10, output_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

replay_buffer = []
LOG_FILE = "replay_buffer.jsonl"

def save_to_file(sample):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(sample) + "\n")

def extract_features(player_data):
    return [
        player_data["position"]["x"],
        player_data["position"]["y"],
        player_data["position"]["z"],
        player_data["velocity"]["x"],
        player_data["velocity"]["y"],
        player_data["velocity"]["z"],
        player_data["rotation"]["x"],
        player_data["rotation"]["y"],
        player_data["rotation"]["z"],
        player_data["rotation"]["w"]
    ]

def format_output(bot_data):
    return [
        bot_data["position"]["x"],
        bot_data["position"]["y"],
        bot_data["position"]["z"],
        bot_data["velocity"]["x"],
        bot_data["velocity"]["y"],
        bot_data["velocity"]["z"],
        bot_data["rotation"]["x"],
        bot_data["rotation"]["y"],
        bot_data["rotation"]["z"],
        bot_data["rotation"]["w"]
    ]

print(f"[INFO] Listening on port {PORT}...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print(f"[INFO] Connected by {addr}")
    with conn:
        while True:
            try:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    break

                payload = json.loads(data.decode())
                player = payload["player"]
                input_tensor = torch.tensor([extract_features(player)], dtype=torch.float32)

                with torch.no_grad():
                    output_tensor = model(input_tensor).squeeze().tolist()

                # Send bot response
                response = {
                    "position": {"x": output_tensor[0], "y": output_tensor[1], "z": output_tensor[2]},
                    "velocity": {"x": output_tensor[3], "y": output_tensor[4], "z": output_tensor[5]},
                    "rotation": {"x": output_tensor[6], "y": output_tensor[7], "z": output_tensor[8], "w": output_tensor[9]}
                }
                conn.sendall((json.dumps(response) + "\n").encode())

                # Log sample for later training
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "input": extract_features(player),
                    "output": output_tensor
                }
                save_to_file(sample)

            except Exception as e:
                print("[ERROR]", e)
