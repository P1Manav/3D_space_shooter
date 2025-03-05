import torch
import torch.nn as nn
import numpy as np
import socket
import json
import threading

# Define actions
actions = ["left", "right", "forward", "shift+forward", "backward", "fire"]

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

# Load trained PPO model
state_dim = 6  # AI spaceship (x, y, z), Player spaceship (x, y, z)
action_dim = len(actions)
model = PPOPolicy(state_dim, action_dim)

try:
    model.load_state_dict(torch.load("ppo_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Ensure model is in inference mode
    print("[INFO] PPO Model Loaded Successfully")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    exit(1)

# Server configuration
HOST = "127.0.0.1"
PORT = 5005

def handle_client(conn):
    global model
    print("[DEBUG] Client handler started")

    buffer = ""  # Buffer for handling incomplete JSON messages

    while True:
        try:
            print("[DEBUG] Waiting for data from Unity...")
            data = conn.recv(1024).decode()
            
            if not data:
                print("[ERROR] No data received. Closing connection.")
                break
            
            buffer += data  # Append received data to buffer

            while "\n" in buffer:  # Process complete JSON messages
                json_data, buffer = buffer.split("\n", 1)  # Extract one full JSON
                print("[DEBUG] Received JSON:", json_data)

                try:
                    # Parse JSON
                    state_json = json.loads(json_data)
                    state = np.array([
                        state_json["bot_x"], state_json["bot_y"], state_json["bot_z"],
                        state_json["player_x"], state_json["player_y"], state_json["player_z"]
                    ], dtype=np.float32)
                    
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)

                    with torch.no_grad():
                        action_probs, _ = model(state_tensor)
                        action = torch.multinomial(action_probs, 1).item()

                    print("[DEBUG] Sending Action to Unity:", action)
                    conn.sendall((str(action) + "\n").encode())  # Add newline for Unity parsing

                except json.JSONDecodeError as e:
                    print("[ERROR] JSON Parsing Error:", e)
                    buffer = ""  # Clear buffer if JSON is invalid
                    break  # Exit loop and wait for fresh data

        except Exception as e:
            print("[ERROR] Exception:", e)
            break

    conn.close()
    print("[DEBUG] Client Disconnected")

def start_server():
    """Starts the TCP server for Unity communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Avoid address reuse errors
        server.bind((HOST, PORT))
        server.listen()
        print(f"[INFO] Server started on {HOST}:{PORT}")

        while True:
            conn, addr = server.accept()
            print(f"[INFO] New Connection from {addr}")
            client_thread = threading.Thread(target=handle_client, args=(conn,), daemon=True)
            client_thread.start()

# Run the server
if __name__ == "__main__":
    start_server()
