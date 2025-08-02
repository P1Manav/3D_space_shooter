import torch
import torch.nn as nn
import torch.optim as optim
import json

from dqn_model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

replay_buffer = []
with open("replay_buffer.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        player = data["player"]
        bot = data["bot"]

        rel_pos = [
            player["position"]["x"] - bot["position"]["x"],
            player["position"]["y"] - bot["position"]["y"],
            player["position"]["z"] - bot["position"]["z"]
        ]

        bot_forward = [bot["rotation"]["x"], bot["rotation"]["y"], bot["rotation"]["z"]]
        rotation = [bot["rotation"]["x"], bot["rotation"]["y"], bot["rotation"]["z"], bot["rotation"]["w"]]

        input_data = rel_pos + bot_forward
        replay_buffer.append((input_data, rotation))

for epoch in range(50):
    total_loss = 0.0
    for state, target in replay_buffer:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(device)

        pred = model(state_tensor)
        loss = criterion(pred, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.6f}")

torch.save(model.state_dict(), "dqn_model.pth")
print("[Saved] Trained model to dqn_model.pth")
