import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from pathlib import Path

class DQNTrainer:
    def __init__(self, model, buffer_path, model_path):
        self.model = model
        self.buffer_path = Path(buffer_path)
        self.model_path = Path(model_path)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def load_buffer(self):
        if not self.buffer_path.exists():
            return []
        with open(self.buffer_path, "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def train_loop(self):
        while True:
            data = self.load_buffer()
            if len(data) < 32:
                time.sleep(2)
                continue

            batch = random.sample(data, 32)
            inputs, targets = [], []
            for sample in batch:
                player = sample["player"]
                bot = sample["bot"]

                inp = list(player["position"].values()) + \
                      list(player["velocity"].values()) + \
                      list(player["rotation"].values())

                tgt = list(bot["position"].values()) + \
                      list(bot["velocity"].values()) + \
                      list(bot["rotation"].values())[:-1]  # Ignore bot["rotation"]["w"]

                inputs.append(inp)
                targets.append(tgt)

            x = torch.tensor(inputs, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)

            self.model.train()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"[TRAIN] Loss: {loss.item():.4f}")
            time.sleep(2)
