import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim=6, output_dim=4):  # Relative position (3) + bot forward (3) => output: rotation (Quaternion: 4)
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # output in range [-1, 1] for Quaternion
        )

    def forward(self, x):
        return self.net(x)
