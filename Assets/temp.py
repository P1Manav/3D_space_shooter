import torch
import torch.nn as nn

# Define the model class
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),  # Input: 10D vector
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)    # Output: dx, dy, dz, dvx, dvy, dvz, drx, dry, drz
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = DQN()

# Load saved weights
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()

# Test inference
input_tensor = torch.rand(1, 10)
output = model(input_tensor)
print("Model output:", output)
