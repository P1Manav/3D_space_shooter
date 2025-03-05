import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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

# Hyperparameters
learning_rate = 0.1
gamma = 0.99
epsilon = 1.0  
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
replay_memory_size = 10000
num_episodes = 500
max_steps = 200  # Prevent infinite loop per episode

# Define action space
actions = ["left", "right", "forward", "shift+forward", "backward", "fire"]

# Create a custom Unity-like environment
class SpaceShooterEnv:
    def __init__(self):
        self.state_dim = 6  # AI spaceship (x, y, z), Player spaceship (x, y, z)
        self.action_dim = len(actions)
        self.state = np.zeros(self.state_dim)
    
    def reset(self):
        """Reset environment and return initial state"""
        self.state = np.random.uniform(-10, 10, size=(6,))
        return self.state

    def step(self, action_idx):
        """Perform action, return next_state, reward, done"""
        action = actions[action_idx]
        reward = -0.1  # Default small negative reward

        # AI spaceship movement
        if action == "left":
            self.state[0] -= 1
        elif action == "right":
            self.state[0] += 1
        elif action == "forward":
            self.state[2] += 1
        elif action == "shift+forward":
            self.state[2] += 2
        elif action == "backward":
            self.state[2] -= 1
        elif action == "fire":
            if abs(self.state[0] - self.state[3]) < 1.5 and abs(self.state[1] - self.state[4]) < 1.5:
                reward = 10  # High reward for hitting the player

        done = reward == 10  # End episode if AI successfully shoots the player
        return self.state.copy(), reward, done  # Ensure return values are not references

# Initialize environment, model, and training utilities
env = SpaceShooterEnv()
state_dim = env.state_dim
action_dim = env.action_dim

model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
replay_memory = deque(maxlen=replay_memory_size)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.FloatTensor(state))).item()

        # Apply action in environment
        next_state, reward, done = env.step(action)

        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Train model if enough experience
        if len(replay_memory) > batch_size:
            batch = random.sample(replay_memory, batch_size)
            states, actions_, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions_ = torch.LongTensor(actions_)  # Direct conversion
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # Compute target Q-values
            with torch.no_grad():
                target_q_values = rewards + gamma * torch.max(model(next_states), dim=1)[0] * (1 - dones)

            # Compute current Q-values
            current_q_values = model(states).gather(1, actions_.unsqueeze(1)).squeeze(1)

            # Compute loss and update model
            loss = loss_fn(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay epsilon for exploration-exploitation tradeoff
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Save trained model
torch.save(model.state_dict(), "dqn_model.pth")
print("DQN Model Saved!")
