import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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

# Hyperparameters
learning_rate = 0.1
gamma = 0.99
clip_epsilon = 0.2
batch_size = 64
epochs = 10
update_frequency = 2000  # Steps before updating policy

# Define action space
actions = ["left", "right", "forward", "shift+forward", "backward", "fire"]

# Create a custom Unity-like environment
class SpaceShooterEnv:
    def __init__(self):
        self.state_dim = 6  # AI spaceship (x, y, z), Player spaceship (x, y, z)
        self.action_dim = len(actions)
        self.state = np.zeros(self.state_dim)
    
    def reset(self):
        self.state = np.random.uniform(-10, 10, size=(6,))
        return self.state

    def step(self, action_idx):
        action = actions[action_idx]
        reward = -0.1

        if action == "left":
            self.state[0] -= 1
        elif action == "right":
            self.state[0] += 1
        elif action == "forward":
            self.state[2] += 1
            reward = 1.0
        elif action == "shift+forward":
            self.state[2] += 2
            reward = 2.0
        elif action == "backward":
            self.state[2] -= 1
            reward = -1.0
        elif action == "fire":
            if abs(self.state[0] - self.state[3]) < 1.5 and abs(self.state[1] - self.state[4]) < 1.5 and abs(self.state[2] - self.state[5]) < 1.5:
                reward = 10
        done = reward == 10
        return self.state.copy(), reward, done

# Initialize environment and model
env = SpaceShooterEnv()
state_dim = env.state_dim
action_dim = env.action_dim
model = PPOPolicy(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Training loop
memory = []
timestep = 0
for episode in range(500):
    state = env.reset()
    total_reward = 0
    for step in range(200):
        timestep += 1
        state_tensor = torch.FloatTensor(state)
        action_probs, state_value = model(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, state_value, action_probs[action].item()))
        state = next_state
        total_reward += reward
        
        if timestep % update_frequency == 0:
            # PPO update
            states, actions, rewards, next_states, state_values, old_probs = zip(*memory)
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            state_values = torch.FloatTensor(state_values)
            old_probs = torch.FloatTensor(old_probs)
            
            # Compute advantages
            with torch.no_grad():
                _, next_state_values = model(next_states)
                target_values = rewards + gamma * next_state_values.squeeze()
                advantages = target_values - state_values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute new probabilities
            new_action_probs, _ = model(states)
            new_action_probs = new_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute ratio (new / old probabilities)
            ratio = new_action_probs / old_probs
            
            # Clipped surrogate loss
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = loss_fn(state_values.squeeze(), target_values)
            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            memory = []  # Clear memory after update
        
        if done:
            break
    print(f"Episode {episode + 1}/500, Total Reward: {total_reward}")

# Save trained model
torch.save(model.state_dict(), "ppo_model.pth")
print("PPO Model Saved!")
