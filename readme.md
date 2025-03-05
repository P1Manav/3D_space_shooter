# 3D Space Shooter with Reinforcement Learning

**(Still in development)**

## Overview
This is a 3D space shooter game built in **Unity** with AI-controlled bots using **Reinforcement Learning (RL)**. The game features **endless waves of AI enemies** that track and attack the player. The AI bots use a custom reinforcement learning framework with **Deep Q-Learning (DQN)** for discrete actions and **Proximal Policy Optimization (PPO)** for continuous actions.

## Features
- **Player vs. AI Combat:** The player must survive against intelligent AI-controlled spaceships.
- **Reinforcement Learning Bots:** AI enemies detect the player's position, move toward them, and fire bullets.
- **Multiple AI Algorithms:** The game allows switching between **DQN** and **PPO** for bot behavior.
- **Single-Player & Multiplayer Modes:** Supports endless single-player waves and potential multiplayer expansion.
- **Custom RL Server:** AI agents communicate with a Python-based RL server for training and decision-making.

## Installation & Setup

### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/space-shooter-rl.git
cd space-shooter-rl
```

### 2. Train the AI Models
Train the reinforcement learning models before running the game:
```sh
python train_dqn.py
python train_ppo.py
```

### 3. Start the RL Server
Run the reinforcement learning server to enable AI decision-making:
```sh
python rl_server.py
```

### 4. Run the Game
- Open the game in Unity and hit **Play** to test AI behaviors.
- Train new models if needed.

## Reinforcement Learning Details
### AI Actions
The AI spaceship can perform the following actions:
- **Move Left / Right**
- **Move Forward / Shift + Forward**
- **Move Backward**
- **Fire Bullet**

### AI Training
- **DQN (Deep Q-Learning):** Used for discrete actions like movement and firing.
- **PPO (Proximal Policy Optimization):** Used for continuous control and smoother movements.
- The AI ships **observe the player’s position** and decide actions to eliminate the player efficiently.

## Future Improvements
- Enhancing AI strategy with better reward shaping.
- Multiplayer AI interactions.
- Improved environment for better training efficiency.

## Credits
- **Game Engine:** Unity
- **Programming Languages:** C# (Unity), Python (RL Server & Training)
- **RL Framework:** Custom implementation using PyTorch/TensorFlow

---
Enjoy the game and happy coding! 🚀

