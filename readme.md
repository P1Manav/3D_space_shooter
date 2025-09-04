# 3D Space Shooter with Reinforcement Learning  
*(Still in development)*  

## Demo  

 **[LINK](https://drive.google.com/file/d/1N2ebzIjOKpwEzS29V-nP8gCQ40ZsWBgs/view?usp=sharing)**
---

## Overview  
This project is a **3D space shooter game built in Unity** with AI-controlled bots powered by **Reinforcement Learning**.  

- The current AI is trained using **Deep Q-Learning (DQN)**, enabling bots to chase, rotate, and fire at the player.  
- Level 1 features **10 DQN-powered bots**.  
- A **second-level AI using Proximal Policy Optimization (PPO)** is under development for smoother continuous control and advanced tactics.  

Your mission: **survive endless waves of AI enemies** that get smarter as the models improve.  

---

## Features  
- 🎮 **Player vs AI Combat** – Take on AI spaceships in level-based gameplay.  
- 🧠 **Reinforcement Learning Bots** – Level 1 has 10 bots trained with DQN; PPO integration is in progress for future levels.  
- 🔄 **Continuous Training** – Replay memory, epsilon-greedy exploration, and target networks improve bot learning over time.  
- 🌍 **Custom RL Server** – Unity communicates with a Python-based server for training and inference.  
- 🛠️ **Reward Shaping** – Encourages efficient aiming, proximity, and shooting while penalizing poor actions.  

---

## Installation & Setup  

### 1. Clone the Repository  
```bash
git clone https://github.com/P1Manav/3D_space_shooter.git
cd 3D_space_shooter
```  

### 2. Install Requirements  
Make sure you have **Python 3.8+** and install the dependencies:  
```bash
pip install torch numpy
```  

### 3. Start the DQN Server  
Run the Python server that handles communication with Unity:  
```bash
python dqn_server.py
```  

### 4. Run the Game  
- Open the Unity project in the Unity Editor.  
- Press **Play** to start the game.  
- Bots will automatically connect to the DQN server and begin acting based on the trained model.  

---

## Reinforcement Learning Details  

### Action Space  
Each bot decides on:  
- **Yaw, Pitch, Roll** adjustments (discrete values).  
- **Shoot or not shoot**.  

Actions are encoded as combinations of rotation deltas plus a shoot flag.  

### State Space (18 values)  
Each bot observes:  
- Player position, velocity, rotation.  
- Bot position, velocity, rotation.  

All inputs are normalized before being passed to the neural network.  

### DQN Model  
- Architecture: **Fully-connected neural net** with layers:  
  - Input (18) → 256 → 128 → Output (action size).  
- Training setup:  
  - **Experience Replay Buffer**  
  - **Target Network Updates**  
  - **Epsilon-Greedy Exploration** (decaying over time)  
  - **Reward Shaping** for efficiency and accuracy.  

### PPO (Second-Level AI – *In Development*)  
- Planned for **continuous control** (smooth rotations and maneuvers).  
- Will allow bots to exhibit more fluid and advanced strategies.  
- Designed as a **higher-difficulty AI mode** beyond DQN.  

---

## Training Process Evolution  

1. **Early Prototype (DQN + PPO hybrid)**  
   - Experimental stage with support for both DQN and PPO.  
   - Dropped due to complexity and stability issues.  

2. **Intermediate DQN Models**  
   - Focused solely on DQN.  
   - Introduced replay buffers and logging for offline training.  

3. **Finalized DQN Server (`dqn_server.py`)**  
   - Implements full reinforcement learning loop with memory, target networks, and reward shaping.  
   - Expanded action space to include rotations and shooting.  
   - Live training while bots play in Unity.  

4. **Planned PPO Integration**  
   - Will introduce smoother, continuous-control AI.  
   - Acts as a **second-level challenge mode**.  

---

## Future Improvements  
- Smarter **multi-bot team coordination**.  
- Curriculum learning for **progressive difficulty**.  
- Hybrid **online/offline training** for faster convergence.  
- Full **PPO integration** for advanced AI strategies.  

---

## Credits  
- **Game Engine**: Unity  
- **Languages**: C# (Unity), Python (RL Server & Training)  
- **RL Framework**: Custom DQN (PyTorch), PPO in development  

🚀 Enjoy the game and happy coding!  
