---

# 3D Space Shooter with Reinforcement Learning (DQN Phase Complete 🚀)

## Status

✅ **Deep Q-Network (DQN) AI implemented, integrated, and running in Unity**
⚙️ **Next:** PPO implementation + comparative analysis between DQN and PPO

---

## Overview

This is a **3D space shooter** game built in **Unity**, featuring AI-controlled spaceships that learn to fight the player in real-time using **Reinforcement Learning**.
The AI logic runs in a separate Python server, and Unity communicates with it via TCP.

We are implementing two RL approaches:

1. **Deep Q-Network (DQN)** – *Completed in this phase*
2. **Proximal Policy Optimization (PPO)** – *Planned for next phase*

---

## Development Timeline

### **Phase 1 – Initial RL Server Experiments**

* Started with **`rl_server.py`** — a generic RL server handling both **DQN** and **PPO** models.
* Used a **very basic state space** (just positions) and simple discrete actions.
* Limited control: bot movements felt unnatural and shooting was random.

### **Phase 2 – Custom Server for Unity Integration**

* Built **`server.py`** as a more Unity-friendly RL interface.
* Added **player/bot state exchange** for real-time action prediction.
* Early tests showed working movement but lacked smart aiming/shooting.

### **Phase 3 – Dedicated Deep Q-Network Server**

* Developed **`dqn_server.py`** — fully dedicated to DQN.
* Major improvements:

  * **Rich state representation** (position, velocity, rotation of both player & bot).
  * **Discrete action space** with combinations of yaw/pitch/roll + shooting bit.
  * **Reward shaping**:

    * Positive for facing player.
    * Bonus for being close + aiming.
    * Large reward for hitting player.
    * Penalty for being hit.
  * **Experience replay** with continuous online training.
  * **Model persistence**: saves checkpoints & always loads latest to continue learning.
  * **Target network** for stable Q-learning.

---

## Current Architecture

### **Unity Side**

* **`BotController.cs`** – Receives yaw/pitch/roll deltas and shoot flag from Python server, applies rotation, fires bullets if conditions are met.
* **`PlayerController.cs`** – Handles manual controls & shooting.
* **`Bullet.cs`** – Bullet physics + collision + TCP hit reporting to RL server.
* **`PositionSender.cs`** – Sends state updates to server every frame.

### **Python Side (`dqn_server.py`)**

* **State Encoding:** Combines normalized positions, velocities, and rotations.
* **Action Encoding:** Maps discrete index → yaw/pitch/roll changes + shoot.
* **Training Loop:** Runs in real-time while Unity plays.
* **Persistence:** Saves the latest model regularly and checkpoints every 2000 steps.

---

## Challenges Overcome

1. **TCP Sync & Latency**

   * Had to batch state messages and carefully parse `\n`-terminated JSON to avoid freezes.
2. **Bullet Handling in Unity**

   * Early issues with bullets spawning but not moving; fixed prefab instantiation & Rigidbody velocity application.
3. **RL Reward Design**

   * Initial DQN failed to learn — fixed by shaping rewards for *alignment*, *distance*, and *combat events*.
4. **Model Forgetting**

   * Prevented loss of progress by **always loading latest model** and training incrementally during gameplay.
5. **Bot Aiming**

   * Added a **view-cone check** so the bot only shoots when the player is in front.

---

## How to Run (DQN Mode)

1. **Start the DQN Server**

```bash
python dqn_server.py
```

2. **Run the Unity Project**

   * Set **BotController** to connect to `127.0.0.1:5000`.
   * Press **Play** in Unity.
   * The bot will train live as the game runs.

---

## Next Steps

* Implement **PPO-based bot** for smoother control.
* Record gameplay data to compare **learning curves** between DQN and PPO.
* Analyze **reaction time, accuracy, and win rate** for both methods.

---
