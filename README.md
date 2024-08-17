# Buck-Boost Converter Control using Deep Reinforcement Learning

This project aims to control a buck-boost converter efficiently using deep reinforcement learning (DRL) algorithms and compare their performance. The buck-boost converter is a power electronic device that converts DC to DC voltage, either stepping up or stepping down the input voltage. It is commonly used in load converters for devices, electric vehicles, photovoltaic cells (solar panels), and more.

## Overview

To achieve optimal control of the buck-boost converter, various techniques have been employed historically, including:

- **Traditional Techniques:** Voltage and Current Mode control, Pulse Width Modulation (PWM)
- **Advanced Techniques:** Predictive control, Adaptive control, Fuzzy logic control

In this project, we explore the use of deep reinforcement learning to develop an intelligent control system. The DRL-based control system enables real-time learning of an RL agent that determines the most accurate action based on the current state and environment. Specifically, we apply the following algorithms:

1. **Actor-Critic (AC) Algorithm**
2. **Proximal Policy Optimization (PPO) Algorithm**
3. **Deep Deterministic Policy Gradient (DDPG) Algorithm**
4. **Hybrid Model:** Combining a DRL agent (AC) with a PID controller

The performance of each algorithm is compared to evaluate their effectiveness in controlling the buck-boost converter.

## Requirements

To run this project, you will need the following:

- MATLAB
- Simulink
