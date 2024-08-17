# DRL_Controllers
This repository contains exploratory work comprising the usage of different reinforcement learning algorithms along with different control algorithms to get improved response of Buck Boost Converter



The following project aims to control a buck-boost converter efficiently using deep reinforcement learning algorithms and compare the performance of each. Buck Boost Converter is a power electronic device which converts from DC to DC voltage, either stepping up or down the source voltage provided. It finds its use in load converters for devices, Electric Vehicles, Photovoltaic cells (Solar Panels) and many more. To ensure the most optimal usage of the converter, appropriate control of the same is required and is ensured by using various techniques. Traditional techniques mainly include Voltage and Current Mode control and Pulse Width modulations.
Newer techniques include Predictive control, Adaptive control and Fuzzy logic control to name a few. Deep Reinforcement Learning based control system involves the real time learning of an RL agent which decides the most accurate response (action) to a given set of input parameters (state and environment). In this project, first an Actor Critic (AC) algorithm is applied on the system, subsequently Proximal Policy Optimisation (PPO) algorithm and Deep Deterministic Policy Gradient
(DDPG) algorithms are applied. Furthermore, a hybrid model consisting of a DRL agent (AC) and a PID controller is applied and the performance of each is compared.
