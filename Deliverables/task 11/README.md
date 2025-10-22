# ot2_hpt
# Reinforcement Learning Environment: OT2 Simulation

## Description
This repository contains a custom reinforcement learning environment implemented using the `gymnasium` library. The environment is designed for simulating pipette movements in a lab setting to achieve precise goals, such as inoculation in phenotyping research.

## Features
- **Action Space:** A continuous action space for pipette movements (x, y, z velocities).
- **Observation Space:** Includes pipette position, goal position, and relative position vectors.
- **Reward Function:** Encourages progress toward the goal and penalizes distance, with termination conditions based on success or step limits.
- **Integration with WandB:** Logs metrics such as reward, distance to the goal, and cumulative success count for experiment tracking.

## Environment Setup
1. **Dependencies:**
   - Python 3.8+
   - `gymnasium`
   - `numpy`
   - `wandb`

2. **Installation:**
   ```bash
   pip install gymnasium numpy wandb
