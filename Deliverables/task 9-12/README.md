# Simulation Environment

## Overview
This simulation demonstrates the working envelope of the pipette in a robotic environment. The working envelope defines the maximum range of motion for the pipette tip along the X, Y, and Z axes.

## Working Envelope of the Pipette

| Axis | Lower Limit | Upper Limit |
|------|-------------|-------------|
| X | -0.1875 | 0.2531 |
| Y | -0.1705 | 0.2197 |
| Z | -0.0001 | 0.2895 |
## Observations
- This range reflects the robot's full range of motion, as defined by the simulation environment.
- The simulation visually demonstrates smooth pipette movement to all boundaries.

## Dependencies
- Python 3.x
- PyBullet
- Pillow
- Simulation files (`textures`, `sim_class.py`, etc.)

## Setup and Execution
1. Ensure all dependencies are installed:
   ```bash
   pip install pybullet
