import numpy as np
from ot2_gym_wrapper import OT2Env
import os

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize the environment
env = OT2Env(render=False)


# Test the environment
obs = env.reset()
done = False
steps = 0

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step: {steps}, Action: {action}, Reward: {reward}, Observation: {obs}")

    steps += 1
    done = terminated or truncated


# Close the environment
env.close()
