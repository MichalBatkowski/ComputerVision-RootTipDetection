import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import os
import wandb

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action space (velocities for x, y, z)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Define observation space ([pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Track the number of steps and set goal position
        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation
        self.sim.reset(num_agents=1)

        # Get the initial pipette position from the simulation state
        state = self.sim.get_states()
        pipette_position = list(state[f'robotId_{self.sim.robotIds[0]}']["pipette_position"])

        # Generate a random goal position within the working envelope
        self.goal_position = np.random.uniform(low=-5.0, high=5.0, size=(3,))

        # Create the initial observation (pipette position + goal position)
        observation = np.array(pipette_position + list(self.goal_position), dtype=np.float32)

        # Reset the step counter
        self.steps = 0

        # Return observation and info dictionary
        return observation, {}



    def step(self, action):
        # Execute the action in the simulation
        action = list(action) + [0]  # Append 0 for the "drop" action
        self.sim.run([action])  # Pass the action as a list of lists

        # Get the current pipette position from the simulation state
        state = self.sim.get_states()
        pipette_position = list(state[f'robotId_{self.sim.robotIds[0]}']["pipette_position"])

        # Create the observation (pipette position + goal position)
        observation = np.array(pipette_position + list(self.goal_position), dtype=np.float32)

        # Calculate the reward (negative distance to the goal)
        distance_to_goal = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        reward = -distance_to_goal

        # Check termination condition
        terminated = bool(distance_to_goal < 0.1)  # Ensure terminated is a boolean

        # Check truncation condition (maximum steps exceeded)
        truncated = bool(self.steps >= self.max_steps)  # Ensure truncated is a boolean

        # Increment the step counter
        self.steps += 1

        # Return the observation, reward, termination status, and info dictionary
        wandb.log({"reward": reward, "distance_to_goal": distance_to_goal})

        self.cumulative_reward += reward
        if terminated or truncated:
            wandb.log({"cumulative_reward": self.cumulative_reward})
            self.cumulative_reward = 0  # Reset for the next episode

        return observation, reward, terminated, truncated, {}


    def render(self, mode="human"):
        if self.render:
            # Add rendering logic if needed (not required here)
            pass

    def close(self):
        self.sim.close()
