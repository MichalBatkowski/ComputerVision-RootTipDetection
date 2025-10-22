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
    def __init__(self, render=False, max_steps=1000, use_wandb=True):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.use_wandb = use_wandb  # Add a flag to enable/disable Wandb logging

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space (velocities for x, y, z)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Define observation space ([pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z, rel_x, rel_y, rel_z])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # Track the number of steps, goal position, and cumulative reward
        self.steps = 0
        self.goal_position = None
        self.cumulative_reward = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation
        self.sim.reset(num_agents=1)

        # Get the initial pipette position from the simulation state
        state = self.sim.get_states()
        pipette_position = np.array(state[f'robotId_{self.sim.robotIds[0]}']["pipette_position"], dtype=np.float32)

        # Generate a random goal position within the working envelope
        self.goal_position = np.random.uniform(low=-5.0, high=5.0, size=(3,)).astype(np.float32)

        # Calculate the relative position vector
        relative_position = self.goal_position - pipette_position

        # Create the observation (pipette position + goal position + relative vector)
        observation = np.concatenate([pipette_position, self.goal_position, relative_position]).astype(np.float32)

        # Reset the step counter and cumulative reward
        self.steps = 0
        self.cumulative_reward = 0

        # Return observation and info dictionary
        return observation, {}



    def step(self, action):
        # Execute the action in the simulation
        action = list(action) + [0]  # Append 0 for the "drop" action
        self.sim.run([action])  # Pass the action as a list of lists

        # Get the current pipette position from the simulation state
        state = self.sim.get_states()
        pipette_position = np.array(state[f'robotId_{self.sim.robotIds[0]}']["pipette_position"], dtype=np.float32)

        # Create the observation (pipette position + goal position + relative vector)
        relative_position = self.goal_position - pipette_position
        observation = np.concatenate([pipette_position, self.goal_position, relative_position])

        # Calculate the reward (negative distance to the goal)
        distance_to_goal = np.linalg.norm(relative_position)
        reward = float(-distance_to_goal)  # Explicitly cast the reward to a float

        # Check termination condition
        terminated = bool(distance_to_goal < 0.004)  # Adjusted to a smaller threshold for better precision

        # Check truncation condition (maximum steps exceeded)
        truncated = bool(self.steps >= self.max_steps)

        # Increment the step counter and log rewards
        self.steps += 1
        self.cumulative_reward += reward

        # Log metrics to Wandb only if Wandb is enabled
        if self.use_wandb:
            wandb.log({
                "reward": reward,
                "distance_to_goal": distance_to_goal,
                "steps": self.steps,
                "terminated": int(terminated),
            })

            # Log cumulative reward at the end of the episode
            if terminated or truncated:
                wandb.log({"cumulative_reward": self.cumulative_reward})
                self.cumulative_reward = 0  # Reset for the next episode

        return observation, reward, terminated, truncated, {}



    def render(self, mode="human"):
        if self.render:
            pass  # Add rendering logic if needed

    def close(self):
        self.sim.close()
