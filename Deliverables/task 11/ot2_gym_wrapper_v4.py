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
        self.use_wandb = use_wandb

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
        self.success_count = 0
        self.previous_distance_to_goal = None  # Track progress

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

        # Create the observation
        observation = np.concatenate([pipette_position, self.goal_position, relative_position]).astype(np.float32)

        # Reset the step counter and cumulative reward
        self.steps = 0
        self.cumulative_reward = 0
        self.previous_distance_to_goal = np.linalg.norm(relative_position)

        return observation, {}

    def step(self, action):
        # Execute the action in the simulation
        action = list(action) + [0]  # Append 0 for the "drop" action
        self.sim.run([action])

        # Get the current pipette position
        state = self.sim.get_states()
        pipette_position = np.array(state[f'robotId_{self.sim.robotIds[0]}']["pipette_position"], dtype=np.float32)

        # Create the observation
        relative_position = self.goal_position - pipette_position
        distance_to_goal = np.linalg.norm(relative_position)
        observation = np.concatenate([pipette_position, self.goal_position, relative_position])

        # Calculate the reward
        progress_reward = self.previous_distance_to_goal - distance_to_goal  # Reward for progress
        reward = progress_reward * 10 - distance_to_goal / 10  # Scale progress and penalize distance

        # Update previous distance
        self.previous_distance_to_goal = distance_to_goal

        # Check termination conditions
        terminated = bool(distance_to_goal < 0.01)  # Ensure boolean type for success condition
        if terminated:
            self.success_count += 1  # Increment success counter

        truncated = bool(self.steps >= self.max_steps)  # Ensure boolean type for max steps

        # Increment the step counter and cumulative reward
        self.steps += 1
        self.cumulative_reward += reward

        # Log metrics to Wandb if enabled
        if self.use_wandb:
            wandb.log({
                "reward": reward,
                "distance_to_goal": distance_to_goal,
                "steps": self.steps,
                "terminated": int(terminated),
                "success_count": self.success_count,
            })

            # Log cumulative reward at the end of the episode
            if terminated or truncated:
                wandb.log({"cumulative_reward": self.cumulative_reward})
                self.cumulative_reward = 0

        return observation, reward, terminated, truncated, {}


    def render(self, mode="human"):
        if self.render:
            pass  # Add rendering logic if needed

    def close(self):
        self.sim.close()
