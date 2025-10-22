import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper_v4 import OT2Env  # Import your custom Gym wrapper
import wandb

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize Wandb
wandb.init(
    project="OT2_RL_Training",
    name="PPO_OT2_training_v4",
    sync_tensorboard=True,
    config={
        "policy": "MlpPolicy",
        "learning_rate": 5e-5,     # Smaller learning rate for stable long-term training
        "batch_size": 1024,        # Larger batch size for better updates
        "gamma": 0.98,             # Slightly lower gamma for faster learning
        "total_timesteps": 20_000_000,  # 20 million training steps
    }
)

# Create the environment
env = OT2Env(render=False, max_steps=1000)

# Check the environment
check_env(env)

# Define the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=wandb.config.learning_rate,
    batch_size=wandb.config.batch_size,
    gamma=wandb.config.gamma,
    verbose=1,
    tensorboard_log="./ppo_ot2_tensorboard_v4/",
    device="cpu"
)

# Train the model
model.learn(
    total_timesteps=wandb.config.total_timesteps,
    callback=WandbCallback(gradient_save_freq=100, verbose=2)
)

# Save the trained model
model_path = "ppo_ot2_model_v4.zip"
model.save(model_path)

# Save the model to Wandb
artifact = wandb.Artifact(name="ppo_ot2_model_v4", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

print("Model training completed and saved!")
