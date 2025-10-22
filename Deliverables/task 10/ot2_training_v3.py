import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper_v3 import OT2Env  # Import your custom Gym wrapper
import wandb

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize Wandb
wandb.init(
    project="OT2_RL_Training",
    name="PPO_OT2_training_v3",
    sync_tensorboard=True,
    config={
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "batch_size": 512,
        "gamma": 0.99,
        "total_timesteps": 5_000_000,
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
    tensorboard_log="./ppo_ot2_tensorboard_v3/",
    device="cpu"
)

# Train the model
model.learn(
    total_timesteps=wandb.config.total_timesteps,
    callback=WandbCallback(gradient_save_freq=100, verbose=2)
)

# Save the trained model
model_path = "ppo_ot2_model_v3.zip"
model.save(model_path)

# Save the model to Wandb
artifact = wandb.Artifact(name="ppo_ot2_model_v3", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

print("Model training completed and saved!")
