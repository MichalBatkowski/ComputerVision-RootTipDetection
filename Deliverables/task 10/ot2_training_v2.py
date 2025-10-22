import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper_v2 import OT2Env  # Import your custom Gym wrapper
import wandb

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize Wandb
wandb.init(
    project="OT2_RL_Training",  # Project name
    name="PPO_OT2_training_v2",  # Name for full training run
    sync_tensorboard=True,         # Sync TensorBoard logs
    config={                       # Log hyperparameters
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,     # Typical PPO learning rate
        "batch_size": 256,         # Larger batch size for stable training
        "gamma": 0.99,
        "total_timesteps": 500_000,  # 1 million training steps
    }
)

# Create the environment
env = OT2Env(render=False, max_steps=1000)

# Check the environment for compliance (optional but recommended)
check_env(env)

# Define the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=wandb.config.learning_rate,
    batch_size=wandb.config.batch_size,
    gamma=wandb.config.gamma,
    verbose=1,
    tensorboard_log="./ppo_ot2_tensorboard_full/",  # TensorBoard logs
    device="cpu"  # Force the model to use the CPU
)

# Train the model with WandbCallback
model.learn(
    total_timesteps=wandb.config.total_timesteps,
    callback=WandbCallback(gradient_save_freq=100, verbose=2)
)

# Save the trained model
model_path = "ppo_ot2_model_full.zip"
model.save(model_path)

# Create a Wandb artifact and add the model file
artifact = wandb.Artifact(name="ppo_ot2_model", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

print("Model training completed and saved!")
