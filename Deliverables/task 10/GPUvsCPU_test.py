import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper_v2 import OT2Env  # Import your custom Gym wrapper
import wandb
import torch

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize Wandb
wandb.init(
    project="OT2_RL_Training",
    name="PPO_OT2_Run_CPU_vs_GPU",  # Name for this experiment
    sync_tensorboard=True,
    config={
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "total_timesteps": 10_000,
    }
)

# Create the environment
env = OT2Env(render=False, max_steps=1000)

# Check the environment for compliance
check_env(env)

# Function to train and time the model
def train_model(device):
    # Log device being used
    print(f"Training with device: {device}")
    wandb.log({"device": device})

    # Define the PPO model with the specified device
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        gamma=wandb.config.gamma,
        verbose=1,
        tensorboard_log=f"./ppo_ot2_tensorboard_{device}/",
        device=device
    )

    # Start timing
    start_time = time.time()

    # Train the model
    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        callback=WandbCallback(gradient_save_freq=100, verbose=2)
    )

    # End timing
    end_time = time.time()

    # Save the trained model
    model_path = f"ppo_ot2_model_{device}.zip"
    model.save(model_path)

    # Create and log the model artifact
    artifact = wandb.Artifact(name=f"ppo_model_{device}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    print(f"Training completed on {device} in {end_time - start_time:.2f} seconds.")
    return end_time - start_time



# Run training on CPU
cpu_runtime = train_model("cpu")

# Run training on GPU if available
if torch.cuda.is_available():
    gpu_runtime = train_model("cuda")
else:
    print("GPU not available, skipping GPU training.")
    gpu_runtime = None

# Log the runtimes
print(f"CPU training runtime: {cpu_runtime:.2f} seconds")
if gpu_runtime is not None:
    print(f"GPU training runtime: {gpu_runtime:.2f} seconds")
    speedup = cpu_runtime / gpu_runtime
    print(f"Speedup using GPU: {speedup:.2f}x")
    wandb.log({"CPU_runtime": cpu_runtime, "GPU_runtime": gpu_runtime, "GPU_speedup": speedup})
else:
    wandb.log({"CPU_runtime": cpu_runtime, "GPU_runtime": "N/A"})

print("Comparison complete.")
