import os
import time
import numpy as np
from sim_class import Simulation

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize simulation
sim = Simulation(num_agents=1, render=True)

# Helper to get pipette position
def get_pipette_position(sim):
    states = sim.get_states()
    robot_id = list(states.keys())[0]
    return np.array(states[robot_id]['pipette_position'], dtype=float)

# Define 8 directional velocity vectors for corner exploration
corner_velocities = [
    [-0.9, -0.9, -0.9, 0],
    [ 0.9, -0.9, -0.9, 0],
    [-0.9,  0.9, -0.9, 0],
    [ 0.9,  0.9, -0.9, 0],
    [-0.9, -0.9,  0.9, 0],
    [ 0.9, -0.9,  0.9, 0],
    [-0.9,  0.9,  0.9, 0],
    [ 0.9,  0.9,  0.9, 0],
]

# Move to each corner and record pipette positions
pipette_positions = []

for i, corner_velocity in enumerate(corner_velocities):
    print(f"Moving to corner {i+1} with velocity: {corner_velocity[:3]}")
    sim.set_start_position(0, 0, 0)

    for _ in range(130):
        sim.run([corner_velocity])
        time.sleep(1/240)

    pipette_pos = get_pipette_position(sim)
    pipette_positions.append(pipette_pos)
    print(f"Corner {i+1} Pipette Position: {pipette_pos}")

# Extract min and max from visited corners
positions_np = np.array(pipette_positions)
low_bound = np.min(positions_np, axis=0)
high_bound = np.max(positions_np, axis=0)

# Write detected bounds to README
envelope_text = "## Working Envelope of the Pipette\n\n"
envelope_text += "| Axis | Lower Limit | Upper Limit |\n"
envelope_text += "|------|-------------|-------------|\n"
envelope_text += f"| X | {low_bound[0]} | {high_bound[0]} |\n"
envelope_text += f"| Y | {low_bound[1]} | {high_bound[1]} |\n"
envelope_text += f"| Z | {low_bound[2]} | {high_bound[2]} |\n"

readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r") as f:
        content = f.read()
else:
    content = ""

if "## Working Envelope of the Pipette" in content:
    before, _, after = content.partition("## Working Envelope of the Pipette")
    after = after.split("##", 1)[1] if "##" in after else ""
    new_content = before + envelope_text + "##" + after
else:
    new_content = content + "\n\n" + envelope_text

with open(readme_path, "w") as f:
    f.write(new_content)

print("README.md updated with working envelope.")

# Close simulation
sim.close()