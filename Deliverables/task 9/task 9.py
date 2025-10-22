import os
from sim_class import Simulation
from PIL import Image

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize the simulation with one robot
sim = Simulation(num_agents=1)

# Function to test the robot's motion range along an axis
def find_axis_limits(sim, axis_index, step_size=0.1, max_steps=50):
    lower_limit = None
    upper_limit = None
    current_position = [0, 0, 0]

    # Move in the negative direction
    for step in range(max_steps):
        current_position[axis_index] -= step_size
        state = sim.run([[current_position[0], current_position[1], current_position[2], 0]])
        new_position = state.get("pipette_position", current_position)
        
        if lower_limit is None or new_position[axis_index] < lower_limit:
            lower_limit = new_position[axis_index]
        
        if not new_position or new_position[axis_index] != current_position[axis_index]:
            break

    current_position = [0, 0, 0]

    # Move in the positive direction
    for step in range(max_steps):
        current_position[axis_index] += step_size
        state = sim.run([[current_position[0], current_position[1], current_position[2], 0]])
        new_position = state.get("pipette_position", current_position)
        
        if upper_limit is None or new_position[axis_index] > upper_limit:
            upper_limit = new_position[axis_index]
        
        if not new_position or new_position[axis_index] != current_position[axis_index]:
            break

    return lower_limit or 0, upper_limit or 0


# Determine the working envelope
working_envelope = {}
axes = ['X', 'Y', 'Z']
for i, axis in enumerate(axes):
    print(f"Determining limits for {axis}-axis...")
    lower, upper = find_axis_limits(sim, i)
    working_envelope[axis] = (lower, upper)

# Create the updated working envelope content
working_envelope_content = "## Working Envelope of the Pipette\n\n"
working_envelope_content += "| Axis | Lower Limit | Upper Limit |\n"
working_envelope_content += "|------|-------------|-------------|\n"
for axis, (lower, upper) in working_envelope.items():
    working_envelope_content += f"| {axis} | {lower} | {upper} |\n"

# Read the existing README.md file
readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r") as readme_file:
        readme_content = readme_file.read()
else:
    readme_content = ""

# Replace the Working Envelope section in the README content
if "## Working Envelope of the Pipette" in readme_content:
    # Find the section and replace it
    pre_section, _, post_section = readme_content.partition("## Working Envelope of the Pipette")
    post_section = post_section.split("##", 1)[1] if "##" in post_section else ""
    updated_readme_content = pre_section + working_envelope_content + "##" + post_section
else:
    # If the section doesn't exist, append it
    updated_readme_content = readme_content + "\n\n" + working_envelope_content

# Write the updated content back to the README.md file
with open(readme_path, "w") as readme_file:
    readme_file.write(updated_readme_content)

print("README.md file has been updated with the working envelope.")