# pid_pybullet_runner.py
# This script integrates the PID controller with the provided PyBullet Simulation class.
# It moves the pipette to random target locations and measures the final error.
#
# Requirements:
#  - The 'simulation.py' file with the Simulation class in the same folder.
#  - 'pid_controller_ot2.py' containing the PID controller logic.
#
# Author: Michal Batkowski
# Date: 2025-04-01

import numpy as np
import time
import pybullet as p
from sim_class import Simulation
from PID_Controller import PID

# === Bounds ===
low_bound = np.array([-0.187, -0.1705, 0.1695])
high_bound = np.array([0.001, 0.2195, 0.2908])

# Example gains; can be tuned
best_gains = {
    'x': (5, 0.1, 0.01),
    'y': (5, 0.1, 0.01),
    'z': (5, 0.1, 0.01)
}

# Tolerance for accuracy in meters
# 0.01 m (10 mm) for ILO 8.7 C; 0.001 m (1 mm) for ILO 8.7 D
DEFAULT_TOLERANCE = 0.001

# Time step for each PID update (in seconds)
DT = 0.01


def generate_random_target():
    """Generate a random target position within the specified bounds."""
    return np.random.uniform(low=low_bound, high=high_bound)


def move_to(sim, pid_x, pid_y, pid_z, target, tolerance=DEFAULT_TOLERANCE, max_steps=500):
    """
    Move the pipette to the desired target using the PID controllers.
    :param sim: The Simulation instance.
    :param pid_x: PID controller for the X axis.
    :param pid_y: PID controller for the Y axis.
    :param pid_z: PID controller for the Z axis.
    :param target: Target (x, y, z) to reach.
    :param tolerance: Error tolerance in meters.
    :param max_steps: Maximum number of simulation steps.
    :return: (bool, final_error) success flag, and final 3D absolute error.
    """
    for step in range(max_steps):
        # Get current pipette position
        states = sim.get_states()
        # We assume a single robot (robotIds[0])
        robot_id_key = list(states.keys())[0]
        current_pos = np.array(states[robot_id_key]["pipette_position"], dtype=float)

        # Compute errors
        error_x = target[0] - current_pos[0]
        error_y = target[1] - current_pos[1]
        error_z = target[2] - current_pos[2]

        # Check if within tolerance
        if (abs(error_x) < tolerance and
            abs(error_y) < tolerance and
            abs(error_z) < tolerance):
            final_error = np.abs(target - current_pos)
            return True, final_error

        # Compute PID outputs (treated as velocities)
        VELOCITY_SCALE = 5.0  # Scale PID output if needed
        velocity_x = VELOCITY_SCALE * pid_x.compute(target[0], current_pos[0], DT)
        velocity_y = VELOCITY_SCALE * pid_y.compute(target[1], current_pos[1], DT)
        velocity_z = VELOCITY_SCALE * pid_z.compute(target[2], current_pos[2], DT)

        actions = [[velocity_x, velocity_y, velocity_z, 0]]

        # Step the simulation
        sim.run(actions, num_steps=1)

    # If we exit the loop, we failed to reach within tolerance
    final_error = np.abs(target - current_pos)
    return False, final_error


def run_random_tests(n=5, tolerance=DEFAULT_TOLERANCE):
    """
    Create a simulation, run N random tests, and summarize errors.
    """
    # Create the simulation with a single agent
    sim = Simulation(num_agents=1, render=True, rgb_array=False)

    # Initialize our PID controllers
    pid_x = PID(*best_gains['x'])
    pid_y = PID(*best_gains['y'])
    pid_z = PID(*best_gains['z'])

    total_errors = []

    for i in range(n):
        target = generate_random_target()
        print(f"\n[TEST {i+1}] Target = {target}")
        start_time = time.time()

        success, final_error = move_to(sim, pid_x, pid_y, pid_z, target, tolerance=tolerance)
        end_time = time.time()

        print(f"  Reached target: {success}")
        print(f"  Final Position Error: {final_error} m")
        print(f"  Time taken: {end_time - start_time:.2f}s")
        total_errors.append(final_error)

        
        # Optionally reset the robot to a known start position before the next test
        # sim.set_start_position(0, 0, 0)  # for example

    # Summary
    total_errors = np.array(total_errors)
    mean_error = np.mean(total_errors, axis=0)
    max_error = np.max(total_errors, axis=0)

    print("\n=== Final Error Summary ===")
    print(f"Mean Error: {mean_error} m")
    print(f"Max Error: {max_error} m")

    # Close the simulation
    sim.close()

if __name__ == "__main__":
    # Example: Run 5 random tests with 1 mm accuracy (ILO 8.7 D)
    run_random_tests(n=5, tolerance=0.001)

    # For 10 mm accuracy, use: run_random_tests(n=5, tolerance=0.01)
