# PID Runner Script for Random Target Testing with Error Analysis
# Author: [Your Name]
# Date: [Date]

import numpy as np
import random
import time
from pid_controller_ot2 import PIDControllerSystem

# === Bounds ===
low_bound = np.array([-0.187, -0.1705, 0.1695])
high_bound = np.array([0.253, 0.2195, 0.2908])

# === Best Performing PID Gains ===
best_gains = {
    'x': (2.0, 0.1, 0.05),
    'y': (2.0, 0.1, 0.05),
    'z': (2.5, 0.1, 0.05)
}

# === Generate Random Target Within Bounds ===
def generate_random_target():
    return np.random.uniform(low=low_bound, high=high_bound)

# === Run Multiple Tests with Error Reporting ===
def run_tests(n=5, tolerance=0.001):
    controller = PIDControllerSystem(best_gains)
    total_errors = []

    for i in range(n):
        target = generate_random_target()
        print(f"\nTest {i + 1}: Target = {target}")

        start_time = time.time()
        success = controller.move_to(target, tolerance=tolerance)
        end_time = time.time()

        # Calculate final position error
        final_position = np.array([
            controller.x.position,
            controller.y.position,
            controller.z.position
        ])
        error_vector = np.abs(target - final_position)
        total_errors.append(error_vector)

        print(f"Final Position = {final_position}")
        print(f"Error (abs) = {error_vector} (meters)")
        print(f"Within tolerance {tolerance} m: {np.all(error_vector <= tolerance)}")
        print(f"Time taken: {end_time - start_time:.2f}s")

    # Summary
    total_errors = np.array(total_errors)
    mean_error = np.mean(total_errors, axis=0)
    max_error = np.max(total_errors, axis=0)

    print("\n=== Error Summary ===")
    print(f"Mean Error (X, Y, Z): {mean_error} m")
    print(f"Max Error (X, Y, Z): {max_error} m")

if __name__ == "__main__":
    # Run with 1mm accuracy requirement (8.7 D)
    run_tests(n=5, tolerance=0.001)

    # To test 10mm accuracy (8.7 C), change tolerance to 0.01