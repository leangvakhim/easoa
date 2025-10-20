import numpy as np
import math
import random
import time
from tqdm import tqdm
from plotingeasosa import visualize_network
# Make sure to import the new vectorized function!
from calculateeasoa import total_coverage_prob_vectorized, total_coverage, total_coverage_prob
from easoa import easoa
from multiprocessing import Pool

def run_failure_simulation(failure_rate):
    """
    Runs a single node failure simulation for a given failure rate.
    """
    print(f"\n--- Running Simulation for {failure_rate*100}% Node Failure ---")

    # --- Initial Setup ---
    random.seed(123)
    N = 20
    D = 50
    MaxIter = 500
    PopSize = 50
    sensing_radius = 10.0
    w1 = 1 - 1e-61
    w2 = w3 = 5e-62
    random_targets = [(random.randrange(0, D), random.randrange(0, D)) for _ in range(100)]
    random_targets_np = np.array(random_targets) # Convert to NumPy array once

    # --- 1. Initial Optimization ---
    print("Running initial EASOA optimization...")
    start_time = time.time()
    optimized_sensor_positions = easoa(N, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets)
    computation_time = time.time() - start_time
    print("Initial optimization complete.")

    # --- 2. Calculate Initial Coverage (Vectorized) ---
    initial_coverage_probs = total_coverage_prob_vectorized(optimized_sensor_positions, random_targets_np, sensing_radius)
    initial_coverage = np.sum(initial_coverage_probs)
    initial_total_coverage = total_coverage(initial_coverage, len(random_targets))
    print(f"Initial Network Coverage: {initial_total_coverage:.4f}")

    # --- 3. Simulate Node Failure ---
    num_failed_nodes = int(N * failure_rate)
    print(f"Simulating failure of {num_failed_nodes} out of {N} sensors.")
    failed_indices = random.sample(range(N), num_failed_nodes)
    remaining_sensor_positions = np.delete(optimized_sensor_positions, failed_indices, axis=0)

    # --- 4. Calculate Coverage Retention (Vectorized) ---
    coverage_after_failure_probs = total_coverage_prob_vectorized(remaining_sensor_positions, random_targets_np, sensing_radius)
    coverage_after_failure_sum = np.sum(coverage_after_failure_probs)
    coverage_after_failure = total_coverage(coverage_after_failure_sum, len(random_targets))

    network_coverage_retention = (coverage_after_failure / initial_total_coverage) * 100
    print(f"Network Coverage After Failure: {coverage_after_failure:.4f}")
    print(f"Network Coverage Retention: {network_coverage_retention:.2f}%")

    # --- 5. Simulate Recovery ---
    print("Running EASOA for recovery...")
    recovery_start_time = time.time()
    easoa(N - num_failed_nodes, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets)
    recovery_time = time.time() - recovery_start_time
    print("Recovery simulation complete.")

    # ... (rest of the function and the script remains the same)
    # --- 6. Display Results for this scenario ---
    print("\n--- Results ---")
    print(f"WS failure rate: {failure_rate*100}%")
    print(f"Network coverage retention: {network_coverage_retention:.2f}%")
    print(f"Recovery time/s: {recovery_time:.2f}s")
    print(f"Node replacement rate/%: {failure_rate*100}%") # As per the paper's context
    print(f"Computation time/s: {computation_time:.2f}s")
    print("---------------")

    return {
        'failure_rate': failure_rate,
        'retention': network_coverage_retention,
        'recovery_time': recovery_time,
        'computation_time': computation_time
    }


if __name__ == "__main__":
    failure_rates = [0.10, 0.20, 0.30]
    # Use a Pool for parallel processing
    with Pool() as pool:
        results = pool.map(run_failure_simulation, failure_rates)

    print("\n\n--- All Simulations Complete ---")
    for result in results:
        print(f"Results for {result['failure_rate']*100}% failure:")
        print(f"  - Coverage Retention: {result['retention']:.2f}%")
        print(f"  - Recovery Time: {result['recovery_time']:.2f}s")
        print(f"  - Initial Computation Time: {result['computation_time']:.2f}s")