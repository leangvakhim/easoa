import numpy as np
import math
import random
import time
from tqdm import tqdm
from plotingeasosa import visualize_network
# Make sure to import the new vectorized function!
from calculateeasoa import total_coverage_prob_vectorized, total_coverage
from easoa import easoa
from multiprocessing import Pool

# Make sure the fitness_value in calculateeasoa.py is updated!

def run_failure_simulation(failure_rate):
    print(f"\n--- Running Simulation for {failure_rate*100}% Node Failure ---")

    # --- Initial Setup ---
    random.seed(123)
    np.random.seed(123)

    N = 20
    D = 50
    MaxIter = 500
    PopSize = 50
    sensing_radius = 10.0
    w1 = 0.8
    w2 = 0.1
    w3 = 0.1
    grid_size = 5 # For dvar calculation

    # --- FIX 2: Correct Monitoring Points (as NumPy array) ---
    grid_points = np.linspace(0, D, 25) # 25x25 grid
    random_targets_list = []
    for x_coord in grid_points:
        for y_coord in grid_points:
            random_targets_list.append((x_coord, y_coord))
    # --- PRE-CALCULATE 1 ---
    random_targets_np = np.array(random_targets_list)
    print(f"Using {len(random_targets_np)} grid-based monitoring points.")

    # --- PRE-CALCULATE 2 ---
    max_dvar_approx = np.var([N] + [0]*(grid_size**2 - 1))
    if max_dvar_approx == 0: # Avoid division by zero if N=0
        max_dvar_approx = 1

    # --- 1. Initial Optimization (to get a good, uniform deployment) ---
    print(f"Running initial EASOA optimization (w1={w1}, w2={w2}) , w3={w3})")
    start_time = time.time()
    # --- Pass new args ---
    optimized_sensor_positions = easoa(N, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets_np, max_dvar_approx)
    computation_time = time.time() - start_time
    print("Initial optimization complete.")

    # --- 2. Calculate Initial Coverage (Vectorized) ---
    initial_coverage_probs = total_coverage_prob_vectorized(optimized_sensor_positions, random_targets_np, sensing_radius)
    initial_coverage_sum = np.sum(initial_coverage_probs)
    initial_total_coverage_percent = total_coverage(initial_coverage_sum, len(random_targets_np)) * 100
    print(f"Initial Network Coverage: {initial_total_coverage_percent:.2f}%")

    # --- 3. Simulate Node Failure ---
    num_failed_nodes = int(N * failure_rate)
    print(f"Simulating failure of {num_failed_nodes} out of {N} sensors.")
    failed_indices = random.sample(range(N), num_failed_nodes)
    remaining_sensor_positions = np.delete(optimized_sensor_positions, failed_indices, axis=0)
    N_recovery = N - num_failed_nodes # Number of sensors for recovery

    # --- 4. Calculate Coverage Retention (Vectorized) ---
    coverage_after_failure_probs = total_coverage_prob_vectorized(remaining_sensor_positions, random_targets_np, sensing_radius)
    coverage_after_failure_sum = np.sum(coverage_after_failure_probs)
    coverage_after_failure_percent = total_coverage(coverage_after_failure_sum, len(random_targets_np)) * 100
    coverage_after_failure_absolute = coverage_after_failure_percent
    print(f"Network Coverage After Failure: {coverage_after_failure_absolute:.2f}%")

    # --- 5. Simulate Recovery ---
    print("Running EASOA for recovery (re-optimizing remaining nodes)...")

    # --- PRE-CALCULATE 3 (for recovery) ---
    max_dvar_recovery = np.var([N_recovery] + [0]*(grid_size**2 - 1))
    if max_dvar_recovery == 0:
        max_dvar_recovery = 1

    recovery_start_time = time.time()
    # --- Pass new args for recovery run ---
    easoa(N_recovery, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets_np, max_dvar_recovery)
    recovery_time = time.time() - recovery_start_time
    print("Recovery simulation complete.")

    # --- 6. Node Replacement Rate ---
    node_replacement_rate = "N/A"

    # --- 7. Display Results for this scenario ---
    print("\n--- Results ---")
    print(f"WS failure rate: {failure_rate*100}%")
    print(f"Network coverage retention/%: {coverage_after_failure_absolute:.2f}% (Paper's EASOA: { {0.1: 91.2, 0.2: 87.5, 0.3: 83.5}[failure_rate] }%)")
    print(f"Recovery time/s: {recovery_time:.2f}s (Paper's EASOA: { {0.1: 2.7, 0.2: 3.2, 0.3: 3.7}[failure_rate] }s)")
    print(f"Node replacement rate/%: {node_replacement_rate} (Paper's EASOA: { {0.1: 9.8, 0.2: 12.5, 0.3: 15.8}[failure_rate] }%)")
    print(f"Computation time/s: {computation_time:.2f}s (Paper's EASOA: { {0.1: 2.2, 0.2: 2.4, 0.3: 2.5}[failure_rate] }s)")
    print("---------------")

    return {
        'failure_rate': failure_rate,
        'retention': coverage_after_failure_absolute,
        'recovery_time': recovery_time,
        'computation_time': computation_time
    }


if __name__ == "__main__":
    failure_rates = [0.10, 0.20, 0.30]

    results = []
    for rate in failure_rates:
        results.append(run_failure_simulation(rate))

    print("\n\n--- All Simulations Complete ---")
    print("Metric               | 10% Failure | 20% Failure | 30% Failure |")
    print("---------------------|-------------|-------------|-------------|")

    ret_paper = {0.1: 91.2, 0.2: 87.5, 0.3: 83.5}
    rec_paper = {0.1: 2.7, 0.2: 3.2, 0.3: 3.7}
    comp_paper = {0.1: 2.2, 0.2: 2.4, 0.3: 2.5}

    r = {res['failure_rate']: res for res in results}

    print(f"My Coverage (%)      | {r[0.1]['retention']:.2f}        | {r[0.2]['retention']:.2f}        | {r[0.3]['retention']:.2f}        |")
    print(f"Paper Coverage (%)    | {ret_paper[0.1]}       | {ret_paper[0.2]}       | {ret_paper[0.3]}       |")
    print(f"My Recovery (s)       | {r[0.1]['recovery_time']:.2f}        | {r[0.2]['recovery_time']:.2f}        | {r[0.3]['recovery_time']:.2f}        |")
    print(f"Paper Recovery (s)     | {rec_paper[0.1]}         | {rec_paper[0.2]}         | {rec_paper[0.3]}         |")
    print(f"My Computation (s)     | {r[0.1]['computation_time']:.2f}        | {r[0.2]['computation_time']:.2f}        | {r[0.3]['computation_time']:.2f}        |")
    print(f"Paper Computation (s)  | {comp_paper[0.1]}         | {comp_paper[0.2]}         | {comp_paper[0.3]}         |")