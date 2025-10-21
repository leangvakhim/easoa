# import numpy as np
# import math
# import random
# import time
# from tqdm import tqdm
# from plotingeasosa import visualize_network
# # Make sure to import the new vectorized function!
# from calculateeasoa import total_coverage_prob_vectorized, total_coverage, total_coverage_prob
# from easoa import easoa
# from multiprocessing import Pool

# def run_failure_simulation(failure_rate):
#     """
#     Runs a single node failure simulation for a given failure rate.
#     """
#     print(f"\n--- Running Simulation for {failure_rate*100}% Node Failure ---")

#     # --- Initial Setup ---
#     random.seed(123)
#     N = 20
#     D = 50
#     MaxIter = 500
#     PopSize = 50
#     sensing_radius = 10.0
#     w1 = 1 - 1e-61
#     w2 = w3 = 5e-62
#     random_targets = [(random.randrange(0, D), random.randrange(0, D)) for _ in range(100)]
#     random_targets_np = np.array(random_targets) # Convert to NumPy array once

#     # --- 1. Initial Optimization ---
#     print("Running initial EASOA optimization...")
#     start_time = time.time()
#     optimized_sensor_positions = easoa(N, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets)
#     computation_time = time.time() - start_time
#     print("Initial optimization complete.")

#     # --- 2. Calculate Initial Coverage (Vectorized) ---
#     initial_coverage_probs = total_coverage_prob_vectorized(optimized_sensor_positions, random_targets_np, sensing_radius)
#     initial_coverage = np.sum(initial_coverage_probs)
#     initial_total_coverage = total_coverage(initial_coverage, len(random_targets))
#     print(f"Initial Network Coverage: {initial_total_coverage:.4f}")

#     # --- 3. Simulate Node Failure ---
#     num_failed_nodes = int(N * failure_rate)
#     print(f"Simulating failure of {num_failed_nodes} out of {N} sensors.")
#     failed_indices = random.sample(range(N), num_failed_nodes)
#     remaining_sensor_positions = np.delete(optimized_sensor_positions, failed_indices, axis=0)

#     # --- 4. Calculate Coverage Retention (Vectorized) ---
#     coverage_after_failure_probs = total_coverage_prob_vectorized(remaining_sensor_positions, random_targets_np, sensing_radius)
#     coverage_after_failure_sum = np.sum(coverage_after_failure_probs)
#     coverage_after_failure = total_coverage(coverage_after_failure_sum, len(random_targets))

#     network_coverage_retention = (coverage_after_failure / initial_total_coverage) * 100
#     print(f"Network Coverage After Failure: {coverage_after_failure:.4f}")
#     print(f"Network Coverage Retention: {network_coverage_retention:.2f}%")

#     # --- 5. Simulate Recovery ---
#     print("Running EASOA for recovery...")
#     recovery_start_time = time.time()
#     easoa(N - num_failed_nodes, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets)
#     recovery_time = time.time() - recovery_start_time
#     print("Recovery simulation complete.")

#     # ... (rest of the function and the script remains the same)
#     # --- 6. Display Results for this scenario ---
#     print("\n--- Results ---")
#     print(f"WS failure rate: {failure_rate*100}%")
#     print(f"Network coverage retention: {network_coverage_retention:.2f}%")
#     print(f"Recovery time/s: {recovery_time:.2f}s")
#     print(f"Node replacement rate/%: {failure_rate*100}%") # As per the paper's context
#     print(f"Computation time/s: {computation_time:.2f}s")
#     print("---------------")

#     return {
#         'failure_rate': failure_rate,
#         'retention': network_coverage_retention,
#         'recovery_time': recovery_time,
#         'computation_time': computation_time
#     }


# if __name__ == "__main__":
#     failure_rates = [0.10, 0.20, 0.30]
#     # Use a Pool for parallel processing
#     with Pool() as pool:
#         results = pool.map(run_failure_simulation, failure_rates)

#     print("\n\n--- All Simulations Complete ---")
#     for result in results:
#         print(f"Results for {result['failure_rate']*100}% failure:")
#         print(f"  - Coverage Retention: {result['retention']:.2f}%")
#         print(f"  - Recovery Time: {result['recovery_time']:.2f}s")
#         print(f"  - Initial Computation Time: {result['computation_time']:.2f}s")
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

# Make sure the fitness_value in calculateeasoa.py is updated with the histogram logic!

def run_failure_simulation(failure_rate):
    """
    Runs a single node failure simulation for a given failure rate.
    """
    print(f"\n--- Running Simulation for {failure_rate*100}% Node Failure ---")

    # --- Initial Setup ---
    # Seeding for reproducibility (though multiprocessing can be tricky)
    random.seed(123)
    np.random.seed(123)

    N = 20
    D = 50
    MaxIter = 500
    PopSize = 50 # This is correct as per paper
    sensing_radius = 10.0

    # --- FIX 1: Correct Weights ---
    # We MUST use balanced weights to get a uniform (robust) initial deployment
    # The paper's goal is a "trade-off"
    # w1 = 1 - 1e-61  # <- This was the bug
    # w2 = w3 = 5e-62 # <- This was the bug

    w1 = 0.7  # Prioritize coverage
    w2 = 0.3  # But also heavily value uniformity
    w3 = 0.0  # Ignore energy for this test


    # --- FIX 2: Correct Monitoring Points ---
    # Must use the 625-point grid, not 100 random points.
    # random_targets = [(random.randrange(0, D), random.randrange(0, D)) for _ in range(100)] # <- This was the bug

    grid_points = np.linspace(0, D, 25) # 25x25 grid
    random_targets = []
    for x_coord in grid_points:
        for y_coord in grid_points:
            random_targets.append((x_coord, y_coord))

    random_targets_np = np.array(random_targets) # Convert to NumPy array once
    print(f"Using {len(random_targets)} grid-based monitoring points.")

    # --- 1. Initial Optimization (to get a good, uniform deployment) ---
    print(f"Running initial EASOA optimization (w1={w1}, w2={w2})...")
    start_time = time.time()
    optimized_sensor_positions = easoa(N, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets)
    computation_time = time.time() - start_time
    print("Initial optimization complete.")

    # --- 2. Calculate Initial Coverage (Vectorized) ---
    initial_coverage_probs = total_coverage_prob_vectorized(optimized_sensor_positions, random_targets_np, sensing_radius)
    initial_coverage_sum = np.sum(initial_coverage_probs)
    initial_total_coverage_percent = total_coverage(initial_coverage_sum, len(random_targets)) * 100
    print(f"Initial Network Coverage: {initial_total_coverage_percent:.2f}%")

    # --- 3. Simulate Node Failure ---
    num_failed_nodes = int(N * failure_rate)
    print(f"Simulating failure of {num_failed_nodes} out of {N} sensors.")
    # Ensure this is random for each run if __name__ == "__main__" is used
    failed_indices = random.sample(range(N), num_failed_nodes)
    remaining_sensor_positions = np.delete(optimized_sensor_positions, failed_indices, axis=0)

    # --- 4. Calculate Coverage Retention (Vectorized) ---
    coverage_after_failure_probs = total_coverage_prob_vectorized(remaining_sensor_positions, random_targets_np, sensing_radius)
    coverage_after_failure_sum = np.sum(coverage_after_failure_probs)
    coverage_after_failure_percent = total_coverage(coverage_after_failure_sum, len(random_targets)) * 100

    # Retention is the % of the *original* coverage that remains
    network_coverage_retention = (coverage_after_failure_percent / initial_total_coverage_percent) * 100

    # The paper's "Network coverage retention/%" is likely just the coverage *after* failure.
    # Let's calculate both.
    coverage_after_failure_absolute = coverage_after_failure_percent

    print(f"Network Coverage After Failure: {coverage_after_failure_absolute:.2f}%")
    # print(f"Network Coverage Retention (relative): {network_coverage_retention:.2f}%")

    # --- 5. Simulate Recovery ---
    # The paper's "Recovery time" is the time to re-optimize the network.
    print("Running EASOA for recovery (re-optimizing remaining nodes)...")
    recovery_start_time = time.time()
    # Re-run optimization with the nodes that are left
    easoa(N - num_failed_nodes, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets)
    recovery_time = time.time() - recovery_start_time
    print("Recovery simulation complete.")

    # --- 6. Node Replacement Rate ---
    # The paper's definition is: "percentage of nodes that are replaced or redeployed"
    # The table values (9.8%, 12.5%, 15.8%) do not match the failure rates (10%, 20%, 30%).
    # This implies a more complex "recovery" where new nodes are added.
    # Your current code *cannot* calculate this metric as defined.
    node_replacement_rate = "N/A"

    # --- 7. Display Results for this scenario ---
    print("\n--- Results ---")
    print(f"WS failure rate: {failure_rate*100}%")
    # Table 1's "Network coverage retention" is the absolute coverage *after* failure.
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

    # Using Pool() can be unreliable with random seeds.
    # For a stable test, let's run them one by one.
    # with Pool() as pool:
    #     results = pool.map(run_failure_simulation, failure_rates)

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

    print(f"My Retention %      | {r[0.1]['retention']:.2f}        | {r[0.2]['retention']:.2f}        | {r[0.3]['retention']:.2f}        |")
    print(f"Paper Retention %    | {ret_paper[0.1]}       | {ret_paper[0.2]}       | {ret_paper[0.3]}       |")
    print(f"My Recovery s        | {r[0.1]['recovery_time']:.2f}        | {r[0.2]['recovery_time']:.2f}        | {r[0.3]['recovery_time']:.2f}        |")
    print(f"Paper Recovery s     | {rec_paper[0.1]}         | {rec_paper[0.2]}         | {rec_paper[0.3]}         |")
    print(f"My Computation s     | {r[0.1]['computation_time']:.2f}        | {r[0.2]['computation_time']:.2f}        | {r[0.3]['computation_time']:.2f}        |")
    print(f"Paper Computation s  | {comp_paper[0.1]}         | {comp_paper[0.2]}         | {comp_paper[0.3]}         |")