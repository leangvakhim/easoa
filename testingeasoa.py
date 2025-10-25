import numpy as np
import math
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from plotingeasosa import visualize_network
from calculateeasoa import total_coverage, total_coverage_prob_vectorized
from easoa import easoa

random.seed(123)
# N: Number of sensor nodes
N = 20
# D: Deployment area (D x D)
D = 50
# MaxIter: Maximum number of iterations
MaxIter = 500
# w1,w2,w3: Weights for coverage, uniformity, and energy consumption in fitness function
w1, w2, w3 = 0.8, 0.1, 0.1
# PopSize: Population size
PopSize = 50
# sensing_radius: Sensing radius of each sensor
sensing_radius = 10.0
coverage = 0.0
grid_size = 5 # Must match calculateeasoa.py

grid_points = np.linspace(0, D, 25)
random_targets = []
for x_coord in grid_points:
    for y_coord in grid_points:
        random_targets.append((x_coord, y_coord))

# --- Pre-calculate arguments ---
random_targets_np = np.array(random_targets)
max_dvar_approx = np.var([N] + [0]*(grid_size**2 - 1))
if max_dvar_approx == 0: max_dvar_approx = 1
# --- End Pre-calculation ---

print("Running EASOA to optimize sensor positions...")
# --- Pass new args to easoa ---
optimized_sensor_positions = easoa(N, D, MaxIter, PopSize, sensing_radius, w1, w2, w3, random_targets_np, max_dvar_approx)
print("Optimization complete.")

all_coverage_probs = total_coverage_prob_vectorized(optimized_sensor_positions, random_targets_np, sensing_radius)
coverage = np.sum(all_coverage_probs)

total_value_coverage = total_coverage(coverage, len(random_targets))
print(f"Total Network Coverage with Optimized Positions: {total_value_coverage:.6f}")

# Generate the visualization
# visualize_network(optimized_sensor_positions, random_targets, sensing_radius)