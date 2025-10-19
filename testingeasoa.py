import numpy as np
import math
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from plotingeasosa import visualize_network
from calculateeasoa import total_coverage_prob, total_coverage
from easoa import easoa

# random.seed(123)
# N: Number of sensor nodes
N = 20
# D: Deployment area (D x D)
D = 50
# MaxIter: Maximum number of iterations
MaxIter = 500
# MaxIter = 5
# w1,w2,w3: Weights for coverage, uniformity, and energy consumption in fitness function
w1, w2, w3 = 0.6, 0.2, 0.2
# PopSize: Population size
PopSize = 50
# sensing_radius: Sensing radius of each sensor
sensing_radius = 10.0

# Example usage
# sensor_positions = [
#     (5, 5), (15, 5), (25, 5), (35, 5), (45, 5),
#     (5, 15), (15, 15), (25, 15), (35, 15), (45, 15),
#     (5, 25), (15, 25), (25, 25), (35, 25), (45, 25),
#     (5, 35), (15, 35), (25, 35), (35, 35), (45, 35),
# ]
coverage = 0.0


random_targets = [(random.randrange(0, D), random.randrange(0, D)) for _ in range(N)]

print("Running EASOA to optimize sensor positions...")
optimized_sensor_positions = easoa(N, D, MaxIter, PopSize, sensing_radius, w1, w2, w3)
print("Optimization complete.")

# for point in random_targets:
#     coverage_prob = total_coverage_prob(sensor_positions, point, sensing_radius)
#     coverage += coverage_prob
    # print(f"Total Coverage Probability for Target Point {point}: {coverage_prob:.6f}")
    # print("-" * 20)

# total_value_coverage = total_coverage(coverage, len(random_targets))
# print(f"Total Network Coverage: {total_value_coverage:.6f}")

for point in random_targets:
    coverage_prob = total_coverage_prob(optimized_sensor_positions, point, sensing_radius)
    coverage += coverage_prob

total_value_coverage = total_coverage(coverage, len(random_targets))
print(f"Total Network Coverage with Optimized Positions: {total_value_coverage:.6f}")

# Generate the visualization
visualize_network(optimized_sensor_positions, random_targets, sensing_radius)
