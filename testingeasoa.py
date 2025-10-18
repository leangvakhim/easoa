import numpy as np
import math
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from plotingeasosa import visualize_network
from calculateeasoa import total_coverage_prob

# def total_coverage(all_point_probabilities, total_monitoring_point):
#     return sum(all_point_probabilities) / len(total_monitoring_point)

# Example usage
sensor_positions = [
    (5, 5), (15, 5), (25, 5), (35, 5), (45, 5),
    (5, 15), (15, 15), (25, 15), (35, 15), (45, 15),
    (5, 25), (15, 25), (25, 25), (35, 25), (45, 25),
    (5, 35), (15, 35), (25, 35), (35, 35), (45, 35),
]
sensing_radius = 5.0
total_coverage = 0.0

random_targets = [(random.randrange(0, 40), random.randrange(0, 40)) for _ in range(20)]

for point in random_targets:
    coverage_prob = total_coverage_prob(sensor_positions, point, sensing_radius)
    total_coverage += coverage_prob
    print(f"Total Coverage Probability for Target Point {point}: {coverage_prob:.6f}")
    print("-" * 20)

# Equation 2
total_value_coverage = total_coverage / len(random_targets)
print(f"Total Network Coverage: {total_value_coverage:.6f}")



# coverage_prob_1 = total_coverage_prob(sensor_positions, target_point_1, sensing_radius)
# coverage_prob_2 = total_coverage_prob(sensor_positions, target_point_2, sensing_radius)

# print(f"Total Coverage Probability for Point 1: {coverage_prob_1:.6f}")
# print("-" * 20)
# print(f"Total Coverage Probability for Point 2: {coverage_prob_2:.6f}")



# Generate the visualization
# visualize_network(sensor_positions, random_targets, sensing_radius)




# def perception_probability_at_a_point(probabilities, si, pj):
#     # P(n, r) = n! / (n - r)!
#     probab_result = math.factorial(si) / math.factorial(si - pj)
#     return 1 - np.prod([1 - p for p in probab_result])

# value = [0.9, 0.9, 0.9]
# random.seed(123)
# random_numbers = []
# for i in range(20):
#     random_numbers.append(random.random())
# # print("Random Numbers:", random_numbers)
# results = perception_probability_at_a_point(random_numbers)
# print(f"Perception Probability at a Point: {results:.6f}")

## Equation 2
# def total_coverage(all_point_probabilities):
#     if not all_point_probabilities:
#         return 0
#     return sum(all_point_probabilities) / len(all_point_probabilities)

# network_coverage = total_coverage(random_numbers)
# print(f"Total Network Coverage: {network_coverage:.6f}")

## Equation 3 & 4
# def reverse_elite_selection(x, x_min, x_max):
#     return x_max + x_min - x