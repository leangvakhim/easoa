import math
import random
import numpy as np
from scipy.spatial.distance import cdist, pdist

## Equation 1
def single_sensor_prob(distance, sensor_radius):

    return (distance <= sensor_radius).astype(int)

def total_coverage_prob_vectorized(sensors, targets, sensor_radius):
    distances = cdist(sensors, targets) # Shape: (num_sensors, num_targets)
    prob_si_detect_pj = single_sensor_prob(distances, sensor_radius)
    failure_prob = 1 - prob_si_detect_pj
    product_failure = np.prod(failure_prob, axis=0)
    total_prob = 1 - product_failure
    return total_prob

def total_coverage_prob(sensors, point, sensor_radius):
    point = np.array(point).reshape(1, -1)
    # Calculate distances from all sensors to the point
    distances = cdist(sensors, point)
    product_failure = 1.0

    for dist in distances:
        prob_si_detect_pj = single_sensor_prob(dist, sensor_radius)
        failure_prob = 1 - prob_si_detect_pj
        product_failure *= failure_prob

    total_prob = 1 - product_failure
    return total_prob

# Equation 2
def total_coverage(total_sum_coverage, total_monitoring_point):
    return total_sum_coverage / total_monitoring_point

# Equation 3
def reverse_elite_selection(x, x_min, x_max):
    return x_max + x_min - x

# Equation 4
def update_reverse_elite(x, x_prime, w1, w2, w3, sensing_radius):
    if fitness_value(x_prime, w1, w2, w3, sensing_radius) < fitness_value(x, w1, w2, w3, sensing_radius):
        return x_prime
    else:
        return x

# Equation 5
def brightness_driven_perturbation(x_i, x_j, beta=0.5, alpha=0.5, gamma=0.7):
    distance_square = np.sum((x_i - x_j)**2)
    theta = np.random.randn(*x_i.shape) * 0.1
    perturbation = beta * np.exp(-gamma * distance_square)
    random_term = alpha * theta
    return x_i + perturbation + random_term

# Equation 6
def update_attraction_coefficient(beta_initial, k, k_max):
    beta = beta_initial * (1 - (k / k_max))
    return beta

# Equation 7
def dynamic_warning_update(x, x_best, delta=0.3):
    r = random.random()
    x_new = x + delta * (r * x_best - x)
    return x_new

def fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets):
    energy = 0
    random_targets_np = np.array(random_targets)
    all_coverage_probs = total_coverage_prob_vectorized(sparrow, random_targets_np, sensing_radius)
    coverage = np.mean(all_coverage_probs)
    grid_size = 5
    cell_size = deployment_area / grid_size

    counts, _, _ = np.histogram2d(
        sparrow[:, 0], # All sensor x-coordinates
        sparrow[:, 1], # All sensor y-coordinates
        # Define the bins for the 5x5 grid
        bins=[np.arange(0, deployment_area + cell_size, cell_size),
              np.arange(0, deployment_area + cell_size, cell_size)]
    )
    dvar = np.var(counts)
    max_possible_variance = (deployment_area**2) / 12.0
    normalized_dvar = dvar / max_possible_variance
    fitness = (w1 * coverage) - (w2 * normalized_dvar) - (w3 * energy)
    return fitness

def is_near_boundary(sparrow, deployment_area, threshold=1.0):
    for sensor in sparrow:
        if np.any(sensor < threshold) or np.any(sensor > (deployment_area - threshold)):
            return True
    return False

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))