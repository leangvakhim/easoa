import math
import random
import numpy as np
from scipy.spatial.distance import cdist

## Equation 1
# def single_sensor_prob(sensor_position, point_position, sensor_radius):
# def single_sensor_prob(distance, sensor_radius):

#     # distance = math.dist(sensor_position, point_position)

#     if distance <= sensor_radius:
#         return 1
#     else:
#         return 0

def single_sensor_prob(distance, sensor_radius):
    # This function is already efficient, but we will call it from a vectorized context.
    # It returns 1 if distance <= sensor_radius, and 0 otherwise.
    return (distance <= sensor_radius).astype(int)

def total_coverage_prob_vectorized(sensors, targets, sensor_radius):
    """
    Calculates the coverage probability for all targets from all sensors in a vectorized manner.
    """
    # Calculate all pairwise distances between sensors and targets
    distances = cdist(sensors, targets) # Shape: (num_sensors, num_targets)

    # Calculate the probability of each sensor detecting each target
    prob_si_detect_pj = single_sensor_prob(distances, sensor_radius) # Shape: (num_sensors, num_targets)

    # Calculate the probability of each sensor *failing* to detect each target
    failure_prob = 1 - prob_si_detect_pj # Shape: (num_sensors, num_targets)

    # Calculate the product of failure probabilities for each target (across all sensors)
    # This gives the probability that a target is NOT detected by ANY sensor.
    product_failure = np.prod(failure_prob, axis=0) # Shape: (num_targets,)

    # The total probability of a target being covered is 1 minus the probability of it not being covered.
    total_prob = 1 - product_failure # Shape: (num_targets,)
    return total_prob

def total_coverage_prob(sensors, point, sensor_radius):
    point = np.array(point).reshape(1, -1)
    # Calculate distances from all sensors to the point
    distances = cdist(sensors, point)
    product_failure = 1.0

    # for sensor in sensors:
    #     prob_si_detect_pj = single_sensor_prob(sensor, point, sensor_radius)
    #     failure_prob = 1 - prob_si_detect_pj
    #     product_failure *= failure_prob
    #     total_prob = 1 - product_failure

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
# def update_reverse_elite(x, x_prime):
#     if fitness_value(x_prime) < fitness_value(x):
#         return x_prime
#     else:
#         return x

# Equation 4
def update_reverse_elite(x, x_prime, w1, w2, w3, sensing_radius):
    if fitness_value(x_prime, w1, w2, w3, sensing_radius) < fitness_value(x, w1, w2, w3, sensing_radius):
        return x_prime
    else:
        return x

# Equation 5
def brightness_driven_perturbation(x_i, x_j, beta=0.5, alpha=0.5, gamma=0.7):
    # distance_square = (x_i - x_j) ** 2
    # theta = np.random.randn(*x_i.shape) * 0.1
    # theta = 0.1
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

# Equation 8
# def fitness_value(coverage, energy, dvar, omega1, omega2, omega3):
#     fitness = (omega1 * coverage) - (omega2 * dvar) - (omega3 * energy)
#     return fitness

# def fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets):
#     # Coverage
#     coverage = 0
#     # random_targets = [(random.randrange(0, deployment_area), random.randrange(0, deployment_area)) for _ in range(50)] # Using 50 random targets for evaluation
#     for point in random_targets:
#         coverage += total_coverage_prob(sparrow, point, sensing_radius)
#     coverage = total_coverage(coverage, len(random_targets))

#     # Spatial distribution variance (dvar)
#     dvar = np.var(sparrow)

#     max_possible_variance = (deployment_area**2) / 12.0
#     normalized_dvar = dvar / max_possible_variance

#     # Energy consumption (simplified as a constant for now)
#     # energy = 100
#     energy = 0

#     # fitness = (w1 * coverage) - (w2 * dvar) - (w3 * energy)
#     fitness = (w1 * coverage) - (w2 * normalized_dvar) - (w3 * energy)
#     return fitness

def fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets):
    """
    Calculates the fitness value for a single sparrow (a set of sensor positions).
    """
    # Ensure random_targets is a NumPy array for vectorized operations
    random_targets_np = np.array(random_targets)

    # 1. Coverage Calculation (Vectorized)
    # Calculate coverage for all targets at once.
    all_coverage_probs = total_coverage_prob_vectorized(sparrow, random_targets_np, sensing_radius)
    # The total coverage is the average of all individual target coverage probabilities.
    coverage = np.mean(all_coverage_probs)

    # 2. Spatial Distribution Variance (dvar)
    # This is already efficient.
    dvar = np.var(sparrow)
    max_possible_variance = (deployment_area**2) / 12.0
    normalized_dvar = dvar / max_possible_variance

    # 3. Energy Consumption
    energy = 0

    # Final fitness calculation
    fitness = (w1 * coverage) - (w2 * normalized_dvar) - (w3 * energy)
    return fitness

def is_near_boundary(sparrow, deployment_area, threshold=1.0):
    for sensor in sparrow:
        if np.any(sensor < threshold) or np.any(sensor > (deployment_area - threshold)):
            return True
    return False

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))