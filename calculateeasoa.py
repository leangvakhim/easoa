import math
import random
import numpy as np
from scipy.spatial.distance import cdist

## Equation 1
# def single_sensor_prob(sensor_position, point_position, sensor_radius):
def single_sensor_prob(distance, sensor_radius):

    # distance = math.dist(sensor_position, point_position)

    if distance <= sensor_radius:
        return 1
    else:
        return 0
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
def update_reverse_elite(x, x_prime):
    if fitness_value(x_prime) < fitness_value(x):
        return x_prime
    else:
        return x

# Equation 5
def brightness_driven_perturbation(x_i, x_j, beta=0.5, alpha=0.5, gamma=0.7):
    distance_square = (x_i - x_j) ** 2
    # theta = np.random.randn(*x_i.shape) * 0.1
    theta = 0.1
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
def fitness_value(coverage, energy, dvar, omega1, omega2, omega3):
    fitness = (omega1 * coverage) - (omega2 * dvar) - (omega3 * energy)
    return fitness