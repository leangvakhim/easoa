## Equation 1
import math


def single_sensor_prob(sensor_position, point_position, sensor_radius):

    distance = math.dist(sensor_position, point_position)

    if distance <= sensor_radius:
        return 1
    else:
        return 0
def total_coverage_prob(sensors, point, sensor_radius):
    product_failure = 1.0
    for sensor in sensors:
        prob_si_detect_pj = single_sensor_prob(sensor, point, sensor_radius)
        failure_prob = 1 - prob_si_detect_pj
        product_failure *= failure_prob
        total_prob = 1 - product_failure
    return total_prob