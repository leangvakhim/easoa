import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from calculateeasoa import total_coverage_prob
import math

def visualize_network(sensors, targets, sensor_radius):
    """
    Creates a plot of the sensor network using Matplotlib.

    Args:
        sensors (list of tuples): List of sensor (x, y) coordinates.
        targets (list of tuples): List of target (x, y) coordinates.
        sensor_radius (float): The sensing radius of the sensors.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Plot each sensor and its coverage radius
    for i, sensor_pos in enumerate(sensors):
        # Plot the sensor location
        ax.plot(sensor_pos[0], sensor_pos[1], 'bo', markersize=10, label='Sensor' if i == 0 else "")
        # Create and add the circle representing the sensing radius
        coverage_circle = Circle(sensor_pos, sensor_radius, color='green', alpha=0.15)
        ax.add_patch(coverage_circle)

    # 2. Plot each target point, colored by its coverage status
    for i, target_pos in enumerate(targets):
        is_covered = total_coverage_prob(sensors, target_pos, sensor_radius) > 0
        color = 'green' if is_covered else 'red'
        label = 'Covered Target' if is_covered and 'Covered Target' not in ax.get_legend_handles_labels()[1] else \
                'Uncovered Target' if not is_covered and 'Uncovered Target' not in ax.get_legend_handles_labels()[1] else ""
        ax.plot(target_pos[0], target_pos[1], 'o', color=color, markersize=8, label=label)

    # 3. Set up the plot aesthetics
    ax.set_title('Wireless Sensor Network Coverage Visualization', fontsize=16)
    ax.set_xlabel('X-coordinate (m)', fontsize=12)
    ax.set_ylabel('Y-coordinate (m)', fontsize=12)
    ax.set_aspect('equal', adjustable='box') # Ensures circles are not warped
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Set plot limits to see everything clearly
    all_x = [s[0] for s in sensors] + [t[0] for t in targets]
    all_y = [s[1] for s in sensors] + [t[1] for t in targets]
    ax.set_xlim(min(all_x) - sensor_radius, max(all_x) + sensor_radius)
    ax.set_ylim(min(all_y) - sensor_radius, max(all_y) + sensor_radius)

    plt.show()