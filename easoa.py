import random
import numpy as np
from tqdm import tqdm
from calculateeasoa import (
    fitness_value,
    reverse_elite_selection,
    brightness_driven_perturbation,
    update_attraction_coefficient,
    dynamic_warning_update,
    is_near_boundary
)

# --- OPTIMIZED FUNCTION SIGNATURE ---
def easoa(num_sensors, deployment_area, max_iter, population_size, sensing_radius, w1, w2, w3, random_targets_np, max_dvar_approx):
    # Initialize the population of sparrows
    sparrows = [np.random.rand(num_sensors, 2) * deployment_area for _ in range(population_size)]
    fitness_scores = np.zeros(population_size)

    for j, sparrow in enumerate(sparrows):
        # Pass new args
        fitness_scores[j] = fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets_np, max_dvar_approx)

    beta_initial = 0.5
    beta = 0.5

    safe_zone_center = np.array([deployment_area / 2.0, deployment_area / 2.0])

    for i in tqdm(range(max_iter), desc="EASOA Optimization Progress"):
        sorted_indices = np.argsort(fitness_scores)
        best_sparrow_index = sorted_indices[-1]
        best_sparrow = sparrows[best_sparrow_index]

        average_fitness = np.mean(fitness_scores)

        all_sparrows_np = np.array(sparrows)
        x_min = np.min(all_sparrows_np, axis=(0, 1))
        x_max = np.max(all_sparrows_np, axis=(0, 1))

        reverse_best_sparrow = reverse_elite_selection(best_sparrow, x_min, x_max)

        # Pass new args
        reverse_fitness = fitness_value(reverse_best_sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets_np, max_dvar_approx)

        current_best_fitness = fitness_scores[best_sparrow_index]
        if reverse_fitness > current_best_fitness:
            sparrows[best_sparrow_index] = reverse_best_sparrow
            fitness_scores[best_sparrow_index] = reverse_fitness
            best_sparrow = sparrows[best_sparrow_index]

        for j in range(population_size):
            if j == best_sparrow_index:
                continue

            if fitness_scores[j] < average_fitness:
                direction_to_safe_zone = safe_zone_center - sparrows[j]
                sparrows[j] += 0.5 * np.random.rand() * direction_to_safe_zone
            else:
                for k in range(num_sensors):
                    rand_sparrow_index = random.randint(0, population_size - 1)
                    perturb_vector = brightness_driven_perturbation(sparrows[j][k], sparrows[rand_sparrow_index][k], beta) - sparrows[j][k]
                    exploitation_vector = 0.5 * (best_sparrow[k] - sparrows[j][k])
                    sparrows[j][k] += perturb_vector + exploitation_vector

            dist_to_best = np.linalg.norm(sparrows[j] - best_sparrow)
            if dist_to_best < 2.0: # If a sparrow is too close, push it away
                for k in range(num_sensors):
                    sparrows[j][k] = dynamic_warning_update(sparrows[j][k], best_sparrow[k])

            if is_near_boundary(sparrows[j], deployment_area, threshold=1.0):
                direction_to_center = safe_zone_center - sparrows[j]
                sparrows[j] += 0.3 * np.random.rand() * direction_to_center

        beta = update_attraction_coefficient(beta_initial, i, max_iter)

        for j, sparrow in enumerate(sparrows):
            sparrows[j] = np.clip(sparrow, 0, deployment_area)
            # Pass new args
            fitness_scores[j] = fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets_np, max_dvar_approx)

    best_sparrow_index = np.argmax(fitness_scores)
    # print("Best Fitness Score:", fitness_scores[best_sparrow_index])
    # print("Best Sparrow Position:", sparrows[best_sparrow_index])
    return sparrows[best_sparrow_index]