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

def easoa(num_sensors, deployment_area, max_iter, population_size, sensing_radius, w1, w2, w3, random_targets):
    # Initialize the population of sparrows
    sparrows = [np.random.rand(num_sensors, 2) * deployment_area for _ in range(population_size)]
    fitness_scores = np.zeros(population_size)

    for j, sparrow in enumerate(sparrows):
        fitness_scores[j] = fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets)

    beta_initial = 0.5
    beta = 0.5

    for i in tqdm(range(max_iter), desc="EASOA Optimization Progress"):
        # Sort by fitness to find the best and worst sparrows
        sorted_indices = np.argsort(fitness_scores)
        best_sparrow_index = sorted_indices[-1] # Highest score is best
        worst_sparrow_index = sorted_indices[0]
        best_sparrow = sparrows[best_sparrow_index]

        # --- Main Population Update Loop ---
        for j in range(population_size):
            # Move most sparrows (joiners) towards the best sparrow
            if j != best_sparrow_index:
                for k in range(num_sensors):
                    # This combines brightness perturbation (exploration) and a move towards the best (exploitation)
                    rand_sparrow_index = random.randint(0, population_size - 1)

                    # Brightness perturbation component
                    perturb_vector = brightness_driven_perturbation(sparrows[j][k], sparrows[rand_sparrow_index][k], beta) - sparrows[j][k]

                    # Movement towards the best solution
                    exploitation_vector = 0.5 * (best_sparrow[k] - sparrows[j][k]) # A simple pull towards the best

                    sparrows[j][k] += perturb_vector + exploitation_vector

        # Apply Dynamic Warning Update for sparrows that get too close to the best (to avoid crowding)
        for j in range(population_size):
            if j != best_sparrow_index:
                dist_to_best = np.linalg.norm(sparrows[j] - best_sparrow)
                if dist_to_best < 2.0: # If a sparrow is too close
                    for k in range(num_sensors):
                        sparrows[j][k] = dynamic_warning_update(sparrows[j][k], best_sparrow[k])

        # Final Fitness Calculation and Boundary Clipping for the new generation
        for j, sparrow in enumerate(sparrows):
            # Clip to ensure sparrows stay within the deployment area
            sparrows[j] = np.clip(sparrow, 0, deployment_area)
            fitness_scores[j] = fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets)

    # Return the best set of sensor positions found
    best_sparrow_index = np.argmax(fitness_scores)
    print("Best Fitness Score:", fitness_scores[best_sparrow_index])
    print("Best Sparrow Position:", sparrows[best_sparrow_index])
    return sparrows[best_sparrow_index]