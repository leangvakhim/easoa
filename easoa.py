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

        # ===================================================================
        # START: REVERSE ELITE SELECTION STRATEGY (NEWLY ADDED)
        # ===================================================================

        # Find the min and max boundaries of the current sparrow population
        # This is needed for Equation (3)
        all_sparrows_np = np.array(sparrows)
        x_min = np.min(all_sparrows_np, axis=(0, 1))
        x_max = np.max(all_sparrows_np, axis=(0, 1))

        # Apply Equation (3) to generate a reverse elite position
        reverse_best_sparrow = reverse_elite_selection(best_sparrow, x_min, x_max)

        # Calculate the fitness of this new "reverse" sparrow
        reverse_fitness = fitness_value(reverse_best_sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets)

        # Apply Equation (4): If the reverse sparrow is better, replace the current best
        # Note: We use ">" because for this problem, a higher fitness score is better.
        current_best_fitness = fitness_scores[best_sparrow_index]
        if reverse_fitness > current_best_fitness:
            sparrows[best_sparrow_index] = reverse_best_sparrow
            fitness_scores[best_sparrow_index] = reverse_fitness
            # Update the 'best_sparrow' variable for the rest of the current iteration
            best_sparrow = sparrows[best_sparrow_index]

        # ===================================================================
        # END: REVERSE ELITE SELECTION STRATEGY
        # ===================================================================

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