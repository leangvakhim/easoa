# easoa_general.py
import random
import numpy as np
from tqdm import tqdm

# Import the same helper functions from your calculateeasoa.py
from calculateeasoa import (
    reverse_elite_selection,
    brightness_driven_perturbation,
    update_attraction_coefficient,
    dynamic_warning_update,
    is_near_boundary # This function will need a small tweak or replacement
)

# A modified is_near_boundary for 1D vectors
def is_near_boundary_general(sparrow_vector, min_bound, max_bound, threshold=1.0):
    """Checks if a 1D vector is near the problem bounds."""
    return np.any(sparrow_vector < (min_bound + threshold)) or \
           np.any(sparrow_vector > (max_bound - threshold))


def easoa_general(objective_func, dimensions, min_bound, max_bound, max_iter, population_size):
    """
    A general-purpose EASOA for D-dimensional benchmark functions.
    """

    # --- 1. Initialization ---
    # Initialize sparrows as 1D vectors of shape (dimensions,)
    sparrows = [
        np.random.rand(dimensions) * (max_bound - min_bound) + min_bound
        for _ in range(population_size)
    ]
    fitness_scores = np.zeros(population_size)

    for j, sparrow in enumerate(sparrows):
        # Call the generic objective function
        fitness_scores[j] = objective_func(sparrow)

    beta_initial = 0.5
    beta = 0.5

    # Safe zone center is the middle of the search space
    safe_zone_center = np.full((dimensions,), (max_bound + min_bound) / 2.0)

    for i in tqdm(range(max_iter), desc="EASOA Optimization Progress"):
        # --- 2. Sorting and Elite Selection ---
        # Note: Standard benchmark functions are for *minimization*.
        # Your WSN code was for *maximization*. We sort from low (best) to high (worst).
        sorted_indices = np.argsort(fitness_scores)
        best_sparrow_index = sorted_indices[0] # Best is index 0
        best_sparrow = sparrows[best_sparrow_index]

        average_fitness = np.mean(fitness_scores)

        # Reverse Elite Selection (Eq. 3)
        # We find the *worst* sparrow to apply reverse elite selection to.
        worst_sparrow_index = sorted_indices[-1]
        worst_sparrow = sparrows[worst_sparrow_index]

        all_sparrows_np = np.array(sparrows)
        x_min = np.min(all_sparrows_np) # Simpler min/max for 1D vectors
        x_max = np.max(all_sparrows_np)

        reverse_worst_sparrow = reverse_elite_selection(worst_sparrow, x_min, x_max)
        reverse_fitness = objective_func(reverse_worst_sparrow)

        # If reverse is better than worst, replace it
        if reverse_fitness < fitness_scores[worst_sparrow_index]:
            sparrows[worst_sparrow_index] = reverse_worst_sparrow
            fitness_scores[worst_sparrow_index] = reverse_fitness

        # --- 3. Population Update ---
        for j in range(population_size):
            if j == best_sparrow_index:
                continue

            # Your original logic: if fitness is "bad" (high for minimization)
            if fitness_scores[j] > average_fitness:
                # Move toward safe zone
                direction_to_safe_zone = safe_zone_center - sparrows[j]
                sparrows[j] += 0.5 * np.random.rand() * direction_to_safe_zone
            else: # if fitness is "good" (low for minimization)
                # This is the key change: No inner loop.
                # Apply updates to the WHOLE vector.
                rand_sparrow_index = random.randint(0, population_size - 1)

                # Brightness-Driven Perturbation (Eq. 5)
                perturb_vector = brightness_driven_perturbation(
                    sparrows[j], sparrows[rand_sparrow_index], beta
                ) - sparrows[j]

                # Exploitation
                exploitation_vector = 0.5 * (best_sparrow - sparrows[j])

                # Update the entire sparrow vector
                sparrows[j] += perturb_vector + exploitation_vector

            # Dynamic Warning Update (Eq. 7)
            dist_to_best = np.linalg.norm(sparrows[j] - best_sparrow)
            if dist_to_best < 2.0: # If too close
                sparrows[j] = dynamic_warning_update(sparrows[j], best_sparrow)

            # Boundary Check
            if is_near_boundary_general(sparrows[j], min_bound, max_bound, threshold=1.0):
                direction_to_center = safe_zone_center - sparrows[j]
                sparrows[j] += 0.3 * np.random.rand() * direction_to_center

        # --- 4. Update Beta and Re-evaluate ---
        beta = update_attraction_coefficient(beta_initial, i, max_iter)

        for j, sparrow in enumerate(sparrows):
            # Clip to bounds
            sparrows[j] = np.clip(sparrow, min_bound, max_bound)
            # Recalculate fitness
            fitness_scores[j] = objective_func(sparrows[j])

    # --- 5. Return Best Result ---
    # Return the best solution found (minimization)
    best_index = np.argmin(fitness_scores)
    print(f"\nOptimization Complete.")
    print(f"Best Fitness Score: {fitness_scores[best_index]}")
    return sparrows[best_index]