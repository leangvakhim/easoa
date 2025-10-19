import random
import numpy as np
from tqdm import tqdm
from calculateeasoa import total_coverage_prob, total_coverage, fitness_value, reverse_elite_selection, update_reverse_elite, brightness_driven_perturbation, update_attraction_coefficient, dynamic_warning_update, is_near_boundary

def easoa(num_sensors, deployment_area, max_iter, population_size, sensing_radius, w1, w2, w3, random_targets):
    # Initialize the population of sparrows
    sparrows = [np.random.rand(num_sensors, 2) * deployment_area for _ in range(population_size)]
    fitness_scores = np.zeros(population_size)

    for j, sparrow in enumerate(sparrows):
        fitness_scores[j] = fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets)

    # Initialize warning value
    # warning_value = 0.5
    warning_value = 0
    beta_initial = 0.5

    for i in tqdm(range(max_iter), desc="EASOA Optimization Progress"):
        # 1. Reverse Elite Selection
        sorted_indices = np.argsort(fitness_scores)
        elite_sparrows = [sparrows[i] for i in sorted_indices[:int(population_size * 0.2)]]
        for elite in elite_sparrows:
            x_min = np.min(elite, axis=0)
            x_max = np.max(elite, axis=0)
            for j in range(len(elite)):
                elite[j] = reverse_elite_selection(elite[j], x_min, x_max)

        # 2. Brightness-Driven Perturbation and Dynamic Warning Update (Main Loop)
        beta = update_attraction_coefficient(beta_initial, i, max_iter)
        best_sparrow_index = np.argmax(fitness_scores)
        best_sparrow = sparrows[best_sparrow_index]

        for j in range(population_size):
            # Brightness-driven perturbation
            rand_sparrow_index = random.randint(0, population_size - 1)
            for k in range(num_sensors):
                sparrows[j][k] = brightness_driven_perturbation(sparrows[j][k], sparrows[rand_sparrow_index][k], beta)

            # Dynamic warning update
            for k in range(num_sensors):
                sparrows[j][k] = dynamic_warning_update(sparrows[j][k], best_sparrow[k])

            # Conditional logic based on flowchart
            if fitness_scores[j] > warning_value:
                # Extensive search (already done by perturbation and warning update)
                pass # This is a simplification; in a more complex version, you might have a different search here
            else:
                # Move closer to the safe zone (e.g., move towards the center)
                center = np.array([deployment_area / 2, deployment_area / 2])
                for k in range(num_sensors):
                    sparrows[j][k] += 0.1 * (center - sparrows[j][k]) # Move 10% towards the center

            if is_near_boundary(sparrows[j], deployment_area):
                # Edge move closer strategy (e.g., move away from the edge)
                 for k in range(num_sensors):
                    if np.any(sparrows[j][k] < 1.0):
                        sparrows[j][k] += 0.2
                    if np.any(sparrows[j][k] > (deployment_area - 1.0)):
                        sparrows[j][k] -= 0.2

        # 3. Fitness Calculation and Sorting (at the end of the loop)
        for j, sparrow in enumerate(sparrows):
            fitness_scores[j] = fitness_value(sparrow, w1, w2, w3, sensing_radius, deployment_area, random_targets)

        # # 2. Fitness Calculation and Sorting
        # for j, sparrow in enumerate(sparrows):
        #     coverage = 0
        #     random_targets = [(random.randrange(0, deployment_area), random.randrange(0, deployment_area)) for _ in range(max_iter)]
        #     for point in random_targets:
        #         coverage += total_coverage_prob(sparrow, point, sensing_radius)

        #     coverage = total_coverage(coverage, len(random_targets))

        #     # spatial distribution variance (dvar)
        #     dvar = np.var(sparrow)

        #     # Energy consumption
        #     energy = 100

        #     fitness_scores[j] = fitness_value(coverage, energy, dvar, w1, w2, w3)

        # # 4. Brightness-Driven Perturbation
        # for j in range(population_size):
        #     # Select a random sparrow to interact with
        #     rand_sparrow_index = random.randint(0, population_size - 1)
        #     for k in range(num_sensors):
        #         sparrows[j][k] = brightness_driven_perturbation(sparrows[j][k], sparrows[rand_sparrow_index][k], beta=0.5)

        # # 5. Dynamic Warning Update
        # best_sparrow_index = np.argmax(fitness_scores)
        # best_sparrow = sparrows[best_sparrow_index]
        # for j in range(population_size):
        #     for k in range(num_sensors):
        #         sparrows[j][k] = dynamic_warning_update(sparrows[j][k], best_sparrow[k], delta=0.3)

    # Return the best set of sensor positions found
    best_sparrow_index = np.argmax(fitness_scores)
    print("Best Fitness Score:", fitness_scores[best_sparrow_index])
    print("Best Sparrow Position:", sparrows[best_sparrow_index])
    return sparrows[best_sparrow_index]