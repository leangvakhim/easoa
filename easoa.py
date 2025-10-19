import random
import numpy as np
from calculateeasoa import total_coverage_prob, total_coverage, fitness_value, reverse_elite_selection, update_reverse_elite, brightness_driven_perturbation, update_attraction_coefficient, dynamic_warning_update

def easoa(num_sensors, deployment_area, max_iter, population_size, sensing_radius, w1, w2, w3):
    # Initialize the population of sparrows
    sparrows = [np.random.rand(num_sensors, 2) * deployment_area for _ in range(population_size)]
    fitness_scores = np.zeros(population_size)

    for i in range(max_iter):
        # 1. Reverse Elite Selection
        sorted_indices = np.argsort(fitness_scores)
        elite_sparrows = [sparrows[i] for i in sorted_indices[:int(population_size * 0.2)]]
        for elite in elite_sparrows:
            x_min = np.min(elite, axis=0)
            x_max = np.max(elite, axis=0)
            for j in range(len(elite)):
                elite[j] = reverse_elite_selection(elite[j], x_min, x_max)

        # 2. Fitness Calculation and Sorting
        for j, sparrow in enumerate(sparrows):
            coverage = 0
            random_targets = [(random.randrange(0, deployment_area), random.randrange(0, deployment_area)) for _ in range(max_iter)]
            for point in random_targets:
                coverage += total_coverage_prob(sparrow, point, sensing_radius)

            coverage = total_coverage(coverage, len(random_targets))

            # spatial distribution variance (dvar)
            dvar = np.var(sparrow)

            # Energy consumption
            energy = 100

            fitness_scores[j] = fitness_value(coverage, energy, dvar, w1, w2, w3)

        # 4. Brightness-Driven Perturbation
        for j in range(population_size):
            # Select a random sparrow to interact with
            rand_sparrow_index = random.randint(0, population_size - 1)
            for k in range(num_sensors):
                sparrows[j][k] = brightness_driven_perturbation(sparrows[j][k], sparrows[rand_sparrow_index][k], beta=0.5)

        # 5. Dynamic Warning Update
        best_sparrow_index = np.argmax(fitness_scores)
        best_sparrow = sparrows[best_sparrow_index]
        for j in range(population_size):
            for k in range(num_sensors):
                sparrows[j][k] = dynamic_warning_update(sparrows[j][k], best_sparrow[k], delta=0.5)

    # Return the best set of sensor positions found
    best_sparrow_index = np.argmax(fitness_scores)
    return sparrows[best_sparrow_index]