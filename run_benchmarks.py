# run_benchmarks.py
import numpy as np
import random
from easoa_general import easoa_general
from benchmark_function import ackley, griewank, schwefel_1_2, high_conditioned_elliptic

# --- Set Global Parameters ---
# Parameters from the paper
DIMENSIONS = 30  # Common dimension for benchmark testing
MAX_ITER = 500
POP_SIZE = 50

# Set random seed for reproducibility
np.random.seed(123)
random.seed(123)

print("--- 1. Testing on Ackley Function ---")
# Domain: [-32.768, 32.768]
best_ackley = easoa_general(
    objective_func=ackley,
    dimensions=DIMENSIONS,
    min_bound=-32.768,
    max_bound=32.768,
    max_iter=MAX_ITER,
    population_size=POP_SIZE
)

print("\n--- 2. Testing on Griewank Function ---")
# Domain: [-600, 600]
best_griewank = easoa_general(
    objective_func=griewank,
    dimensions=DIMENSIONS,
    min_bound=-600.0,
    max_bound=600.0,
    max_iter=MAX_ITER,
    population_size=POP_SIZE
)

print("\n--- 3. Testing on Schwefel's Problem 1.2 ---")
# Domain: [-100, 100]
best_schwefel = easoa_general(
    objective_func=schwefel_1_2,
    dimensions=DIMENSIONS,
    min_bound=-100.0,
    max_bound=100.0,
    max_iter=MAX_ITER,
    population_size=POP_SIZE
)

print("\n--- 4. Testing on High Conditioned Elliptic ---")
# Domain: [-100, 100]
best_elliptic = easoa_general(
    objective_func=high_conditioned_elliptic,
    dimensions=DIMENSIONS,
    min_bound=-100.0,
    max_bound=100.0,
    max_iter=MAX_ITER,
    population_size=POP_SIZE
)

print("\n\n--- All Benchmark Tests Complete ---")