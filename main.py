import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist # Make sure to import pdist

# --- Helper Functions (No changes here) ---

def calculate_coverage(positions, area_dim, perception_radius, grid_points=100):
    """Calculates network coverage using the fast vectorized approach."""
    grid_x, grid_y = np.meshgrid(np.linspace(0, area_dim, grid_points),
                                 np.linspace(0, area_dim, grid_points))
    monitoring_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    if positions.shape[0] == 0:
        return 0.0
    
    distance_matrix = cdist(monitoring_points, positions)
    min_distances = np.min(distance_matrix, axis=1)
    covered_points = np.sum(min_distances <= perception_radius)
            
    return covered_points / len(monitoring_points)

def calculate_distribution_variance(positions):
    """Calculates the spatial distribution variance (D_var) of nodes."""
    if len(positions) < 2:
        return 0.0
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return np.var(distances)

def objective_function(positions, area_dim, perception_radius, w1, w2, w3):
    """The multi-objective fitness function from Equation (8) with the missing constraint."""
    num_nodes = len(positions.flatten()) // 2
    nodes = positions.reshape(num_nodes, 2)
    
    r_cover = calculate_coverage(nodes, area_dim, perception_radius)
    d_var = calculate_distribution_variance(nodes)
    
    # --- START: NEW PENALTY CODE ---
    # This implements the missing constraint from the paper 
    
    # Define a minimum distance to enforce separation. 
    # A fraction of the perception radius is a good starting point.
    d_min = perception_radius / 2.0 
    
    penalty = 0.0
    if num_nodes > 1:
        # pdist calculates the pairwise distances between all nodes
        distances = pdist(nodes)
        
        # Find all distances that are less than the minimum allowed distance
        violating_distances = distances[distances < d_min]
        
        # Add a large penalty for each violation. 
        # The penalty should be significant enough to guide the search.
        # We penalize based on how close the nodes are.
        if len(violating_distances) > 0:
            penalty = np.sum(d_min - violating_distances) * 10 # Multiplier makes the penalty stronger

    # --- END: NEW PENALTY CODE ---

    # The final fitness now includes the penalty
    # Note: penalty is subtracted because we are maximizing. A higher penalty lowers the fitness.
    fitness = w1 * r_cover - w2 * d_var - penalty
    return fitness


# --- FINAL CORRECTED EASOA Implementation ---

class EASOA:
    def __init__(self, obj_func, num_sparrows, num_dimensions, max_iter, bounds, func_args):
        self.obj_func = obj_func
        self.num_sparrows = num_sparrows
        self.num_dimensions = num_dimensions
        self.max_iter = max_iter
        self.bounds = np.array(bounds)
        self.func_args = func_args
        
        # [cite_start]Parameters from the paper's text (Section 3.1) [cite: 402]
        self.producer_ratio = 0.2
        self.scout_ratio = 0.1
        self.elite_rate = 0.2
        self.beta = 0.5  # Brightness drive perturbation coefficient
        self.delta = 0.3 # Warning update coefficient
        
        self.num_producers = int(self.num_sparrows * self.producer_ratio)
        self.num_scouts = int(self.num_sparrows * self.scout_ratio)
        
        # Initialize positions and fitness
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_sparrows, self.num_dimensions))
        self.fitness = np.zeros(self.num_sparrows)
        print("Evaluating initial population fitness...")
        for i in tqdm(range(self.num_sparrows), desc="Initializing Population"):
            self.fitness[i] = self.obj_func(self.positions[i], **self.func_args)

        self.global_best_pos = np.zeros(self.num_dimensions)
        self.global_best_fit = -np.inf
        self.convergence_curve = []
        self._update_global_best()

    def _update_global_best(self):
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.global_best_fit:
            self.global_best_fit = self.fitness[best_idx]
            self.global_best_pos = self.positions[best_idx].copy()
            
    def _reverse_elite_selection(self):
        # [cite_start]Implements Equations (3) and (4) [cite: 137, 140]
        num_elites = int(self.num_sparrows * self.elite_rate)
        elite_indices = np.argsort(self.fitness)[-num_elites:]
        
        for idx in elite_indices:
            current_pos = self.positions[idx]
            reverse_pos = self.bounds[1] + self.bounds[0] - current_pos
            reverse_fitness = self.obj_func(reverse_pos, **self.func_args)
            
            if reverse_fitness > self.fitness[idx]:
                self.positions[idx] = reverse_pos

    def optimize(self):
        for t in tqdm(range(self.max_iter), desc="EASOA Optimization"):
            
            # --- 1. Reverse Elite Selection (EASOA Enhancement) ---
            # [cite_start]Applied at the start of the iteration to increase diversity [cite: 237]
            self._reverse_elite_selection()

            # --- Sort sparrows by fitness to assign roles ---
            sorted_indices = np.argsort(self.fitness)[::-1] # Best fitness is at index 0
            
            # --- 2. Producer (Discoverer) Phase (CORRECTED) ---
            # The best sparrows explore widely
            for i in range(self.num_producers):
                idx = sorted_indices[i]
                r1 = np.random.rand()
                
                # Standard SSA producer update rule to prevent collapsing to origin
                # This encourages exploration of the search space
                if np.random.rand() < 0.8: # A common threshold value (ST in SSA literature)
                    # Move towards a random direction
                    self.positions[idx] = self.positions[idx] * (1 + np.random.uniform(-0.5, 0.5))
                else:
                    # Jump to a new random position in the vicinity
                    self.positions[idx] = self.positions[idx] + np.random.normal(0, 1, self.num_dimensions)

            # --- 3. Scrounger (Joiner) Phase ---
            # The other sparrows follow the best producer, but with the EASOA enhancement
            for i in range(self.num_producers, self.num_sparrows):
                idx = sorted_indices[i]
                best_producer_pos = self.positions[sorted_indices[0]]
                
                # [cite_start]Brightness-Driven Perturbation - Equation (5) [cite: 144]
                gamma = 0.5 # Attenuation coefficient
                distance_sq = np.sum((self.positions[idx] - best_producer_pos)**2)
                attraction = self.beta * np.exp(-gamma * distance_sq) * (best_producer_pos - self.positions[idx])
                self.positions[idx] += attraction

            # --- 4. Scout Phase (EASOA Enhancement) ---
            # A random subset becomes scouts to prevent getting stuck
            scout_indices = np.random.choice(self.num_sparrows, self.num_scouts, replace=False)
            for idx in scout_indices:
                # [cite_start]Dynamic Warning Update - Equation (7) [cite: 165]
                r = np.random.rand()
                self.positions[idx] += self.delta * (r * self.global_best_pos - self.positions[idx])

            # --- 5. Recalculate Fitness and Update Best ---
            for i in range(self.num_sparrows):
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                self.fitness[i] = self.obj_func(self.positions[i], **self.func_args)
            
            self._update_global_best()
            self.convergence_curve.append(self.global_best_fit)
            
        return self.global_best_pos, self.global_best_fit


def plot_results(positions, area_dim, perception_radius, fitness, curve):
    num_nodes = len(positions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.set_title(f'Optimized WSN Configuration (Fitness: {fitness:.4f})')
    ax1.set_xlim(0, area_dim)
    ax1.set_ylim(0, area_dim)
    ax1.set_xlabel('X/m')
    ax1.set_ylabel('Y/m')
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    
    for i in range(num_nodes):
        node = positions[i]
        ax1.plot(node[0], node[1], 'ko', markersize=8, label='WS Node' if i == 0 else "")
        circle = plt.Circle((node[0], node[1]), perception_radius, color='g', alpha=0.2, label='Coverage' if i == 0 else "")
        ax1.add_artist(circle)
    if num_nodes > 0:
      ax1.legend(loc='upper right')

    ax2.set_title('EASOA Convergence Curve')
    ax2.plot(curve)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # [cite_start]Simulation Parameters from the paper [cite: 389]
    AREA_DIMENSION = 50
    NUM_NODES = 20
    PERCEPTION_RADIUS = 10

    # [cite_start]Algorithm Parameters from the paper [cite: 401]
    NUM_SPARROWS = 50
    MAX_ITERATIONS = 500
    NUM_DIMENSIONS = NUM_NODES * 2
    BOUNDS = [0, AREA_DIMENSION]

    # Objective function weights, prioritized for uniformity
    W1, W2, W3 = 0.4, 0.5, 0.1

    func_args = {
        'area_dim': AREA_DIMENSION,
        'perception_radius': PERCEPTION_RADIUS,
        'w1': W1, 'w2': W2, 'w3': W3
    }
    
    print("Starting EASOA for WSN node configuration...")
    
    easoa = EASOA(
        obj_func=objective_function,
        num_sparrows=NUM_SPARROWS,
        num_dimensions=NUM_DIMENSIONS,
        max_iter=MAX_ITERATIONS,
        bounds=BOUNDS,
        func_args=func_args
    )
    
    best_positions_flat, best_fitness = easoa.optimize()
    
    best_node_positions = best_positions_flat.reshape(NUM_NODES, 2)
    
    print(f"\nOptimization Finished.")
    print(f"Best Fitness Found: {best_fitness:.4f}")
    
    plot_results(best_node_positions, AREA_DIMENSION, PERCEPTION_RADIUS, best_fitness, easoa.convergence_curve)