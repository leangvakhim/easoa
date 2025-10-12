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
    """
    The corrected multi-objective fitness function, focusing only on the relevant
    goals for node placement: Coverage and Uniformity.
    """
    num_nodes = len(positions.flatten()) // 2
    nodes = positions.reshape(num_nodes, 2)
    
    # --- Constraint 1: Node Separation (from the paper) ---
    # We add a penalty if nodes are too close. This is the most critical part
    # for forcing a uniform, grid-like distribution.
    d_min = perception_radius / 1.5 # A slightly larger minimum distance
    
    penalty = 0.0
    if num_nodes > 1:
        distances = pdist(nodes)
        violations = distances[distances < d_min]
        if len(violations) > 0:
            # The penalty MUST be severe to act as a hard constraint.
            penalty = 1e7 * len(violations)

    # --- Main Objectives (Coverage vs. Uniformity) ---
    r_cover = calculate_coverage(nodes, area_dim, perception_radius)
    d_var = calculate_distribution_variance(nodes)
    
    # This is the corrected fitness calculation.
    # We are MAXIMIZING coverage and MINIMIZING variance and penalties.
    # The w3 term is correctly removed.
    fitness = (w1 * r_cover) - (w2 * d_var) - penalty
    
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
        self.alpha = 0.1     # <<< ADD THIS LINE: Disturbance intensity factor
        
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
        ST = 0.8  # Security threshold for producers
        
        for t in tqdm(range(self.max_iter), desc="EASOA Optimization"):
            
            # 1. Reverse Elite Selection (EASOA Enhancement)
            self._reverse_elite_selection()

            # Sort sparrows by fitness
            sorted_indices = np.argsort(self.fitness)[::-1]
            best_pos_current_iter = self.positions[sorted_indices[0]].copy()
            
            # --- 2. Producer (Discoverer) Phase ---
            for i in range(self.num_producers):
                idx = sorted_indices[i]
                r2 = np.random.rand()
                
                if r2 < ST:
                    # Explore in a random direction
                    self.positions[idx] += np.random.randn(self.num_dimensions)
                else:
                    # Move towards the center of the search space ("safe zone")
                    self.positions[idx] = best_pos_current_iter + np.random.randn(self.num_dimensions) * 0.5

            # --- 3. Scrounger (Joiner) Phase ---
            for i in range(self.num_producers, self.num_sparrows):
                idx = sorted_indices[i]
                
                # Brightness-Driven Perturbation (Equation 5)
                best_producer_pos = self.positions[sorted_indices[0]]
                gamma = 0.5
                distance_sq = np.sum((self.positions[idx] - best_producer_pos)**2)
                attraction = self.beta * np.exp(-gamma * distance_sq) * (best_producer_pos - self.positions[idx])
                disturbance = self.alpha * np.random.randn(self.num_dimensions)
                self.positions[idx] += attraction + disturbance

            # --- 4. Scout Phase (with "Safe Zone" logic) ---
            num_scouts = int(self.num_sparrows * self.scout_ratio)
            scout_indices = sorted_indices[-num_scouts:] # The worst-performing sparrows become scouts

            for idx in scout_indices:
                # If the scout is not the global best, reset its position
                if not np.array_equal(self.positions[idx], self.global_best_pos):
                    # Dynamic Warning Update (Equation 7)
                    r = np.random.uniform(-1, 1)
                    self.positions[idx] = self.global_best_pos + r * np.abs(self.positions[idx] - self.global_best_pos)
                else:
                    # If it is somehow the best, move it randomly
                    self.positions[idx] += np.random.uniform(-1, 1)

            # --- 5. Recalculate Fitness and Update Best ---
            for i in range(self.num_sparrows):
                # Clip positions to be within bounds
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                # Re-evaluate fitness with the new positions
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