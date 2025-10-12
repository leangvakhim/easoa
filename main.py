import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist # <-- IMPORT the optimized distance function

# --- Helper Functions ---

def calculate_coverage(positions, area_dim, perception_radius, grid_points=100):
    """
    Calculates network coverage using a highly optimized vectorized approach.
    This version is much faster than the loop-based method.
    """
    # 1. Create the grid of points to monitor
    grid_x, grid_y = np.meshgrid(np.linspace(0, area_dim, grid_points),
                                 np.linspace(0, area_dim, grid_points))
    monitoring_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # 2. Calculate the distance from EVERY monitoring point to EVERY sensor node at once.
    # This creates a matrix where rows are points and columns are nodes.
    distance_matrix = cdist(monitoring_points, positions)
    
    # 3. For each monitoring point, find the distance to the CLOSEST node.
    min_distances = np.min(distance_matrix, axis=1)
    
    # 4. Count how many points are within the perception radius of at least one node.
    covered_points = np.sum(min_distances <= perception_radius)
            
    return covered_points / len(monitoring_points)


def calculate_distribution_variance(positions):
    # This function is already fast, no changes needed
    if len(positions) < 2:
        return 0.0
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return np.var(distances)

def objective_function(positions, area_dim, perception_radius, w1, w2, w3):
    # No changes needed here
    num_nodes = len(positions.flatten()) // 2
    nodes = positions.reshape(num_nodes, 2)
    
    r_cover = calculate_coverage(nodes, area_dim, perception_radius)
    d_var = calculate_distribution_variance(nodes)
    e_total_penalty = d_var 
    
    fitness = w1 * r_cover - w2 * d_var - w3 * e_total_penalty
    return fitness


# --- EASOA Implementation (No changes from the previous version) ---

class EASOA:
    def __init__(self, obj_func, num_sparrows, num_dimensions, max_iter, bounds, func_args):
        self.obj_func = obj_func
        self.num_sparrows = num_sparrows
        self.num_dimensions = num_dimensions
        self.max_iter = max_iter
        self.bounds = bounds
        self.func_args = func_args
        
        self.discoverer_ratio = 0.2
        self.scout_ratio = 0.1
        self.beta_initial = 0.5
        self.elite_rate = 0.2
        
        self.num_discoverers = int(self.num_sparrows * self.discoverer_ratio)
        
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_sparrows, self.num_dimensions))
        
        self.fitness = np.zeros(self.num_sparrows)
        print("Evaluating initial population fitness...")
        for i in tqdm(range(self.num_sparrows), desc="Initializing Population"):
            self.fitness[i] = self.obj_func(self.positions[i], **self.func_args)

        self.global_best_pos = np.zeros(self.num_dimensions)
        self.global_best_fit = -np.inf
        self.convergence_curve = []
        self._update_best()

    def _update_best(self):
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.global_best_fit:
            self.global_best_fit = self.fitness[best_idx]
            self.global_best_pos = self.positions[best_idx].copy()

    def _reverse_elite_selection(self):
        num_elites = int(self.num_sparrows * self.elite_rate)
        elite_indices = np.argsort(self.fitness)[-num_elites:]
        
        for idx in elite_indices:
            current_pos = self.positions[idx]
            reverse_pos = self.bounds[1] + self.bounds[0] - current_pos
            reverse_pos = np.clip(reverse_pos, self.bounds[0], self.bounds[1])
            reverse_fitness = self.obj_func(reverse_pos, **self.func_args)
            
            if reverse_fitness > self.fitness[idx]:
                self.positions[idx] = reverse_pos
                self.fitness[idx] = reverse_fitness

    def _brightness_driven_perturbation(self, iter_num):
        beta = self.beta_initial * (1 - iter_num / self.max_iter)
        sorted_indices = np.argsort(self.fitness)
        
        for i in range(self.num_discoverers, self.num_sparrows):
            idx = sorted_indices[i]
            brighter_idx = sorted_indices[np.random.randint(self.num_discoverers, self.num_sparrows)]
            brighter_pos = self.positions[brighter_idx]

            distance_sq = np.sum((self.positions[idx] - brighter_pos)**2)
            attraction = beta * np.exp(-distance_sq) * (brighter_pos - self.positions[idx])
            random_perturb = 0.1 * np.random.randn(self.num_dimensions)
            
            self.positions[idx] += attraction + random_perturb
            self.positions[idx] = np.clip(self.positions[idx], self.bounds[0], self.bounds[1])
            self.fitness[idx] = self.obj_func(self.positions[idx], **self.func_args)


    def _dynamic_warning_update(self):
        num_scouts = int(self.num_sparrows * self.scout_ratio)
        scout_indices = np.random.choice(self.num_sparrows, num_scouts, replace=False)
        delta = 0.5
        
        for idx in scout_indices:
            r = np.random.rand()
            self.positions[idx] = self.positions[idx] + delta * (r * self.global_best_pos - self.positions[idx])
            self.positions[idx] = np.clip(self.positions[idx], self.bounds[0], self.bounds[1])
            self.fitness[idx] = self.obj_func(self.positions[idx], **self.func_args)

    def optimize(self):
        for t in tqdm(range(self.max_iter), desc="EASOA Optimization"):
            self._reverse_elite_selection()
            sorted_indices = np.argsort(self.fitness)
            
            for i in range(self.num_discoverers):
                idx = sorted_indices[-(i + 1)]
                r2 = np.random.rand()
                if r2 < 0.8:
                    self.positions[idx] += np.random.randn(self.num_dimensions)
                else:
                    self.positions[idx] += np.random.normal(0, 1, self.num_dimensions)
                self.positions[idx] = np.clip(self.positions[idx], self.bounds[0], self.bounds[1])
                self.fitness[idx] = self.obj_func(self.positions[idx], **self.func_args)
            
            self._brightness_driven_perturbation(t)
            self._dynamic_warning_update()
            
            self._update_best()
            self.convergence_curve.append(self.global_best_fit)

        return self.global_best_pos, self.global_best_fit


def plot_results(positions, area_dim, perception_radius, fitness, curve):
    # No changes needed here
    num_nodes = len(positions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_title(f'Optimized WSN Configuration (Fitness: {fitness:.4f})')
    ax1.set_xlim(0, area_dim)
    ax1.set_ylim(0, area_dim)
    ax1.set_xlabel('X/m')
    ax1.set_ylabel('Y/m')
    ax1.grid(True)
    
    for i in range(num_nodes):
        node = positions[i]
        ax1.plot(node[0], node[1], 'ko', markersize=8, label='WS Node' if i == 0 else "")
        circle = plt.Circle((node[0], node[1]), perception_radius, color='g', alpha=0.2, label='Coverage' if i == 0 else "")
        ax1.add_artist(circle)
        
    ax1.legend()

    ax2.set_title('EASOA Convergence Curve')
    ax2.plot(curve)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# --- Main Execution (No changes) ---
if __name__ == '__main__':
    AREA_DIMENSION = 50
    NUM_NODES = 20
    PERCEPTION_RADIUS = 10

    NUM_SPARROWS = 50
    MAX_ITERATIONS = 500
    NUM_DIMENSIONS = NUM_NODES * 2
    BOUNDS = [0, AREA_DIMENSION]

    W1, W2, W3 = 0.7, 0.2, 0.1

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
    
    print("\nOptimization Finished.")
    print(f"Best Fitness Found: {best_fitness}")
    
    plot_results(best_node_positions, AREA_DIMENSION, PERCEPTION_RADIUS, best_fitness, easoa.convergence_curve)