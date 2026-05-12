import numpy as np
import random
import time

class SOS:
    """
    Symbiotic Organisms Search (SOS) Optimizer
    """
    def __init__(self, pop_size, max_iterations, patience=50):
        # hyperparameters
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.patience = patience

        self.candidate_solutions = []
        self.param_ranges = []
        self.param_names = []
        self.best_params = {}
        self.best_fitness = float('inf')
        self.history = {}
        self.fitness = []

        self.epoch_count = 0
        self.time = 0.0
        self.early_stop = False

    def _init_population(self, bounds):
        self.candidate_solutions = []
        lows = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]
        
        # Initialize randomly within bounds
        self.candidate_solutions = lows + np.random.rand(self.pop_size, len(self.param_names)) * (highs - lows)

    def _evaluate_all(self, fitness_fn):
        self.fitness = np.array([fitness_fn(row) for row in self.candidate_solutions])

    def get_best_params(self):
        return dict(zip(self.param_names, self.best_params))

    def run(self, fitness_fn, bounds):
        self.history = {
            "LE_energy": [],       # Keeping same name as AOS for runner compatibility
            "LE_position": [],
            "population": [],
            "population_avg_fitness": []
        }
        self.param_names = list(bounds.keys())
        self.param_ranges = np.array([bounds[k] for k in self.param_names])
        
        lows = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        self._init_population(bounds)
        self._evaluate_all(fitness_fn)
        
        # Find initial best organism
        best_idx = np.argmin(self.fitness)
        self.best_params = self.candidate_solutions[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        no_improve_count = 0
        best_energy_seen = self.best_fitness

        start_time = time.perf_counter()
        iteration = 0
        while iteration < self.max_iterations:
            for i in range(self.pop_size):
                # Phase 1: Mutualism
                j = random.choice([x for x in range(self.pop_size) if x != i])
                mutual_vector = (self.candidate_solutions[i] + self.candidate_solutions[j]) / 2.0
                bf1 = random.choice([1, 2])
                bf2 = random.choice([1, 2])
                
                new_xi = self.candidate_solutions[i] + np.random.rand(len(self.param_names)) * (self.best_params - mutual_vector * bf1)
                new_xj = self.candidate_solutions[j] + np.random.rand(len(self.param_names)) * (self.best_params - mutual_vector * bf2)
                
                new_xi = np.clip(new_xi, lows, highs)
                new_xj = np.clip(new_xj, lows, highs)
                
                fit_i = fitness_fn(new_xi)
                fit_j = fitness_fn(new_xj)
                
                if fit_i < self.fitness[i]:
                    self.candidate_solutions[i] = new_xi
                    self.fitness[i] = fit_i
                if fit_j < self.fitness[j]:
                    self.candidate_solutions[j] = new_xj
                    self.fitness[j] = fit_j

                # Update best
                current_best_idx = np.argmin(self.fitness)
                self.best_params = self.candidate_solutions[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

                # Phase 2: Commensalism
                j = random.choice([x for x in range(self.pop_size) if x != i])
                new_xi = self.candidate_solutions[i] + (np.random.rand(len(self.param_names)) * 2 - 1) * (self.best_params - self.candidate_solutions[j])
                new_xi = np.clip(new_xi, lows, highs)
                
                fit_i = fitness_fn(new_xi)
                if fit_i < self.fitness[i]:
                    self.candidate_solutions[i] = new_xi
                    self.fitness[i] = fit_i

                # Update best
                current_best_idx = np.argmin(self.fitness)
                self.best_params = self.candidate_solutions[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

                # Phase 3: Parasitism
                j = random.choice([x for x in range(self.pop_size) if x != i])
                parasite_vector = self.candidate_solutions[i].copy()
                
                # Modify random dimension for the parasite
                dim = random.randint(0, len(self.param_names) - 1)
                parasite_vector[dim] = lows[dim] + np.random.rand() * (highs[dim] - lows[dim])
                
                fit_p = fitness_fn(parasite_vector)
                if fit_p < self.fitness[j]:
                    self.candidate_solutions[j] = parasite_vector
                    self.fitness[j] = fit_p

                # Update best
                current_best_idx = np.argmin(self.fitness)
                self.best_params = self.candidate_solutions[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

            # Logging
            self.history["LE_energy"].append(self.best_fitness)
            self.history["LE_position"].append(self.best_params.copy())
            self.history["population_avg_fitness"].append(np.mean(self.fitness))

            # early stopping
            if self.best_fitness < best_energy_seen:
                best_energy_seen = self.best_fitness
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= self.patience:
                self.early_stop = True
                iteration += 1
                break

            iteration += 1

        self.epoch_count = iteration
        self.time = (time.perf_counter() - start_time) * 1000
        return self.best_params
