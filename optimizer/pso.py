import numpy as np
import time

class PSO:

    def __init__(self, pop_size, max_iterations, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1, patience=50):
        # hyperparameters
        self.candidate_solutions = []
        self.param_ranges = []
        self.param_names = []
        self.best_params = {}
        self.best_fitness = {}
        self.history = {}
        self.pop_size = pop_size

        self.max_iterations = max_iterations
        self.w  = w_max
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1  # cognitive coefficient (personal best pull)
        self.c2 = c2  # social coefficient (global best pull)
        self.max_vel_frac = max_vel_frac
        self.patience = patience

        self.epoch_count = 0
        self.time = 0.0
        self.early_stop = False

    # --- setup ---

    def _init_population(self, bounds, initial_population=None):
        """
        Example of bounds = {
            "window": (2, 200),
            "alpha":  (0.0, 1.0)
        }
        """
        lows  = self.param_ranges[:, 0]  # all minimums
        highs = self.param_ranges[:, 1]  # all maximums

        if initial_population is not None:
            self.candidate_solutions = np.array(initial_population, dtype=float)
        else:
            self.candidate_solutions = lows + np.random.rand(self.pop_size, len(self.param_names)) * (highs - lows)

        # velocities initialised
        span = highs - lows
        max_velocity = self.max_vel_frac * span
        self.velocities = -max_velocity + np.random.rand(self.pop_size, len(self.param_names)) * 2 * max_velocity

        # each particle starts as its own personal best
        self.pbest_positions = self.candidate_solutions.copy()
        self.pbest_energy    = np.full(self.pop_size, np.inf)

    def _evaluate_all(self, fitness_fn):
        self.energy = np.array([fitness_fn(row) for row in self.candidate_solutions])

    # --- core PSO mechanics ---

    def _update_personal_best(self):
        improved = self.energy < self.pbest_energy
        self.pbest_positions[improved] = self.candidate_solutions[improved].copy()
        self.pbest_energy[improved]    = self.energy[improved]

    def _update_global_best(self):
        best_idx = np.argmin(self.pbest_energy)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_energy   = self.pbest_energy[best_idx]

    def _update_velocities_and_positions(self):
        lows  = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        span = highs - lows

        r1 = np.random.rand(self.pop_size, len(self.param_names))
        r2 = np.random.rand(self.pop_size, len(self.param_names))

        # Eq. 2a Modified PSO (1998)
        cognitive = self.c1 * r1 * (self.pbest_positions - self.candidate_solutions)
        social    = self.c2 * r2 * (self.gbest_position  - self.candidate_solutions)
        self.velocities = self.w * self.velocities + cognitive + social
        max_velocity = self.max_vel_frac * span
        self.velocities = np.clip(self.velocities, -max_velocity, max_velocity)
        self.candidate_solutions = np.clip(self.candidate_solutions + self.velocities, lows, highs)

    def get_best_params(self):
        return dict(zip(self.param_names, self.best_params))

    # --- main entry point ---
    def run(self, fitness_fn, bounds, initial_population=None):
        self.history = {
            "gbest_energy":   [],
            "gbest_position": [],
            "population":     [],
        }
        self.early_stop = False
        self.param_names  = list(bounds.keys())
        self.param_ranges = np.array([bounds[k] for k in self.param_names])

        self._init_population(bounds, initial_population)
        self._evaluate_all(fitness_fn)
        self._update_personal_best()
        self._update_global_best()

        no_improve_count = 0
        best_energy_seen = self.gbest_energy

        start_time = time.perf_counter()
        iteration = 0
        while iteration <= self.max_iterations:
            # Decreasing function of time for weight based on Eberhart paper
            self.w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iterations)

            self._update_velocities_and_positions()
            self._evaluate_all(fitness_fn)
            self._update_personal_best()
            self._update_global_best()

            # logging
            self.history["gbest_energy"].append(self.gbest_energy)
            self.history["gbest_position"].append(self.gbest_position.copy())

            # early stopping
            if self.gbest_energy < best_energy_seen:
                best_energy_seen = self.gbest_energy
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
        self.best_params  = self.gbest_position
        self.best_fitness = self.gbest_energy
