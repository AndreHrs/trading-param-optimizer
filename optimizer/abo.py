import numpy as np
import time


class ABO:
    """
    African Buffalo Optimization (ABO) Optimizer.

    Adapted from: Odili et al., "African Buffalo Optimization: A Swarm-Intelligence Technique",
    Procedia Computer Science 76, 2015, pp. 443-448.

    The algorithm models two buffalo vocalisations:
        maaa (m.k) — exploitation vector: tendency to stay and exploit a good region
        waaa (w.k) — position: tendency to explore new regions

    Eq. 1 (maaa update):
        m.k+1 = m.k + lp1 * (bgmax - w.k) + lp2 * (bpmax.k - w.k)

    Eq. 2 (waaa/position update, ±0.5 interpreted as averaging):
        w.k+1 = (w.k + m.k) / 2

    When the global best stagnates for stagnation_limit iterations the entire herd
    is re-initialised (positions and exploitation vectors reset), while the global
    best found so far is preserved.
    """

    def __init__(self, pop_size, max_iterations, lp1=0.5, lp2=0.5, stagnation_limit=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.lp1 = lp1
        self.lp2 = lp2
        self.stagnation_limit = stagnation_limit

        self.candidate_solutions = []
        self.exploitation_vectors = []
        self.personal_best_positions = []
        self.personal_best_fitness = []
        self.fitness = []
        self.param_names = []
        self.param_ranges = []
        self.best_params = {}
        self.best_fitness = float('inf')
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.history = {}

        self.epoch_count = 0
        self.time = 0.0
        self.early_stop = False

    # --- setup ---

    def _init_population(self, bounds, initial_population=None):
        """
        Randomly place all buffalos within bounds and zero their exploitation vectors.

        Example bounds = {
            "short_window": (2, 50),
            "alpha":        (0.01, 0.99),
        }
        """
        lows  = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        if initial_population is not None:
            self.candidate_solutions = np.array(initial_population, dtype=float)
        else:
            self.candidate_solutions = lows + np.random.rand(self.pop_size, len(self.param_names)) * (highs - lows)
        self.exploitation_vectors   = np.zeros((self.pop_size, len(self.param_names)))
        self.personal_best_positions = self.candidate_solutions.copy()
        self.personal_best_fitness   = np.full(self.pop_size, np.inf)

    def _evaluate_all(self, fitness_fn):
        self.fitness = np.array([fitness_fn(row) for row in self.candidate_solutions])

    # --- core ABO mechanics ---

    def _update_personal_best(self):
        improved = self.fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.candidate_solutions[improved].copy()
        self.personal_best_fitness[improved]   = self.fitness[improved]

    def _update_global_best(self):
        """Update the herd's global best. Returns True if improved."""
        best_idx = np.argmin(self.personal_best_fitness)
        candidate_best_fitness = self.personal_best_fitness[best_idx]
        if candidate_best_fitness < self.global_best_fitness:
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            self.global_best_fitness  = candidate_best_fitness
            return True
        return False

    def _update_buffalo_positions(self):
        """Apply Eq. 1 (maaa update) then Eq. 2 (position update), fully vectorised."""
        lows  = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        # Eq. 1: pull exploitation vector toward global best and personal best
        self.exploitation_vectors = (
            self.exploitation_vectors
            + self.lp1 * (self.global_best_position - self.candidate_solutions)
            + self.lp2 * (self.personal_best_positions - self.candidate_solutions)
        )

        # Eq. 2: new position is midpoint of current position and exploitation vector
        new_positions = (self.candidate_solutions + self.exploitation_vectors) / 2.0
        self.candidate_solutions = np.clip(new_positions, lows, highs)

    def get_best_params(self):
        return dict(zip(self.param_names, self.best_params))

    # --- main entry point ---

    def run(self, fitness_fn, bounds, initial_population=None):
        self.history = {
            "gbest_energy":   [],
            "gbest_position": [],
            "population":     [],
        }
        self.param_names  = list(bounds.keys())
        self.param_ranges = np.array([bounds[k] for k in self.param_names])

        self.global_best_fitness = float('inf')
        self.global_best_position = None

        self._init_population(bounds, initial_population)
        self._evaluate_all(fitness_fn)
        self._update_personal_best()
        self._update_global_best()

        stagnation_count = 0
        iteration = 0

        start_time = time.perf_counter()
        while iteration <= self.max_iterations:
            self._update_buffalo_positions()
            self._evaluate_all(fitness_fn)
            self._update_personal_best()
            improved = self._update_global_best()

            if improved:
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Re-initialise entire herd when global best stagnates, but preserve it
            if stagnation_count >= self.stagnation_limit:
                self._init_population(bounds)
                self._evaluate_all(fitness_fn)
                self._update_personal_best()
                self._update_global_best()
                stagnation_count = 0

            self.history["gbest_energy"].append(self.global_best_fitness)
            self.history["gbest_position"].append(self.global_best_position.copy())

            iteration += 1

        self.epoch_count = iteration
        self.time = (time.perf_counter() - start_time) * 1000
        self.best_params  = self.global_best_position
        self.best_fitness = self.global_best_fitness
