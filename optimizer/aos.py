import numpy as np
class AOS:

    def __init__(self, pop_size, max_evals, n_shells):
        # hyperparameters
        self.candidate_solutions = []
        self.param_ranges =[]
        self.param_names = []
        self.best_params = {}
        self.best_fitness = {}
        self.history = {}
        self.n_evals = {}
        self.pop_size = pop_size
        pass

    # --- setup ---

    def _init_population(self, bounds):
        """
        Example of bounds = {
            "window": (2, 200),
            "alpha":  (0.0, 1.0)
        }
        """
        self.candidate_solutions = []

        lows  = self.param_ranges[:, 0]  # all minimums
        highs = self.param_ranges[:, 1]  # all maximums
        
        # Numpy equivalent for formula 2, no need for nested loops
        self.candidate_solutions = lows + np.random.rand(self.pop_size, len(self.param_names)) * (highs - lows)

    def _evaluate_all(self, fitness_fn):
        rounded_all = np.round(self.candidate_solutions).astype(int)  # shape (pop_size, d)
        self.energy = np.array([fitness_fn(row) for row in rounded_all])

    # --- core AOS mechanics ---

    def _assign_shells(self):
        # sort candidates by fitness, assign to shells via PDF (Eq. 3 / Fig. 3)
        pass

    def _absorption(self, electron_idx, bounds):
        # electron absorbs energy = moves to outer shell (exploration EQ.11) 
        pass

    def _emission(self, electron_idx, bounds):
        # electron emits energy = moves to inner shell (exploitation EQ.10)
        pass

    def _update_nucleus(self):
        # track best solution found so far
        pass

    # --- main entry point ---

    def run(self, fitness_fn, bounds):
        self.param_names = list(bounds.keys())
        self.param_ranges = np.array([bounds[k] for k in self.param_names])

        # init -> evaluate -> determine BS and BE, pick LE
        # loop: assign shells, absorption/emission, update nucleus
        # populates self.best_params, self.best_fitness, self.history, self.n_evals
        # returns self.best_params
        self._init_population(bounds)
        self._evaluate_all(fitness_fn)
        pass
