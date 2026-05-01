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

        self.max_evals = max_evals
        self.n_shells = n_shells # This is for max number of n shells not fixed
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

        # Eq(3) sort the electrons
        sorted_indices = np.argsort(self.energy)
        
        # Update the nucleus position first
        self._update_nucleus(sorted_indices)
    
        # remaining candidates get split into shells
        remaining = sorted_indices[1:]  # exclude the copied nucleus

        # generate random integer n shells or rings (page 5),
        #  bounded by min(n_shells and population) size
        n = np.random.randint(1, min(self.n_shells, len(remaining)) + 1)
        self.shells = np.array_split(remaining, n)

    def _absorption(self, electron_idx, bounds):
        # electron absorbs energy = moves to outer shell (exploration EQ.11) 
        pass

    def _emission(self, electron_idx, bounds):
        # electron emits energy = moves to inner shell (exploitation EQ.10)
        pass

    def _update_nucleus(self, sorted_indices):
        # track best solution found so far
        # nucleus = single best candidate (LE)
        self.nucleus_idx = sorted_indices[0]
        self.LE = self.candidate_solutions[self.nucleus_idx].copy()

        # SELF_NOTE: This is my suspect that might cause this to not perform as best
        #   LE (Best candidate) is replaced every epoch without any memory
        #   This mean if the solution get worse, those best solution will just get lost

    def _calculate_binding(self, candidate, energy):
        binding_state = np.mean(candidate, axis=0)
        binding_energy = np.mean(energy)
        return binding_state, binding_energy

    def _calculate_global_binding(self):
        self.BS, self.BE = self._calculate_binding(self.candidate_solutions, self.energy)
        
    def _calculate_shell_binding(self, shell):
        shell_candidates = self.candidate_solutions[shell]
        shell_energies   = self.energy[shell]
        BSk, BEk = self._calculate_binding(shell_candidates, shell_energies)
        LEk = shell_candidates[np.argmin(shell_energies)]
        return BSk, BEk, LEk
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

        self._calculate_global_binding()
        
        # While (iteration < Maximum number of iterations):
        self._assign_shells()
        print(f"BS, BE, LE = {self.BS} {self.BE} {self.LE}")
        # Iterate over shells
        for k, shell in enumerate(self.shells):
            BSk, BEk, LEk = self._calculate_shell_binding(shell)
            # Iterate over candidate solutions in shells

    
