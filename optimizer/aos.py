import numpy as np
import random
import time
class AOS:

    def __init__(self, pop_size, max_iterations, n_shells, photon_rate=0.5, patience=50):
        # hyperparameters
        self.candidate_solutions = []
        self.param_ranges =[]
        self.param_names = []
        self.best_params = {}
        self.best_fitness = {}
        self.history = {}
        self.pop_size = pop_size

        self.max_iterations = max_iterations
        self.n_shells = n_shells # This is for max number of n shells not fixed
        self.photon_rate = photon_rate
        self.patience = patience

        self.epoch_count = 0
        self.time = 0.0
        self.early_stop = False

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
        self.energy = np.array([fitness_fn(row) for row in self.candidate_solutions])

    # --- core AOS mechanics ---

    def _assign_shells(self):
        # sort candidates by fitness, assign to shells via PDF (Eq. 3 / Fig. 3)

        # Eq(3) sort the electrons
        sorted_indices = np.argsort(self.energy)
    
        # remaining candidates get split into shells
        remaining = sorted_indices[1:]  # exclude the copied nucleus

        # generate random integer n shells or rings (page 5),
        #  bounded by min(n_shells and population) size
        n = np.random.randint(1, min(self.n_shells, len(remaining)) + 1)
        self.shells = np.array_split(remaining, n)

    def _update_nucleus(self, sorted_indices):
        # track best solution found so far
        # nucleus = single best candidate (LE)
        self.nucleus_idx = sorted_indices[0]
        self.LE = self.candidate_solutions[self.nucleus_idx].copy()
        self.LE_energy = self.energy[self.nucleus_idx]

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
        LEk = shell_candidates[np.argmin(shell_energies)]  # best in this shell only
        return BSk, BEk, LEk

    def get_best_params(self):
        return dict(zip(self.param_names, self.best_params))
        
    # --- main entry point ---
    def run(self, fitness_fn, bounds):
        self.history = {
            "LE_energy":   [],
            "LE_position": [],
            "population":  [],
        }
        self.early_stop = False
        self.param_names = list(bounds.keys())
        self.param_ranges = np.array([bounds[k] for k in self.param_names])

        # init -> evaluate -> determine BS and BE, pick LE
        # loop: assign shells, absorption/emission, update nucleus
        # populates self.best_params, self.best_fitness, self.history, self.n_evals
        # returns self.best_params
        self._init_population(bounds)
        self._evaluate_all(fitness_fn)

        sorted_indices = np.argsort(self.energy)
        self._update_nucleus(sorted_indices)
        self._calculate_global_binding()
        self._assign_shells()

        no_improve_count = 0
        best_energy_seen = self.LE_energy

        start_time = time.perf_counter()
        iteration = 0
        while (iteration <= self.max_iterations):
            # clip to bounds so moves cannot go out of range
            lows  = self.param_ranges[:, 0]
            highs = self.param_ranges[:, 1]
            for k, shell in enumerate(self.shells):
                BSk, BEk, LEk = self._calculate_shell_binding(shell)

                for electron_idx in shell:
                    phi   = np.random.rand()
                    alpha = np.random.rand(len(self.param_names))
                    beta  = np.random.rand(len(self.param_names))
                    gamma = np.random.rand(len(self.param_names))

                    current = self.candidate_solutions[electron_idx]

                    if phi >= self.photon_rate:
                        # photon interaction to determine emission or absorption
                        if self.energy[electron_idx] >= BEk:
                            # Eq. 10 emission, pull toward global LE and BS
                            new = current + (alpha * (beta * self.LE - gamma * self.BS)) / (k + 1)
                        else:
                            # Eq. 11 absorption, pull toward shell LEk and BSk
                            new = current + alpha * (beta * LEk - gamma * BSk)
                    else:
                        # Eq. 12 random walk
                        new = current + np.random.rand(len(self.param_names))
                    self.candidate_solutions[electron_idx] = np.clip(new, lows, highs)

            # re-evaluate after all candidates have moved
            self._evaluate_all(fitness_fn)
            sorted_indices = np.argsort(self.energy)   # fresh sort
            self._update_nucleus(sorted_indices)
            self._calculate_global_binding()
            self._assign_shells()

            # logging
            self.history["LE_energy"].append(self.LE_energy)
            self.history["LE_position"].append(self.LE.copy())

            # early stopping
            if self.LE_energy < best_energy_seen:
                best_energy_seen = self.LE_energy
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
        self.best_params = self.LE
        self.best_fitness = self.LE_energy
                
