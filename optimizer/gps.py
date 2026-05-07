import numpy as np
import random
class GPS:
    def __init__(self, initial_step_size,
            tolerance, decay_rate, max_iterations=50):
        # hyperparameters
        self.initial_step_size = initial_step_size # alpha
        self.tolerance = tolerance # epsilon
        self.decay_rate = decay_rate # gamma
        self.max_iterations = max_iterations
        pass

    # --- setup ---

    def _initiate_solution(self, bounds):
        """
        Example of bounds = {
            "window": (2, 200),
            "alpha":  (0.0, 1.0)
        }
        """
        lows  = self.param_ranges[:, 0]  # all minimums
        highs = self.param_ranges[:, 1]  # all maximums
        
        # Generate single solution, not candidate solution
        self.candidate_solution = lows + np.random.rand(len(self.param_names)) * (highs - lows)

    # --- core GPS mechanics ---

    def get_best_params(self):
        return dict(zip(self.param_names, self.best_params))
        
    # --- main entry point ---
    def run(self, fitness_fn, D, bounds):
        self.history = {
            "gbest_energy": [],
            "position": [],
        }

        # I pass D as regular python list, needs to convert ot numpy array
        D = [np.array(d, dtype=float) for d in D]

        self.param_names = list(bounds.keys())
        self.param_ranges = np.array([bounds[k] for k in self.param_names])

        lows  = self.param_ranges[:, 0]  # all minimums
        highs = self.param_ranges[:, 1]  # all maximums

        self._initiate_solution(bounds)

        step_size = self.initial_step_size
        current_solution = self.candidate_solution
        current_fitness = fitness_fn(current_solution)
        iteration = 0
        while step_size > self.tolerance:
            if iteration > self.max_iterations:
                break
            improved = False
            for (i,d) in enumerate(D):
                candidate_new = current_solution + step_size * d

                # Clip to prevent invalid solution
                candidate_new = np.clip(candidate_new, lows, highs)

                fitness_new = fitness_fn(candidate_new)
                # print(f"     DEBUG:: Candidate fitness after movement {d}({candidate_new}): {fitness_new}")
                # > because we aim to maximize
                if (fitness_new > current_fitness):
                    current_solution, current_fitness, improved = \
                        candidate_new, fitness_new, True
                    D.pop(i)
                    D.insert(0, d)
                    break
            if not improved:
                step_size *= self.decay_rate

            self.history["gbest_energy"].append(current_fitness)
            self.history["position"].append(current_solution.copy())

            iteration += 1
            # print(f"DEBUG::: step: {step_size}; D: {D}")
            # print(f"Position in iteration {iteration}: {dict(zip(self.param_names, current_solution))}")

        self.best_params  = current_solution
        self.best_fitness = current_fitness
                
