import numpy as np
import time

class MRFO:
    """
    Manta Ray Foraging Optimization (MRFO)
    Designed to be API-compatible with the existing implementation.

    Supports:
    - continuous optimization
    - bounded search
    - global elitism memory
    - convergence tracking

    Intended usage:
        mrfo = MRFO(pop_size=30, max_iterations=100)

        mrfo.run(fitness_fn, bounds)

        print(mrfo.get_best_params())
        print(mrfo.best_fitness)
    """

    def __init__(
        self,
        pop_size,
        max_iterations,
        somersault_range=2.0,
        exploration_prob=0.5,
        seed=None,
        patience=50
    ):

        # hyperparameters
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.somersault_range = somersault_range
        self.exploration_prob = exploration_prob
        self.patience = patience

        # reproducibility
        if seed is not None:
            np.random.seed(seed)

        # optimizer state
        self.population = None
        self.fitness = None

        # parameter metadata
        self.param_names = []
        self.param_ranges = None

        # best solution tracking
        self.best_position = None
        self.best_fitness = np.inf

        # public compatibility fields
        self.best_params = {}
        self.history = {}

        self.epoch_count = 0
        self.time = 0.0
        self.early_stop = False

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _init_population(self, bounds, initial_population=None):

        self.param_names = list(bounds.keys())
        self.param_ranges = np.array(
            [bounds[k] for k in self.param_names],
            dtype=float
        )

        lows = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        if initial_population is not None:
            self.population = np.array(initial_population, dtype=float)
        else:
            self.population = (
                lows
                + np.random.rand(self.pop_size, len(self.param_names))
                * (highs - lows)
            )

    # ============================================================
    # BOUNDARY HANDLING
    # ============================================================

    def _clip_to_bounds(self, X):

        lows = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        return np.clip(X, lows, highs)

    # ============================================================
    # PARAMETER SANITIZATION
    # ============================================================

    def _sanitize_candidate(self, candidate):
        """
        Convert parameters into valid strategy inputs.

        - moving average windows should be integers
        - alpha should remain float
        """

        candidate = candidate.copy()

        for i, name in enumerate(self.param_names):

            lname = name.lower()

            # common integer parameter naming
            if (
                "window" in lname
                or "period" in lname
                or "short" in lname
                or "long" in lname
                or "length" in lname
                or "n_" in lname
            ):
                candidate[i] = int(round(candidate[i]))

                # ensure >= 1
                candidate[i] = max(1, candidate[i])

        return candidate

    # ============================================================
    # EVALUATION
    # ============================================================

    def _evaluate_all(self, fitness_fn):

        self.fitness = np.zeros(self.pop_size)

        for i in range(self.pop_size):

            candidate = self._sanitize_candidate(
                self.population[i]
            )

            try:
                fit = fitness_fn(candidate)

                # protect optimizer from invalid values
                if np.isnan(fit) or np.isinf(fit):
                    fit = np.inf

            except Exception:
                fit = np.inf

            self.fitness[i] = fit

        # update global best
        best_idx = np.argmin(self.fitness)

        if self.fitness[best_idx] < self.best_fitness:

            self.best_fitness = self.fitness[best_idx]

            self.best_position = (
                self.population[best_idx].copy()
            )

    # ============================================================
    # CHAIN FORAGING
    # ============================================================

    def _chain_foraging(self, iteration):

        new_population = self.population.copy()

        T = iteration / self.max_iterations

        for i in range(self.pop_size):

            r = np.random.rand()
            alpha = 2 * r * np.sqrt(abs(np.log(r + 1e-12)))

            if i == 0:
                leader = self.best_position
            else:
                leader = self.population[i - 1]

            current = self.population[i]

            # MRFO chain foraging equation
            new_position = (
                current
                + r * (leader - current)
                + alpha * (self.best_position - current)
            )

            new_population[i] = self._clip_to_bounds(
                new_position
            )

        self.population = new_population

    # ============================================================
    # CYCLONE FORAGING
    # ============================================================

    def _cyclone_foraging(self, iteration):

        new_population = self.population.copy()

        T = iteration / self.max_iterations

        lows = self.param_ranges[:, 0]
        highs = self.param_ranges[:, 1]

        for i in range(self.pop_size):

            r1 = np.random.rand()
            beta = 2 * np.exp(r1 * ((self.max_iterations - iteration)
                                    / self.max_iterations)) * np.sin(2 * np.pi * r1)

            current = self.population[i]

            # exploration vs exploitation
            if np.random.rand() < self.exploration_prob:

                # random target exploration
                random_position = (
                    lows
                    + np.random.rand(len(self.param_names))
                    * (highs - lows)
                )

                target = random_position

            else:

                # exploit around best
                target = self.best_position

            if i == 0:
                reference = target
            else:
                reference = self.population[i - 1]

            new_position = (
                target
                + r1 * (reference - current)
                + beta * (target - current)
            )

            new_population[i] = self._clip_to_bounds(
                new_position
            )

        self.population = new_population

    # ============================================================
    # SOMERSAULT FORAGING
    # ============================================================

    def _somersault_foraging(self):

        new_population = self.population.copy()

        for i in range(self.pop_size):

            r1 = np.random.rand()
            r2 = np.random.rand()

            current = self.population[i]

            # MRFO somersault equation
            new_position = (
                current
                + self.somersault_range
                * (
                    r1 * self.best_position
                    - r2 * current
                )
            )

            new_population[i] = self._clip_to_bounds(
                new_position
            )

        self.population = new_population

    # ============================================================
    # PUBLIC API
    # ============================================================
    
    def get_best_params(self):

        best = self._sanitize_candidate(
            self.best_position
        )

        clean = {}

        for k, v in zip(self.param_names, best):

            if isinstance(v, np.integer):
                clean[k] = int(v)

            elif isinstance(v, np.floating):
                clean[k] = float(v)

            else:
                clean[k] = v

        return clean

    # ============================================================
    # MAIN OPTIMIZATION LOOP
    # ============================================================

    def run(self, fitness_fn, bounds, initial_population=None):

        # initialize logs
        self.history = {
            "best_fitness": [],
            "best_position": [],
            "mean_fitness": [],
            "population": [],
        }
        self.early_stop = False

        # initialize population
        self._init_population(bounds, initial_population)

        # initial evaluation
        self._evaluate_all(fitness_fn)

        no_improve_count = 0
        best_energy_seen = self.best_fitness

        start_time = time.perf_counter()

        # optimization loop
        for iteration in range(self.max_iterations):

            # ----------------------------------------------------
            # chain foraging
            # ----------------------------------------------------
            self._chain_foraging(iteration)
            self._evaluate_all(fitness_fn)

            # ----------------------------------------------------
            # cyclone foraging
            # ----------------------------------------------------
            self._cyclone_foraging(iteration)
            self._evaluate_all(fitness_fn)

            # ----------------------------------------------------
            # somersault foraging
            # ----------------------------------------------------
            self._somersault_foraging()
            self._evaluate_all(fitness_fn)

            # ----------------------------------------------------
            # history tracking
            # ----------------------------------------------------
            self.history["best_fitness"].append(
                self.best_fitness
            )

            self.history["best_position"].append(
                self.best_position.copy()
            )

            self.history["mean_fitness"].append(
                np.mean(self.fitness)
            )

            self.history["population"].append(
                self.population.copy()
            )

            # early stopping
            if self.best_fitness < best_energy_seen:
                best_energy_seen = self.best_fitness
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= self.patience:
                self.early_stop = True
                self.epoch_count = iteration + 1
                break

        else:
            self.epoch_count = self.max_iterations

        self.time = (time.perf_counter() - start_time) * 1000

        # public compatibility
        self.best_params = self.get_best_params()

        return self