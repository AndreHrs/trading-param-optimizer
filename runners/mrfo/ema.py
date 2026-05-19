from optimizer.mrfo import MRFO
from optimizer.evaluator import evaluate
from runners.shared import (
    get_signals_ema_shared as get_signals_shared,
    get_signals_ema_independent as get_signals_independent,
    _ema_signals_shared, _ema_signals_independent,
)

# --- bounds, used 0.01 to 0.99 to avoid extreme, as ranging from 1e-6 to 1.0 causing no trade ---
BOUNDS_SHARED = {
    "short_window": (5, 40),
    "long_window":  (41, 120),
    "alpha":        (0.01, 0.99),
}

BOUNDS_INDEPENDENT = {
    "short_window": (5, 40),
    "long_window":  (41, 120),
    "alpha_short":  (0.01, 0.99),
    "alpha_long":   (0.01, 0.99),
}


def run_shared(prices, pop_size=100, max_iter=50, somersault=2.0, seed=None, initial_population=None):
    def fitness(candidate):
        short, long = _ema_signals_shared(prices, candidate[0], candidate[1], candidate[2])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    mrfo = MRFO(pop_size=pop_size, max_iterations=max_iter, somersault_range=somersault, seed=seed)
    mrfo.run(fitness, bounds=BOUNDS_SHARED, initial_population=initial_population)
    return mrfo


def run_independent(prices, pop_size=100, max_iter=50, somersault=2.0, seed=None, initial_population=None):
    def fitness(candidate):
        short, long = _ema_signals_independent(prices, candidate[0], candidate[1], candidate[2], candidate[3])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    mrfo = MRFO(pop_size=pop_size, max_iterations=max_iter, somersault_range=somersault, seed=seed)
    mrfo.run(fitness, bounds=BOUNDS_INDEPENDENT, initial_population=initial_population)
    return mrfo
