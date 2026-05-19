from optimizer.gps import GPS
from optimizer.evaluator import evaluate
from runners.shared import (
    get_signals_ema_shared as get_signals_shared,
    get_signals_ema_independent as get_signals_independent,
    _ema_signals_shared, _ema_signals_independent,
    BOUNDS_EMA_SHARED as BOUNDS_SHARED,
    BOUNDS_EMA_INDEPENDENT as BOUNDS_INDEPENDENT,
)


def run_shared(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], seed=None, initial_position=None):
    def fitness(candidate):
        short, long = _ema_signals_shared(prices, candidate[0], candidate[1], candidate[2])
        cash, *_ = evaluate(prices, short, long)
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations, seed=seed)
    gps.run(fitness, D=D, bounds=BOUNDS_SHARED, initial_position=initial_position)
    return gps


def run_independent(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], seed=None, initial_position=None):
    def fitness(candidate):
        short, long = _ema_signals_independent(prices, candidate[0], candidate[1], candidate[2], candidate[3])
        cash, *_ = evaluate(prices, short, long)
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations, seed=seed)
    gps.run(fitness, D=D, bounds=BOUNDS_INDEPENDENT, initial_position=initial_position)
    return gps
