from optimizer.pso import PSO
from optimizer.evaluator import evaluate
from runners.shared import (
    get_signals_ema_shared as get_signals_shared,
    get_signals_ema_independent as get_signals_independent,
    _ema_signals_shared, _ema_signals_independent,
    BOUNDS_EMA_SHARED as BOUNDS_SHARED,
    BOUNDS_EMA_INDEPENDENT as BOUNDS_INDEPENDENT,
)


def run_shared(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1, initial_population=None):
    def fitness(candidate):
        short, long = _ema_signals_shared(prices, candidate[0], candidate[1], candidate[2])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2, max_vel_frac)
    pso.run(fitness, bounds=BOUNDS_SHARED, initial_population=initial_population)
    return pso


def run_independent(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1, initial_population=None):
    def fitness(candidate):
        short, long = _ema_signals_independent(prices, candidate[0], candidate[1], candidate[2], candidate[3])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2, max_vel_frac)
    pso.run(fitness, bounds=BOUNDS_INDEPENDENT, initial_population=initial_population)
    return pso
