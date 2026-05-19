from optimizer.abo import ABO
from optimizer.evaluator import evaluate
from runners.shared import (
    get_signals_ema_shared as get_signals_shared,
    get_signals_ema_independent as get_signals_independent,
    _ema_signals_shared, _ema_signals_independent,
    BOUNDS_EMA_SHARED as BOUNDS_SHARED,
    BOUNDS_EMA_INDEPENDENT as BOUNDS_INDEPENDENT,
)


def run_shared(prices, pop_size=40, max_iter=50, lp1=0.5, lp2=0.5, stagnation_limit=10, initial_population=None):
    def fitness(candidate):
        short, long = _ema_signals_shared(prices, candidate[0], candidate[1], candidate[2])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    abo = ABO(pop_size, max_iter, lp1, lp2, stagnation_limit)
    abo.run(fitness, bounds=BOUNDS_SHARED, initial_population=initial_population)
    return abo


def run_independent(prices, pop_size=40, max_iter=50, lp1=0.5, lp2=0.5, stagnation_limit=10, initial_population=None):
    def fitness(candidate):
        short, long = _ema_signals_independent(prices, candidate[0], candidate[1], candidate[2], candidate[3])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    abo = ABO(pop_size, max_iter, lp1, lp2, stagnation_limit)
    abo.run(fitness, bounds=BOUNDS_INDEPENDENT, initial_population=initial_population)
    return abo
