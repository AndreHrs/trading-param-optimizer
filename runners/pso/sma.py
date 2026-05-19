from optimizer.pso import PSO
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_SMA as BOUNDS, get_signals_sma as get_signals, _sma_signals


def run(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1, initial_population=None):
    def fitness(candidate):
        short, long = _sma_signals(prices, candidate[0], candidate[1])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2, max_vel_frac)
    pso.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return pso
