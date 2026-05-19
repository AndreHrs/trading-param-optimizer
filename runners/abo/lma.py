from optimizer.abo import ABO
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_LMA as BOUNDS, get_signals_lma as get_signals, _lma_signals


def run(prices, pop_size=40, max_iter=50, lp1=0.5, lp2=0.5, stagnation_limit=10, initial_population=None):
    def fitness(candidate):
        short, long = _lma_signals(prices, candidate[0], candidate[1])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    abo = ABO(pop_size, max_iter, lp1, lp2, stagnation_limit)
    abo.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return abo
