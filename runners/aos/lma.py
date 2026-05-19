from optimizer.aos import AOS
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_LMA as BOUNDS, get_signals_lma as get_signals, _lma_signals


def run(prices, pop_size=100, max_iter=50, n_shells=5, initial_population=None):
    def fitness(candidate):
        short, long = _lma_signals(prices, candidate[0], candidate[1])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    aos = AOS(pop_size, max_iter, n_shells)
    aos.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return aos
