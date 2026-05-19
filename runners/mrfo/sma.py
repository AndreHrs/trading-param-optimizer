from optimizer.mrfo import MRFO
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_SMA as BOUNDS, get_signals_sma as get_signals, _sma_signals


def run(prices, pop_size=100, max_iter=50, somersault=2.0, initial_population=None):
    def fitness(candidate):
        short, long = _sma_signals(prices, candidate[0], candidate[1])
        cash, *_ = evaluate(prices, short, long)
        return -cash

    mrfo = MRFO(pop_size=pop_size, max_iterations=max_iter, somersault_range=somersault)
    mrfo.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return mrfo
