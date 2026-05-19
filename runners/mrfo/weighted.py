from optimizer.mrfo import MRFO
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_WEIGHTED as BOUNDS, get_signals_weighted as get_signals, _weighted_signal


def run(prices, pop_size=100, max_iter=50, somersault=2.0, initial_population=None):
    def fitness(candidate):
        high = _weighted_signal(
            prices,
            candidate[0], candidate[1], candidate[2],
            int(round(candidate[3])), int(round(candidate[4])), int(round(candidate[5])),
            candidate[6],
        )
        low = _weighted_signal(
            prices,
            candidate[7], candidate[8], candidate[9],
            int(round(candidate[10])), int(round(candidate[11])), int(round(candidate[12])),
            candidate[13],
        )
        cash, *_ = evaluate(prices, high, low)
        return -cash

    mrfo = MRFO(pop_size=pop_size, max_iterations=max_iter, somersault_range=somersault)
    mrfo.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return mrfo
