from optimizer.gps import GPS
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_WEIGHTED as BOUNDS, get_signals_weighted as get_signals, _weighted_signal


def run(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], seed=None, initial_position=None):
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
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations, seed=seed)
    gps.run(fitness, D=D, bounds=BOUNDS, initial_position=initial_position)
    return gps
