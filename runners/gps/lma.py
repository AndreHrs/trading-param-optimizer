from optimizer.gps import GPS
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_LMA as BOUNDS, get_signals_lma as get_signals, _lma_signals


def run(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], seed=None, initial_position=None):
    def fitness(candidate):
        short, long = _lma_signals(prices, candidate[0], candidate[1])
        cash, *_ = evaluate(prices, short, long)
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations, seed=seed)
    gps.run(fitness, D=D, bounds=BOUNDS, initial_position=initial_position)
    return gps
