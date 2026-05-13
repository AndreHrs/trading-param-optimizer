from optimizer.gps import GPS
from optimizer.evaluator import evaluate
from utilities.filters import wma, sma_filter

BOUNDS = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
}


def get_signals(best_params, prices):
    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))
    short = wma(prices, short_n, sma_filter(short_n))
    long  = wma(prices, long_n,  sma_filter(long_n))
    return short, long


def run(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], initial_position=None):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        short = wma(prices, short_n, sma_filter(short_n))
        long  = wma(prices, long_n,  sma_filter(long_n))
        cash, *_ = evaluate(prices, short, long)
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations)
    gps.run(fitness, D=D, bounds=BOUNDS, initial_position=initial_position)
    return gps
