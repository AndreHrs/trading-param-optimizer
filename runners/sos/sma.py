from optimizer.sos import SOS
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


def run(prices, pop_size=100, max_iter=50):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        short = wma(prices, short_n, sma_filter(short_n))
        long  = wma(prices, long_n,  sma_filter(long_n))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    sos = SOS(pop_size, max_iter)
    sos.run(fitness, bounds=BOUNDS)
    return sos
