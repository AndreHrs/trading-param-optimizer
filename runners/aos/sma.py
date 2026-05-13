from optimizer.aos import AOS
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


def run(prices, pop_size=100, max_iter=50, n_shells=5, initial_population=None):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        short = wma(prices, short_n, sma_filter(short_n))
        long  = wma(prices, long_n,  sma_filter(long_n))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    aos = AOS(pop_size, max_iter, n_shells)
    aos.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return aos
