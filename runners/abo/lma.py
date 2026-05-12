from optimizer.abo import ABO
from optimizer.evaluator import evaluate
from utilities.filters import wma, lma_filter

BOUNDS = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
}


def get_signals(best_params, prices):
    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))
    short = wma(prices, short_n, lma_filter(short_n))
    long  = wma(prices, long_n,  lma_filter(long_n))
    return short, long


def run(prices, pop_size=40, max_iter=50, lp1=0.5, lp2=0.5, stagnation_limit=10, initial_population=None):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        short = wma(prices, short_n, lma_filter(short_n))
        long  = wma(prices, long_n,  lma_filter(long_n))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    abo = ABO(pop_size, max_iter, lp1, lp2, stagnation_limit)
    abo.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return abo
