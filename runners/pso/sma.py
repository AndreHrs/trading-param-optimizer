from optimizer.pso import PSO
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


def run(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, initial_population=None):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        short = wma(prices, short_n, sma_filter(short_n))
        long  = wma(prices, long_n,  sma_filter(long_n))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2)
    pso.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return pso
