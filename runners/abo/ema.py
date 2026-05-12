from optimizer.abo import ABO
from optimizer.evaluator import evaluate
from utilities.filters import wma, ema_filter

BOUNDS_SHARED = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha":        (1e-6, 1),
}

BOUNDS_INDEPENDENT = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha_short":  (1e-6, 1),
    "alpha_long":   (1e-6, 1),
}


def get_signals_shared(best_params, prices):
    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))
    alpha   = best_params["alpha"]
    short = wma(prices, short_n, ema_filter(short_n, alpha))
    long  = wma(prices, long_n,  ema_filter(long_n,  alpha))
    return short, long


def get_signals_independent(best_params, prices):
    short_n     = int(round(best_params["short_window"]))
    long_n      = int(round(best_params["long_window"]))
    alpha_short = best_params["alpha_short"]
    alpha_long  = best_params["alpha_long"]
    short = wma(prices, short_n, ema_filter(short_n, alpha_short))
    long  = wma(prices, long_n,  ema_filter(long_n,  alpha_long))
    return short, long


def run_shared(prices, pop_size=40, max_iter=50, lp1=0.5, lp2=0.5, stagnation_limit=10):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        alpha   = candidate[2]
        short = wma(prices, short_n, ema_filter(short_n, alpha))
        long  = wma(prices, long_n,  ema_filter(long_n,  alpha))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    abo = ABO(pop_size, max_iter, lp1, lp2, stagnation_limit)
    abo.run(fitness, bounds=BOUNDS_SHARED)
    return abo


def run_independent(prices, pop_size=40, max_iter=50, lp1=0.5, lp2=0.5, stagnation_limit=10):
    def fitness(candidate):
        short_n     = int(round(candidate[0]))
        long_n      = int(round(candidate[1]))
        alpha_short = candidate[2]
        alpha_long  = candidate[3]
        short = wma(prices, short_n, ema_filter(short_n, alpha_short))
        long  = wma(prices, long_n,  ema_filter(long_n,  alpha_long))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    abo = ABO(pop_size, max_iter, lp1, lp2, stagnation_limit)
    abo.run(fitness, bounds=BOUNDS_INDEPENDENT)
    return abo
