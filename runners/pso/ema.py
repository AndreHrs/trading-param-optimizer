from optimizer.pso import PSO
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


def run_shared(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        alpha   = candidate[2]
        short = wma(prices, short_n, ema_filter(short_n, alpha))
        long  = wma(prices, long_n,  ema_filter(long_n,  alpha))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2)
    pso.run(fitness, bounds=BOUNDS_SHARED)
    return pso


def run_independent(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1):
    def fitness(candidate):
        short_n     = int(round(candidate[0]))
        long_n      = int(round(candidate[1]))
        alpha_short = candidate[2]
        alpha_long  = candidate[3]
        short = wma(prices, short_n, ema_filter(short_n, alpha_short))
        long  = wma(prices, long_n,  ema_filter(long_n,  alpha_long))
        cash, *_ = evaluate(prices, short, long)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2)
    pso.run(fitness, bounds=BOUNDS_INDEPENDENT)
    return pso
