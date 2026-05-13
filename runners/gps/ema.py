from optimizer.gps import GPS
from optimizer.evaluator import evaluate
from utilities.filters import wma, ema_filter

BOUNDS_SHARED = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha":        (0.01, 0.99),
}

BOUNDS_INDEPENDENT = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha_short":  (0.01, 0.99),
    "alpha_long":   (0.01, 0.99),
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


def run_shared(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], initial_position=None):
    def fitness(candidate):
        short_n = int(round(candidate[0]))
        long_n  = int(round(candidate[1]))
        alpha   = candidate[2]
        short = wma(prices, short_n, ema_filter(short_n, alpha))
        long  = wma(prices, long_n,  ema_filter(long_n,  alpha))
        cash, *_ = evaluate(prices, short, long)
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations)
    gps.run(fitness, D=D, bounds=BOUNDS_SHARED, initial_position=initial_position)
    return gps


def run_independent(prices, initial_step_size=1, tolerance=1e-5, decay_rate=0.5, max_iterations=50, D=[], initial_position=None):
    def fitness(candidate):
        short_n     = int(round(candidate[0]))
        long_n      = int(round(candidate[1]))
        alpha_short = candidate[2]
        alpha_long  = candidate[3]
        short = wma(prices, short_n, ema_filter(short_n, alpha_short))
        long  = wma(prices, long_n,  ema_filter(long_n,  alpha_long))
        cash, *_ = evaluate(prices, short, long)
        return cash

    gps = GPS(initial_step_size, tolerance, decay_rate, max_iterations)
    gps.run(fitness, D=D, bounds=BOUNDS_INDEPENDENT, initial_position=initial_position)
    return gps
