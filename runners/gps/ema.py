from optimizer.gps import GPS
from optimizer.evaluator import evaluate
from utilities.filters import wma, ema_filter

BOUNDS_INDEPENDENT = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha_short":  (1e-6, 1),
    "alpha_long":   (1e-6, 1),
}

def get_signals_independent(best_params, prices):
    short_n     = int(round(best_params["short_window"]))
    long_n      = int(round(best_params["long_window"]))
    alpha_short = best_params["alpha_short"]
    alpha_long  = best_params["alpha_long"]
    short = wma(prices, short_n, ema_filter(short_n, alpha_short))
    long  = wma(prices, long_n,  ema_filter(long_n,  alpha_long))
    return short, long

def run_independent(prices, initial_step_size = 1,
    tolerance = 1e-5, decay_rate = 0.5, max_iterations=50, D = []):
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
    gps.run(fitness, D = D, bounds=BOUNDS_INDEPENDENT)
    return gps
