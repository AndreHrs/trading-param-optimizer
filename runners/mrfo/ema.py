from optimizer.mrfo import MRFO
from optimizer.evaluator import evaluate
from utilities.filters import wma, ema_filter

# --- bounds, used 0.01 to 0.99 to avoid extreme, as ranging from 1e-6 to 1.0 causing no trade ---
BOUNDS_SHARED = {
    "short_window": (5, 40),
    "long_window":  (41, 120),
    "alpha":        (0.01, 0.99),
}

BOUNDS_INDEPENDENT = {
    "short_window": (5, 40),
    "long_window":  (41, 120),
    "alpha_short":  (0.01, 0.99), 
    "alpha_long":   (0.01, 0.99),
}

# --- signal generation ---
def get_signals_shared(best_params, prices):

    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))
    alpha   = best_params["alpha"]

    short = wma(prices, short_n, ema_filter(short_n, alpha))
    long  = wma(prices, long_n,  ema_filter(long_n, alpha))

    return short, long

def get_signals_independent(best_params, prices):

    short_n     = int(round(best_params["short_window"]))
    long_n      = int(round(best_params["long_window"]))
    alpha_short = best_params["alpha_short"]
    alpha_long  = best_params["alpha_long"]

    short = wma(prices, short_n, ema_filter(short_n, alpha_short))
    long  = wma(prices, long_n,  ema_filter(long_n, alpha_long))

    return short, long

# --- fitness func ---
def _fitness_shared(candidate, prices):

    short_n = int(round(candidate[0]))
    long_n  = int(round(candidate[1]))
    alpha   = candidate[2]

    short = wma(prices, short_n, ema_filter(short_n, alpha))
    long  = wma(prices, long_n,  ema_filter(long_n, alpha))

    cash, *_ = evaluate(prices, short, long)

    return -cash   # unified minimization

def _fitness_independent(candidate, prices):

    short_n     = int(round(candidate[0]))
    long_n      = int(round(candidate[1]))
    alpha_short = candidate[2]
    alpha_long  = candidate[3]

    short = wma(prices, short_n, ema_filter(short_n, alpha_short))
    long  = wma(prices, long_n,  ema_filter(long_n, alpha_long))

    cash, *_ = evaluate(prices, short, long)

    return -cash   # unified minimization

# --- runner, i adapted from the AOS structure ---
def run_shared(prices, pop_size=100, max_iter=50, somersault=2.0):

    def fitness(candidate):
        return _fitness_shared(candidate, prices)

    mrfo = MRFO(
        pop_size=pop_size,
        max_iterations=max_iter,   # FIXED (consistent w/ MRFO core)
        somersault_range=somersault
    )

    mrfo.run(fitness, bounds=BOUNDS_SHARED)

    return mrfo

def run_independent(prices, pop_size=100, max_iter=50, somersault=2.0):

    def fitness(candidate):
        return _fitness_independent(candidate, prices)

    mrfo = MRFO(
        pop_size=pop_size,
        max_iterations=max_iter,
        somersault_range=somersault
    )

    mrfo.run(fitness, bounds=BOUNDS_INDEPENDENT)

    return mrfo