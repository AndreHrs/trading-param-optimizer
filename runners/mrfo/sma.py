from optimizer.mrfo import MRFO
from optimizer.evaluator import evaluate
from utilities.filters import wma, sma_filter

# --- bounds ---
BOUNDS = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
}

# --- signal generation ---
def get_signals(best_params, prices):

    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))

    short = wma(prices, short_n, sma_filter(short_n))
    long  = wma(prices, long_n,  sma_filter(long_n))

    return short, long

# --- fitness func ---
def _fitness(candidate, prices):

    short_n = int(round(candidate[0]))
    long_n  = int(round(candidate[1]))

    short = wma(prices, short_n, sma_filter(short_n))
    long  = wma(prices, long_n,  sma_filter(long_n))

    cash, *_ = evaluate(prices, short, long)

    return -cash   # unified minimisation

# --- runner ---
def run(prices, pop_size=100, max_iter=50, somersault=2.0):

    def fitness(candidate):
        return _fitness(candidate, prices)

    mrfo = MRFO(
        pop_size=pop_size,
        max_iterations=max_iter,
        somersault_range=somersault
    )

    mrfo.run(fitness, bounds=BOUNDS)

    return mrfo