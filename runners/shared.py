from optimizer.evaluator import evaluate
from utilities.filters import wma, sma_filter, lma_filter, ema_filter

BOUNDS_SMA = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
}

BOUNDS_LMA = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
}

BOUNDS_EMA_SHARED = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha":        (0.01, 0.99),
}

BOUNDS_EMA_INDEPENDENT = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha_short":  (0.01, 0.99),
    "alpha_long":   (0.01, 0.99),
}

BOUNDS_WEIGHTED = {
    "high_w1":     (0.05, 1.0),
    "high_w2":     (0.05, 1.0),
    "high_w3":     (0.05, 1.0),
    "high_d1":     (2, 50),
    "high_d2":     (2, 50),
    "high_d3":     (2, 50),
    "high_alpha3": (0.01, 0.99),
    "low_w1":      (0.05, 1.0),
    "low_w2":      (0.05, 1.0),
    "low_w3":      (0.05, 1.0),
    "low_d1":      (2, 50),
    "low_d2":      (2, 50),
    "low_d3":      (2, 50),
    "low_alpha3":  (0.01, 0.99),
}



def re_evaluate(best_params, prices, get_signals_fn):
    short, long = get_signals_fn(best_params, prices)
    cash, buy_at, sell_at, equity_curve = evaluate(prices, short, long)
    return short, long, buy_at, sell_at, equity_curve, cash


# Private signal computation helpers
def _sma_signals(prices, short_n, long_n):
    sn, ln = int(round(short_n)), int(round(long_n))
    return wma(prices, sn, sma_filter(sn)), wma(prices, ln, sma_filter(ln))


def _lma_signals(prices, short_n, long_n):
    sn, ln = int(round(short_n)), int(round(long_n))
    return wma(prices, sn, lma_filter(sn)), wma(prices, ln, lma_filter(ln))


def _ema_signals_shared(prices, short_n, long_n, alpha):
    sn, ln = int(round(short_n)), int(round(long_n))
    return wma(prices, sn, ema_filter(sn, alpha)), wma(prices, ln, ema_filter(ln, alpha))


def _ema_signals_independent(prices, short_n, long_n, alpha_short, alpha_long):
    sn, ln = int(round(short_n)), int(round(long_n))
    return wma(prices, sn, ema_filter(sn, alpha_short)), wma(prices, ln, ema_filter(ln, alpha_long))

def _weighted_signal(prices, w1, w2, w3, d1, d2, d3, alpha):
    s1 = wma(prices, d1, sma_filter(d1))
    s2 = wma(prices, d2, lma_filter(d2))
    s3 = wma(prices, d3, ema_filter(d3, alpha))
    w_sum = w1 + w2 + w3 + 1e-12
    return (w1 * s1 + w2 * s2 + w3 * s3) / w_sum

# Public get_signals_*

def get_signals_sma(best_params, prices):
    return _sma_signals(prices, best_params["short_window"], best_params["long_window"])


def get_signals_lma(best_params, prices):
    return _lma_signals(prices, best_params["short_window"], best_params["long_window"])


def get_signals_ema_shared(best_params, prices):
    return _ema_signals_shared(
        prices, best_params["short_window"], best_params["long_window"], best_params["alpha"]
    )


def get_signals_ema_independent(best_params, prices):
    return _ema_signals_independent(
        prices, best_params["short_window"], best_params["long_window"],
        best_params["alpha_short"], best_params["alpha_long"],
    )

def get_signals_weighted(best_params, prices):
    high = _weighted_signal(
        prices,
        best_params["high_w1"], best_params["high_w2"], best_params["high_w3"],
        int(round(best_params["high_d1"])),
        int(round(best_params["high_d2"])),
        int(round(best_params["high_d3"])),
        best_params["high_alpha3"],
    )
    low = _weighted_signal(
        prices,
        best_params["low_w1"], best_params["low_w2"], best_params["low_w3"],
        int(round(best_params["low_d1"])),
        int(round(best_params["low_d2"])),
        int(round(best_params["low_d3"])),
        best_params["low_alpha3"],
    )
    return high, low
