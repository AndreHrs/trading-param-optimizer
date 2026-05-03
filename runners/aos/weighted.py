from optimizer.aos import AOS
from optimizer.evaluator import evaluate
from utilities.filters import wma, sma_filter, lma_filter, ema_filter

BOUNDS = {
    "high_w1":     (0.0, 1.0),
    "high_w2":     (0.0, 1.0),
    "high_w3":     (0.0, 1.0),
    "high_d1":     (2, 50),
    "high_d2":     (2, 50),
    "high_d3":     (2, 50),
    "high_alpha3": (0.01, 0.99),

    "low_w1":      (0.0, 1.0),
    "low_w2":      (0.0, 1.0),
    "low_w3":      (0.0, 1.0),
    "low_d1":      (2, 50),
    "low_d2":      (2, 50),
    "low_d3":      (2, 50),
    "low_alpha3":  (0.01, 0.99),
}


def _calculate_weighted_signal(prices, w1, w2, w3, d1, d2, d3, alpha):
    """Weighted blend of SMA, LMA, and EMA signals (Eq. 7)."""
    signal_sma = wma(prices, d1, sma_filter(d1))
    signal_lma = wma(prices, d2, lma_filter(d2))
    signal_ema = wma(prices, d3, ema_filter(d3, alpha))
    weight_sum = w1 + w2 + w3
    return (w1 * signal_sma + w2 * signal_lma + w3 * signal_ema) / weight_sum


def get_signals(best_params, prices):
    high = _calculate_weighted_signal(
        prices,
        best_params["high_w1"], best_params["high_w2"], best_params["high_w3"],
        int(round(best_params["high_d1"])),
        int(round(best_params["high_d2"])),
        int(round(best_params["high_d3"])),
        best_params["high_alpha3"],
    )
    low = _calculate_weighted_signal(
        prices,
        best_params["low_w1"], best_params["low_w2"], best_params["low_w3"],
        int(round(best_params["low_d1"])),
        int(round(best_params["low_d2"])),
        int(round(best_params["low_d3"])),
        best_params["low_alpha3"],
    )
    return high, low


def run(prices, pop_size=100, max_iter=50, n_shells=5):
    def fitness(candidate):
        high = _calculate_weighted_signal(
            prices,
            candidate[0], candidate[1], candidate[2],
            int(round(candidate[3])), int(round(candidate[4])), int(round(candidate[5])),
            candidate[6],
        )
        low = _calculate_weighted_signal(
            prices,
            candidate[7], candidate[8], candidate[9],
            int(round(candidate[10])), int(round(candidate[11])), int(round(candidate[12])),
            candidate[13],
        )
        cash, *_ = evaluate(prices, high, low)
        return -cash

    aos = AOS(pop_size, max_iter, n_shells)
    aos.run(fitness, bounds=BOUNDS)
    return aos
