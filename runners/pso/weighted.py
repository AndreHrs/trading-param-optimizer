from optimizer.pso import PSO
from optimizer.evaluator import evaluate
from utilities.filters import wma, sma_filter, lma_filter, ema_filter

BOUNDS = {
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


def _weighted_signal(prices, w1, w2, w3, d1, d2, d3, alpha):
    s1 = wma(prices, d1, sma_filter(d1))
    s2 = wma(prices, d2, lma_filter(d2))
    s3 = wma(prices, d3, ema_filter(d3, alpha))
    w_sum = w1 + w2 + w3 + 1e-12
    return (w1 * s1 + w2 * s2 + w3 * s3) / w_sum


def get_signals(best_params, prices):
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


def run(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2):
    def fitness(candidate):
        high = _weighted_signal(
            prices,
            candidate[0], candidate[1], candidate[2],
            int(round(candidate[3])), int(round(candidate[4])), int(round(candidate[5])),
            candidate[6],
        )
        low = _weighted_signal(
            prices,
            candidate[7], candidate[8], candidate[9],
            int(round(candidate[10])), int(round(candidate[11])), int(round(candidate[12])),
            candidate[13],
        )
        cash, *_ = evaluate(prices, high, low)
        return -cash

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2)
    pso.run(fitness, bounds=BOUNDS)
    return pso
