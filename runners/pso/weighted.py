from optimizer.pso import PSO
from optimizer.evaluator import evaluate
from runners.shared import BOUNDS_WEIGHTED as BOUNDS, get_signals_weighted as get_signals, _weighted_signal


def run(prices, pop_size=100, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2, max_vel_frac=0.1, initial_population=None):
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

    pso = PSO(pop_size, max_iter, w_max, w_min, c1, c2, max_vel_frac)
    pso.run(fitness, bounds=BOUNDS, initial_population=initial_population)
    return pso
