# Import the Data
from utilities.data_loader import load_data

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

from matplotlib import pyplot as plt
def plot_equity_and_purchases(aos, prices, short_signal, long_signal, buy_at, sell_at):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))

    # LE energy curve
    ax_left.plot(aos.history["LE_energy"], label="LE energy", color="blue")
    ax_left.set_title("Convergence Curve")
    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("Energy (negated cash)")
    ax_left.legend()

    # price signals and buy/sell points
    ax_right.plot(prices, label="prices", color="black", alpha=0.5)
    ax_right.plot(short_signal, label="short signal", color="orange")
    ax_right.plot(long_signal, label="long signal", color="blue")

    for point in buy_at:
        ax_right.plot(point[0], point[1], 'gx', markersize=8)

    for point in sell_at:
        ax_right.plot(point[0], point[1], 'rx', markersize=8)

    ax_right.set_title("Trading Signals")
    ax_right.set_xlabel("Time")
    ax_right.set_ylabel("Price")
    ax_right.legend()

    plt.tight_layout()
    plt.show()

from optimizer.aos import AOS
from optimizer.evaluator import evaluate
from utilities.filters import lma_filter, ema_filter, wma, sma_filter

prices = train_prices

# Example for running with SMA fitness
def sma_fitness(candidate):
    short_n = int(round(candidate[0]))
    long_n  = int(round(candidate[1]))
    long = (wma(prices, long_n, sma_filter(long_n)))
    short = (wma(prices, short_n, sma_filter(short_n)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return -cash #flip the sign for fitness since AOS is minimizing function

# Example run with SMA fitness
sma_aos = AOS(100, 50, 5)
sma_aos.run(sma_fitness, bounds = {
    "short_window": (2, 50),
    "long_window":  (51, 200)
})

# print(aos.history["LE_energy"])
# print(aos.history["LE_position"])
print("SMA RUN:::")
print(f"best param   : {sma_aos.get_best_params()}")
print(f"best fitness : {sma_aos.best_fitness}")
print("="*64)

# Example for running with LMA fitness
def lma_fitness(candidate):
    short_n = int(round(candidate[0]))
    long_n  = int(round(candidate[1]))
    long = (wma(prices, long_n, lma_filter(long_n)))
    short = (wma(prices, short_n, lma_filter(short_n)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return -cash #flip the sign for fitness since AOS is minimizing function

# Example run with SMA fitness
lma_aos = AOS(100, 50, 5)
lma_aos.run(lma_fitness, bounds = {
    "short_window": (2, 50),
    "long_window":  (51, 200)
})

# print(aos.history["LE_energy"])
# print(aos.history["LE_position"])
print("LMA RUN:::")
print(f"best param   : {lma_aos.get_best_params()}")
print(f"best fitness : {lma_aos.best_fitness}")
print("="*64)

# Example for EMA run with shared alpha
def ema_fitness_shared(candidate):
    short_n = int(round(candidate[0]))
    long_n  = int(round(candidate[1]))
    alpha = candidate[2]
    long = (wma(prices, long_n, ema_filter(long_n, alpha)))
    short = (wma(prices, short_n, ema_filter(short_n, alpha)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return -cash #flip the sign for fitness since AOS is minimizing function

# Example run with SMA fitness
ema_aos = AOS(100, 50, 5)
ema_aos.run(ema_fitness_shared, bounds = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha":  (1e-6, 1),
})

# print(aos.history["LE_energy"])
# print(aos.history["LE_position"])
print("EMA SHARED ALPHA RUN:::")
print(f"best param   : {ema_aos.get_best_params()}")
print(f"best fitness : {ema_aos.best_fitness}")
print("="*64)

# Example of EMA run with independent alpha
def ema_fitness_independent(candidate):
    short_n = int(round(candidate[0]))
    long_n  = int(round(candidate[1]))
    alpha_short = candidate[2]
    alpha_long  = candidate[3]
    long = (wma(prices, long_n, ema_filter(long_n, alpha_long)))
    short = (wma(prices, short_n, ema_filter(short_n, alpha_short)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return -cash #flip the sign for fitness since AOS is minimizing function

# Example run with SMA fitness
ema_aos_ind = AOS(100, 50, 5)
ema_aos_ind.run(ema_fitness_independent, bounds = {
    "short_window": (2, 50),
    "long_window":  (51, 200),
    "alpha_short":  (1e-6, 1),
    "alpha_long":  (1e-6, 1),
})

# print(aos.history["LE_energy"])
# print(aos.history["LE_position"])
print("EMA INDEPENDENT RUN:::")
print(f"best param   : {ema_aos_ind.get_best_params()}")
print(f"best fitness : {ema_aos_ind.best_fitness}")
print("="*64)


def get_ema_signals(best_params, prices):
    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))
    alpha = best_params["alpha"]
    long = (wma(prices, long_n, ema_filter(long_n, alpha)))
    short = (wma(prices, short_n, ema_filter(short_n, alpha)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return short, long, buy_at, sell_at, equity_curve

best_params = ema_aos.get_best_params()
short_signal, long_signal, buy_at, sell_at, equity_curve = get_ema_signals(best_params, prices)
plot_equity_and_purchases(ema_aos, prices, short_signal, long_signal, buy_at, sell_at)

def get_ema_signals_independent(best_params, prices):
    short_n = int(round(best_params["short_window"]))
    long_n  = int(round(best_params["long_window"]))
    alpha_short = best_params["alpha_short"]
    alpha_long = best_params["alpha_long"]
    long = (wma(prices, long_n, ema_filter(long_n, alpha_long)))
    short = (wma(prices, short_n, ema_filter(short_n, alpha_short)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return short, long, buy_at, sell_at, equity_curve

best_params = ema_aos_ind.get_best_params()
short_signal, long_signal, buy_at, sell_at, equity_curve = get_ema_signals_independent(best_params, prices)
plot_equity_and_purchases(ema_aos_ind, prices, short_signal, long_signal, buy_at, sell_at)

bounds_weighted = {
    "high_w1":      (0.0, 1.0),
    "high_w2":      (0.0, 1.0),
    "high_w3":      (0.0, 1.0),
    "high_d1":      (2, 50),    # SMA window
    "high_d2":      (2, 50),    # LMA window
    "high_d3":      (2, 50),    # EMA window
    "high_alpha3":  (0.01, 0.99),

    "low_w1":      (0.0, 1.0),
    "low_w2":      (0.0, 1.0),
    "low_w3":      (0.0, 1.0),
    "low_d1":      (2, 50),    # SMA window
    "low_d2":      (2, 50),    # LMA window
    "low_d3":      (2, 50),    # EMA window
    "low_alpha3":  (0.01, 0.99),
}

def weighted_filters_fitness(candidate):
    high_w1, high_w2, high_w3 = candidate[0], candidate[1], candidate[2]
    high_d1, high_d2, high_d3 = int(round(candidate[3])), int(round(candidate[4])), int(round(candidate[5]))
    high_alpha3 = candidate[6]

    low_w1, low_w2, low_w3 = candidate[7], candidate[8], candidate[9]
    low_d1, low_d2, low_d3 = int(round(candidate[10])), int(round(candidate[11])), int(round(candidate[12]))
    low_alpha3 = candidate[13]


    # HIGH signal Eq. 7 brief
    sma_high = wma(prices, high_d1, sma_filter(high_d1))
    lma_high = wma(prices, high_d2, lma_filter(high_d2))
    ema_high = wma(prices, high_d3, ema_filter(high_d3, high_alpha3))

    weight_sum_high = high_w1 + high_w2 + high_w3
    HIGH = (high_w1 * sma_high + high_w2 * lma_high + high_w3 * ema_high) / weight_sum_high

    # LOW signal same structure, different windows
    sma_low = wma(prices, low_d1, sma_filter(low_d1))
    lma_low = wma(prices, low_d2, lma_filter(low_d2))
    ema_low = wma(prices, low_d3, ema_filter(low_d3, low_alpha3))
    
    weight_sum_low = low_w1 + low_w2 + low_w3
    LOW = (low_w1 * sma_low + low_w2 * lma_low + low_w3 * ema_low) / weight_sum_low

    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, HIGH, LOW)
    return -cash

# Example run with weighted fitness
lma_weighted = AOS(100, 50, 5)
lma_weighted.run(weighted_filters_fitness, bounds = bounds_weighted)

# print(aos.history["LE_energy"])
# print(aos.history["LE_position"])
print("Weighted Filter RUN:::")
print(f"best param   : {lma_weighted.get_best_params()}")
print(f"best fitness : {lma_weighted.best_fitness}")
print("="*64)