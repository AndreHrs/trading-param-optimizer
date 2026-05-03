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

# TODO:: Example of Weighted filters run

