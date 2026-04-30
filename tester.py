from utilities.data_loader import load_data

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

from utilities.filters import lma_filter, ema_filter, wma, sma_filter

import numpy as np
P = train_prices[:300]

Long_N = 20
Short_N = 5
long = (wma(P, Long_N, sma_filter(Long_N)))
short = (wma(P, Short_N, sma_filter(Short_N)))
diff = short - long
# print(len(diff))
# print("signs", np.sign(diff))
Sign_kernel = np.array([0.5, -0.5])
signals = (wma(np.sign(diff), 2, Sign_kernel))

# Replace longest N with 0 to invalidate trade window
longest = max(Long_N, Short_N)
signals[:longest] = 0

# print("signals", signals)
# print("len signals", len(signals))
# print("signals != 0", signals[signals != 0])
cash = 1000.0 
btc_held = 0.0
fee = 0.03

for t in range(len(P)):
    if signals[t] == 1 and btc_held == 0:
        print(f"DO buy at time {t} with price {P[t]}")
        spend = cash * (1 - fee)
        btc_held = spend / P[t]
        cash = 0

    if signals[t] == -1 and btc_held > 0:
        print(f"Do sell at time {t} with price {P[t]}")
        gain = btc_held * P[t]
        gainminusfee = gain * (1 - fee)
        cash = gainminusfee
        btc_held = 0

# If at the end still holding bitcoin, sell them
if btc_held > 0:
    print(f"Holding btc at final epoch. Do sell at time {t} with price {P[t]}")
    gain = btc_held * P[t]
    gainminusfee = gain * (1 - fee)
    cash += gainminusfee

print("Final cash", cash)

