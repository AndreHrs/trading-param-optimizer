"""
Drafted backtester.
Takes the parameters vector, build the signals and generate buy/sell.

Handle the simulation of $1000 sim, returns final cash (aka the fitness)

*Don't forget the broker fee
"""

import matplotlib.pyplot as plt
import numpy as np

from utilities.filters import wma


def evaluate(prices, short_signals, long_signals, fee = 0.03):
    """Do the backtesting.

    Args:
        prices (numpyArray): numpy array of the prices
        short_signals (numpyArray): WMA arrays across the prices for short signals
        long_signals (numpyArray): WMA arrays across the prices for long signals
        fee (float): the broker fee, defaults to 3%

    Returns:
        float: cash
    """
    diff = short_signals - long_signals
    # print(len(diff))
    # print("signs", np.sign(diff))
    Sign_kernel = np.array([0.5, -0.5])
    signals = (wma(np.sign(diff), 2, Sign_kernel))

    # print("signals", signals)
    # print("len signals", len(signals))
    # print("signals != 0", signals[signals != 0])
    cash = 1000.0 
    btc_held = 0.0
    fee = 0.03

    for t in range(len(prices)):
        if signals[t] == -1 and btc_held == 0:
            print(f"DO buy at time {t} with price {prices[t]}")
            spend = cash * (1 - fee)
            btc_held = spend / prices[t]
            cash = 0
            plt.plot(t, prices[t], 'go')
        if signals[t] == 1 and btc_held > 0:
            print(f"Do sell at time {t} with price {prices[t]}")
            gain = btc_held * prices[t]
            gainminusfee = gain * (1 - fee)
            cash = gainminusfee
            btc_held = 0
            plt.plot(t, prices[t], 'ro')
    # If at the end still holding bitcoin, sell them
    if btc_held > 0:
        print(f"Do sell at time {t} with price {prices[t]}")
        gain = btc_held * prices[t]
        gainminusfee = gain * (1 - fee)
        cash += gainminusfee
        plt.plot(t, prices[t], 'ro')
    print("Final cash", cash)


    plt.plot(long_signals)
    plt.plot(short_signals)
    plt.show()
    return cash
