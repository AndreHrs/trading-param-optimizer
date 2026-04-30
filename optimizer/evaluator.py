"""
Drafted backtester.
Takes the parameters vector, build the signals and generate buy/sell.

Handle the simulation of $1000 sim, returns final cash (aka the fitness)

*Don't forget the broker fee
"""

import matplotlib.pyplot as plt
import numpy as np

from utilities.filters import wma

def execute_buy(cash, price, fee) -> tuple[int, int]:
    """Execute buy action.

    Args:
        cash (float): cash on hand
        price (float): current price of the bitcoin closing price
        fee (float): the broker fee

    Returns:
        tuple(int, int): for [cash, btc_held]
    """
    spend = cash * (1 - fee)
    btc_held = spend / price
    return 0.0, btc_held

def execute_sell(btc_held, price, fee) -> tuple[int, int]:
    """Execute sell action.

    Args:
        btc_held (float): btc held
        price (float): current price of the bitcoin closing price
        fee (float): the broker fee

    Returns:
        tuple(int, int): for [cash, btc_held]
    """
    gain = btc_held * price
    gainminusfee = gain * (1 - fee)
    cash = gainminusfee
    return cash, 0.0

def evaluate(prices, short_signals, long_signals, fee = 0.03):
    """Do the backtesting.

    Args:
        prices (numpyArray): numpy array of the prices
        short_signals (numpyArray): WMA arrays across the prices for short signals
        long_signals (numpyArray): WMA arrays across the prices for long signals
        fee (float): the broker fee, defaults to 3%

    Returns:
        tuple(float, list[tuple[int, float], list[tuple[int, float], list[float]]:
            tuples for:
            - final cash on hand
            - coordinate where buy happens
            - coordinate where sell happens
            - portfolio value curve
    """
    diff = short_signals - long_signals

    sign_kernel = np.array([0.5, -0.5])
    signals = (wma(np.sign(diff), 2, sign_kernel))

    equity_curve = []
    buy_at = []
    sell_at = []
    cash = 1000.0 
    btc_held = 0.0
    fee = 0.03
    for t in range(len(prices)):
        if signals[t] == -1 and btc_held == 0:
            cash, btc_held = execute_buy(cash, prices[t], fee)
            buy_at.append((t, prices[t]))
        if signals[t] == 1 and btc_held > 0:
            cash, btc_held = execute_sell(btc_held, prices[t], fee)
            sell_at.append((t, prices[t]))
        equity_curve.append(cash + btc_held * prices[t])
    # If at the end still holding bitcoin, sell it all
    if btc_held > 0:
        cash, btc_held = execute_sell(btc_held, prices[t], fee)
        sell_at.append((t, prices[t]))
        equity_curve[-1] = cash

    return cash, buy_at, sell_at, equity_curve
