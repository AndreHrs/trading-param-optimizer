"""Shared plotting helpers for all optimizer runners."""

import numpy as np
from matplotlib import pyplot as plt
from utilities.filters import wma


def trade_signal(short_signal, long_signal):
    """Replicate evaluate() crossover signal for visualization."""
    diff = short_signal - long_signal
    sign_kernel = np.array([0.5, -0.5])
    return wma(np.sign(diff), 2, sign_kernel)


def energy_history(optimizer):
    h = optimizer.history
    if "gbest_energy" in h:
        return h["gbest_energy"], "gbest energy", "Global Best Energy Curve"
    if "LE_energy" in h:
        return h["LE_energy"], "best fitness", "Optimization Convergence"
    if "best_fitness" in h:
        return h["best_fitness"], "best fitness", "Optimization Progress"
    return [], "fitness", "Optimization Progress"


def plot_signals_panels(ax_price, ax_diff, prices, short_signal, long_signal, buy_at, sell_at):
    times = np.arange(len(prices))

    ax_price.plot(times, prices, label="prices", color="black", linewidth=1.2)
    ax_price.plot(times, short_signal, label="short signal", color="darkorange", linestyle="--", linewidth=1.5)
    ax_price.plot(times, long_signal, label="long signal", color="steelblue", linestyle="-.", linewidth=1.5)

    for i, (t, p) in enumerate(buy_at):
        ax_price.plot(t, p, "gx", markersize=10, label="buy" if i == 0 else "_nolegend_")
    for i, (t, p) in enumerate(sell_at):
        ax_price.plot(t, p, "rx", markersize=10, label="sell" if i == 0 else "_nolegend_")

    ax_price.set_title("Price & filter signals")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")

    diff = short_signal - long_signal
    trade = trade_signal(short_signal, long_signal)
    ax_diff.plot(times, diff, label="short − long", color="purple", linewidth=1.2)
    ax_diff.axhline(0, color="gray", linewidth=0.8)
    ax_diff.plot(times, trade, label="trade signal", color="green", linewidth=1.0, alpha=0.85)
    ax_diff.set_title("Signal spread & trade signal")
    ax_diff.set_xlabel("Time")
    ax_diff.set_ylabel("Spread / signal")
    ax_diff.legend(loc="upper left")


def plot_equity_and_purchases(
    optimizer,
    prices,
    short_signal,
    long_signal,
    buy_at,
    sell_at,
    title="Performance",
    equity_curve=None,
):
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1])
    ax_left = fig.add_subplot(gs[:, 0])
    ax_price = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[1, 1])
    fig.suptitle(title)

    if equity_curve is not None:
        ax_left.plot(equity_curve, label="equity", color="green")
        ax_left.set_title("Equity Curve")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("Portfolio Value")
    else:
        values, label, subtitle = energy_history(optimizer)
        ax_left.plot(values, label=label, color="blue")
        ax_left.set_title(subtitle)
        ax_left.set_xlabel("Iteration")
        ax_left.set_ylabel("Energy (negated cash)")
    ax_left.legend()

    plot_signals_panels(ax_price, ax_diff, prices, short_signal, long_signal, buy_at, sell_at)

    plt.tight_layout()
    plt.show()
