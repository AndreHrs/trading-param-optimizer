from matplotlib import pyplot as plt


def plot_equity_and_purchases(abo, prices, short_signal, long_signal, buy_at, sell_at, title="Train Set Performance", equity_curve=None):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title)

    if equity_curve is not None:
        ax_left.plot(equity_curve, label="equity", color="green")
        ax_left.set_title("Equity Curve")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("Portfolio Value")
    else:
        if "gbest_energy" in abo.history.keys():
            energy_history = abo.history["gbest_energy"]
        else:
            energy_history = abo.history["LE_energy"]

        ax_left.plot(energy_history, label="gbest energy", color="blue")
        ax_left.set_title("Global Best Energy Curve")
        ax_left.set_xlabel("Iteration")
        ax_left.set_ylabel("Energy (negated cash)")
    ax_left.legend()

    ax_right.plot(prices, label="prices", color="black", alpha=0.5)
    ax_right.plot(short_signal, label="short signal", color="orange")
    ax_right.plot(long_signal, label="long signal", color="blue")

    for i, point in enumerate(buy_at):
        ax_right.plot(point[0], point[1], "gx", markersize=10, label="buy" if i == 0 else "_nolegend_")

    for i, point in enumerate(sell_at):
        ax_right.plot(point[0], point[1], "rx", markersize=10, label="sell" if i == 0 else "_nolegend_")

    ax_right.set_title("Trading Signals")
    ax_right.set_xlabel("Time")
    ax_right.set_ylabel("Price")
    ax_right.legend()

    plt.tight_layout()
    plt.show()
