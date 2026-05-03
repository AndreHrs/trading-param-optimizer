from matplotlib import pyplot as plt


def plot_equity_and_purchases(aos, prices, short_signal, long_signal, buy_at, sell_at, title="Train Set Performance", equity_curve=None):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title)

    if equity_curve is not None:
        ax_left.plot(equity_curve, label="equity", color="green")
        ax_left.set_title("Equity Curve")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("Portfolio Value")
    else:
        ax_left.plot(aos.history["LE_energy"], label="LE energy", color="blue")
        ax_left.set_title("LE Curve")
        ax_left.set_xlabel("Iteration")
        ax_left.set_ylabel("Energy (negated cash)")
    ax_left.legend()

    ax_right.plot(prices, label="prices", color="black", alpha=0.5)
    ax_right.plot(short_signal, label="short signal", color="orange")
    ax_right.plot(long_signal, label="long signal", color="blue")

    for point in buy_at:
        ax_right.plot(point[0], point[1], "gx", markersize=10)

    for point in sell_at:
        ax_right.plot(point[0], point[1], "rx", markersize=10)

    ax_right.set_title("Trading Signals")
    ax_right.set_xlabel("Time")
    ax_right.set_ylabel("Price")
    ax_right.legend()

    plt.tight_layout()
    plt.show()
