from matplotlib import pyplot as plt


def plot_equity_and_purchases(
    optimizer,
    prices,
    short_signal,
    long_signal,
    buy_at,
    sell_at,
    title="Performance",
    equity_curve=None
):

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title)

    # --- LEFT - optimization / equity ---
    if equity_curve is not None:

        ax_left.plot(equity_curve, label="equity", color="green")
        ax_left.set_title("Equity Curve")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("Portfolio Value")

    else:
        history = optimizer.history

        if "best_fitness" in history:
            values = history["best_fitness"]
        elif "LE_energy" in history:
            values = history["LE_energy"]
        elif "gbest_energy" in history:
            values = history["gbest_energy"]
        else:
            values = []

        ax_left.plot(values, label="fitness", color="blue")
        ax_left.set_title("Optimization Progress")
        ax_left.set_xlabel("Iteration")
        ax_left.set_ylabel("Fitness (-cash, lower is better)")

    ax_left.legend()

    # --- RIGHT - trading signals ---
    ax_right.plot(prices, label="prices", color="black", alpha=0.5)
    ax_right.plot(short_signal, label="short signal", color="orange")
    ax_right.plot(long_signal, label="long signal", color="blue")

    for idx, (t, p) in enumerate(buy_at):
        ax_right.plot(
            t,
            p,
            "gx",
            markersize=10,
            label="buy" if idx == 0 else ""
        )

    for idx, (t, p) in enumerate(sell_at):
        ax_right.plot(
            t,
            p,
            "rx",
            markersize=10,
            label="sell" if idx == 0 else ""
        )

    ax_right.set_title("Trading Signals")
    ax_right.set_xlabel("Time")
    ax_right.set_ylabel("Price")
    ax_right.legend()

    plt.tight_layout()
    plt.show()

# --- for convergence check ---
def plot_optimization_history(optimizer, title="MRFO Convergence"):

    history = getattr(optimizer, "history", None)

    if not history or "best_fitness" not in history:
        print("No best_fitness found in MRFO history")
        return

    values = history["best_fitness"]

    if values is None or len(values) == 0:
        print("Empty optimization history")
        return

    plt.figure(figsize=(10, 5))

    plt.plot(values, color="blue", linewidth=2, label="Best Fitness")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (-cash)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()