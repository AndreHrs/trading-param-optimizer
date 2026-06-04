from matplotlib import pyplot as plt

from runners.plot_utils import plot_equity_and_purchases


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
