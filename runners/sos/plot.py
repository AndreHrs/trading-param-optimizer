import numpy as np
from matplotlib import pyplot as plt

def plot_equity_and_purchases(sos, prices, short_signal, long_signal, buy_at, sell_at, title="Train Set Performance", equity_curve=None):
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    # Equity Curve + Drawdown OR Best Fitness
    ax_equity = axs[0, 0]
    if equity_curve is not None:
        # Calculate Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        ax_equity.plot(equity_curve, label="Equity", color="green")
        ax_equity.set_title("Equity Curve & Drawdown")
        ax_equity.set_xlabel("Time")
        ax_equity.set_ylabel("Portfolio Value", color="green")
        ax_equity.tick_params(axis='y', labelcolor="green")
        
        ax_dd = ax_equity.twinx()
        ax_dd.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3, label="Drawdown")
        ax_dd.set_ylabel("Drawdown", color="red")
        ax_dd.tick_params(axis='y', labelcolor="red")
    else:
        ax_equity.plot(sos.history["LE_energy"], label="Best Fitness", color="blue")
        ax_equity.set_title("Optimization Convergence (SOS)")
        ax_equity.set_xlabel("Iteration")
        ax_equity.set_ylabel("Best Fitness (Negative Cash)")
        ax_equity.legend()

    # (Top Right): Trading Signals
    ax_signals = axs[0, 1]
    ax_signals.plot(prices, label="Prices", color="black", alpha=0.5)
    ax_signals.plot(short_signal, label="Short Signal", color="orange")
    ax_signals.plot(long_signal, label="Long Signal", color="blue")

    for point in buy_at:
        ax_signals.plot(point[0], point[1], "g^", markersize=8, label="Buy" if point == buy_at[0] else "")

    for point in sell_at:
        ax_signals.plot(point[0], point[1], "rv", markersize=8, label="Sell" if point == sell_at[0] else "")

    ax_signals.set_title("Trading Signals")
    ax_signals.set_xlabel("Time")
    ax_signals.set_ylabel("Price")
    ax_signals.legend()

    # Parameter Evolution
    ax_params = axs[1, 0]
    # LE_position is a list of arrays (the best parameters at each iteration)
    history_params = np.array(sos.history["LE_position"])
    
    if len(history_params) > 0:
        for i, param_name in enumerate(sos.param_names):
            # Normalize to 0-1 scale so they fit nicely on the same plot
            param_values = history_params[:, i]
            param_min = sos.param_ranges[i, 0]
            param_max = sos.param_ranges[i, 1]
            normalized_values = (param_values - param_min) / (param_max - param_min + 1e-9)
            ax_params.plot(normalized_values, label=f"{param_name} (norm)")
            
        ax_params.set_title("Parameter Evolution (Normalized)")
        ax_params.set_xlabel("Iteration")
        ax_params.set_ylabel("Normalized Value [0, 1]")
        ax_params.legend()
    else:
        ax_params.set_title("Parameter Evolution (No data)")

    # Population Diversit
    ax_div = axs[1, 1]
    if "population_avg_fitness" in sos.history and len(sos.history["population_avg_fitness"]) > 0:
        ax_div.plot(sos.history["population_avg_fitness"], label="Avg Population Fitness", color="orange", linestyle="--")
        ax_div.plot(sos.history["LE_energy"], label="Best Organism Fitness", color="blue")
        ax_div.set_title("Population Diversity & Convergence")
        ax_div.set_xlabel("Iteration")
        ax_div.set_ylabel("Fitness (Negative Cash)")
        ax_div.legend()
    else:
        ax_div.set_title("Population Diversity (No data)")

    plt.tight_layout()
    plt.show()
