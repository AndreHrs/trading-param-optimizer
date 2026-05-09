from utilities.data_loader import load_data

# --- MRFO utilities ---
from runners.mrfo.common import re_evaluate
from runners.mrfo.plot import plot_equity_and_purchases
from runners.mrfo.plot import plot_optimization_history

# --- MRFO strategy runner ---
import runners.mrfo.sma as sma_runner
import runners.mrfo.lma as lma_runner
import runners.mrfo.ema as ema_runner
import runners.mrfo.weighted as weighted_runner

# --- helper func ---
def print_result(name, result):
    """
    Standardized optimization result printer.
    """
    print(f"{name}")
    print(f"best param   : {result.get_best_params()}")
    print(f"best fitness : {result.best_fitness}")
    print("=" * 64)

def evaluate_and_plot(
    optimizer_result,
    prices,
    signal_fn,
    title,
    is_test=False
):
    """
    Re-evaluate optimized parameters and generate plots.
    """
    short, long, buy_at, sell_at, equity_curve, cash = re_evaluate(
        optimizer_result.get_best_params(),
        prices,
        signal_fn
    )

    plot_equity_and_purchases(
        optimizer_result,
        prices,
        short,
        long,
        buy_at,
        sell_at,
        title=title,
        equity_curve=equity_curve if is_test else None
    )

    return {
        "short": short,
        "long": long,
        "buy_at": buy_at,
        "sell_at": sell_at,
        "equity_curve": equity_curve,
        "cash": cash
    }

# --- load data ---
train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

print("\n--- RUNNING MRFO TRADING BOT OPTIMIZATION ---")
print("=" * 64)

# --- SMA ---
sma_res = sma_runner.run(train_prices)

print_result(
    "SMA MRFO RUN:::",
    sma_res
)

# --- LMA ---
lma_res = lma_runner.run(train_prices)

print_result(
    "LMA MRFO RUN:::",
    lma_res
)

# --- EMA shared alpha ---
ema_shared_res = ema_runner.run_shared(train_prices)

print_result(
    "EMA SHARED ALPHA MRFO RUN:::",
    ema_shared_res
)

# --- train performance ---
evaluate_and_plot(
    optimizer_result=ema_shared_res,
    prices=train_prices,
    signal_fn=ema_runner.get_signals_shared,
    title="MRFO Shared EMA - Train Performance"
)


# --- test performance ---
shared_test = evaluate_and_plot(
    optimizer_result=ema_shared_res,
    prices=test_prices,
    signal_fn=ema_runner.get_signals_shared,
    title="MRFO Shared EMA - Test Performance",
    is_test=True
)

print("SHARED BUY AT TEST:::", shared_test["buy_at"])
print("SHARED SELL AT TEST:::", shared_test["sell_at"])
print(f"SHARED CASH AT TEST END::: ${shared_test['cash']:.2f}")
print("=" * 64)

# --- EMA independent alpha ---

ema_ind_res = ema_runner.run_independent(train_prices)

print_result(
    "EMA INDEPENDENT ALPHA MRFO RUN:::",
    ema_ind_res
)

# --- train performance ---
evaluate_and_plot(
    optimizer_result=ema_ind_res,
    prices=train_prices,
    signal_fn=ema_runner.get_signals_independent,
    title="MRFO Independent EMA - Train Performance"
)

# --- test performance ---
ind_test = evaluate_and_plot(
    optimizer_result=ema_ind_res,
    prices=test_prices,
    signal_fn=ema_runner.get_signals_independent,
    title="MRFO Independent EMA - Test Performance",
    is_test=True
)

print("INDEPENDENT BUY AT TEST:::", ind_test["buy_at"])
print("INDEPENDENT SELL AT TEST:::", ind_test["sell_at"])
print(f"INDEPENDENT CASH AT TEST END::: ${ind_test['cash']:.2f}")
print("=" * 64)

# --- Weighted filters ---
weighted_res = weighted_runner.run(train_prices)

print_result(
    "WEIGHTED FILTER MRFO RUN:::",
    weighted_res
)

# --- train performance ---
evaluate_and_plot(
    optimizer_result=weighted_res,
    prices=train_prices,
    signal_fn=weighted_runner.get_signals,
    title="MRFO Weighted Filter - Train Performance"
)

# --- test performance ---
weighted_test = evaluate_and_plot(
    optimizer_result=weighted_res,
    prices=test_prices,
    signal_fn=weighted_runner.get_signals,
    title="MRFO Weighted Filter - Test Performance",
    is_test=True
)

# --- convergence plots (MRFO-safe only) ---

if hasattr(sma_res, "history"):
    plot_optimization_history(sma_res, "SMA Convergence")

if hasattr(lma_res, "history"):
    plot_optimization_history(lma_res, "LMA Convergence")

if hasattr(ema_shared_res, "history"):
    plot_optimization_history(ema_shared_res, "EMA Shared Convergence")

if hasattr(ema_ind_res, "history"):
    plot_optimization_history(ema_ind_res, "EMA Independent Convergence")

if hasattr(weighted_res, "history"):
    plot_optimization_history(weighted_res, "Weighted Convergence")

print("WEIGHTED BUY AT TEST:::", weighted_test["buy_at"])
print("WEIGHTED SELL AT TEST:::", weighted_test["sell_at"])
print(f"WEIGHTED CASH AT TEST END::: ${weighted_test['cash']:.2f}")
print("=" * 64)

print("\nMRFO optimization completed.")