from utilities.data_loader import load_data
from runners.abo.common import re_evaluate
from runners.abo.plot import plot_equity_and_purchases
import runners.abo.sma as sma_runner
import runners.abo.lma as lma_runner
import runners.abo.ema as ema_runner
import runners.abo.weighted as weighted_runner

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

# --- SMA ---
sma_abo = sma_runner.run(train_prices)
print("SMA RUN:::")
print(f"best param   : {sma_abo.get_best_params()}")
print(f"best fitness : {sma_abo.best_fitness}")
print("=" * 64)

# --- LMA ---
lma_abo = lma_runner.run(train_prices)
print("LMA RUN:::")
print(f"best param   : {lma_abo.get_best_params()}")
print(f"best fitness : {lma_abo.best_fitness}")
print("=" * 64)

# --- EMA shared alpha ---
ema_shared_abo = ema_runner.run_shared(train_prices)
print("EMA SHARED ALPHA RUN:::")
print(f"best param   : {ema_shared_abo.get_best_params()}")
print(f"best fitness : {ema_shared_abo.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_shared_abo.get_best_params(), train_prices, ema_runner.get_signals_shared
)
plot_equity_and_purchases(ema_shared_abo, train_prices, short, long, buy_at, sell_at, "Train performance shared ema")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    ema_shared_abo.get_best_params(), test_prices, ema_runner.get_signals_shared
)
plot_equity_and_purchases(ema_shared_abo, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test performance shared ema", equity_curve=equity_curve_test)
print("SHARED BUY AT TEST:::", buy_at_test)
print("SHARED SELL AT TEST:::", sell_at_test)
print("SHARED CASH AT TEST END:::", cash_test)

# --- EMA independent alpha ---
ema_ind_abo = ema_runner.run_independent(train_prices)
print("EMA INDEPENDENT RUN:::")
print(f"best param   : {ema_ind_abo.get_best_params()}")
print(f"best fitness : {ema_ind_abo.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_ind_abo.get_best_params(), train_prices, ema_runner.get_signals_independent
)

# --- Weighted filters ---
weighted_abo = weighted_runner.run(train_prices)
print("Weighted Filter RUN:::")
print(f"best param   : {weighted_abo.get_best_params()}")
print(f"best fitness : {weighted_abo.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    weighted_abo.get_best_params(), train_prices, weighted_runner.get_signals
)
plot_equity_and_purchases(weighted_abo, train_prices,
    short, long, buy_at, sell_at)

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    weighted_abo.get_best_params(), test_prices, weighted_runner.get_signals
)
plot_equity_and_purchases(weighted_abo, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test Performance", equity_curve=equity_curve_test)
print("WEIGHTED CASH AT TEST END:::", cash_test)
