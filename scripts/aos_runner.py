import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_loader import load_data
from runners.aos.common import re_evaluate
from runners.aos.plot import plot_equity_and_purchases
import runners.aos.sma as sma_runner
import runners.aos.lma as lma_runner
import runners.aos.ema as ema_runner
import runners.aos.weighted as weighted_runner

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

# --- SMA ---
sma_aos = sma_runner.run(train_prices)
print("SMA RUN:::")
print(f"best param   : {sma_aos.get_best_params()}")
print(f"best fitness : {sma_aos.best_fitness}")
print("=" * 64)

# --- LMA ---
lma_aos = lma_runner.run(train_prices)
print("LMA RUN:::")
print(f"best param   : {lma_aos.get_best_params()}")
print(f"best fitness : {lma_aos.best_fitness}")
print("=" * 64)

# --- EMA shared alpha ---
ema_shared_aos = ema_runner.run_shared(train_prices)
print("EMA SHARED ALPHA RUN:::")
print(f"best param   : {ema_shared_aos.get_best_params()}")
print(f"best fitness : {ema_shared_aos.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_shared_aos.get_best_params(), train_prices, ema_runner.get_signals_shared
)
plot_equity_and_purchases(ema_shared_aos, train_prices, short, long, buy_at, sell_at, "Train performance shared ema")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    ema_shared_aos.get_best_params(), test_prices, ema_runner.get_signals_shared
)
plot_equity_and_purchases(ema_shared_aos, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test performance shared ema", equity_curve=equity_curve_test)
print("SHARED BUY AT TEST:::", buy_at_test)
print("SHARED SELL AT TEST:::", sell_at_test)
print("SHARED CASH AT TEST END:::", cash_test)

# --- EMA independent alpha ---
ema_ind_aos = ema_runner.run_independent(train_prices)
print("EMA INDEPENDENT RUN:::")
print(f"best param   : {ema_ind_aos.get_best_params()}")
print(f"best fitness : {ema_ind_aos.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_ind_aos.get_best_params(), train_prices, ema_runner.get_signals_independent
)
plot_equity_and_purchases(ema_ind_aos, train_prices, short, long, buy_at, sell_at, "Train performance independent ema")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    ema_ind_aos.get_best_params(), test_prices, ema_runner.get_signals_independent
)
plot_equity_and_purchases(ema_ind_aos, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test performance independent ema", equity_curve=equity_curve_test)

# --- Weighted filters ---
weighted_aos = weighted_runner.run(train_prices)
print("Weighted Filter RUN:::")
print(f"best param   : {weighted_aos.get_best_params()}")
print(f"best fitness : {weighted_aos.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    weighted_aos.get_best_params(), train_prices, weighted_runner.get_signals
)
plot_equity_and_purchases(weighted_aos, train_prices,
    short, long, buy_at, sell_at, title="Weighted Train Set Performance")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    weighted_aos.get_best_params(), test_prices, weighted_runner.get_signals
)
plot_equity_and_purchases(weighted_aos, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Weighted Test Set Performance", equity_curve=equity_curve_test)
print("WEIGHTED CASH AT TEST END:::", cash_test)
