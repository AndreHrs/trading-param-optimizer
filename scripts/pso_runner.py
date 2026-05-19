import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_loader import load_data
import runners.pso.ema as ema_runner

from runners.aos.common import re_evaluate
from runners.aos.plot import plot_equity_and_purchases

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

# --- EMA shared alpha ---
ema_shared_pso = ema_runner.run_shared(train_prices)
print("EMA SHARED ALPHA RUN:::")
print(f"best param   : {ema_shared_pso.get_best_params()}")
print(f"best fitness : {ema_shared_pso.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_shared_pso.get_best_params(), train_prices, ema_runner.get_signals_shared
)
plot_equity_and_purchases(ema_shared_pso, train_prices, short, long, buy_at, sell_at, "Train performance shared ema")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    ema_shared_pso.get_best_params(), test_prices, ema_runner.get_signals_shared
)
plot_equity_and_purchases(ema_shared_pso, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test performance shared ema", equity_curve=equity_curve_test)
print("SHARED BUY AT TEST:::", buy_at_test)
print("SHARED SELL AT TEST:::", sell_at_test)
print("SHARED CASH AT TEST END:::", cash_test)

# --- EMA independent alpha ---
ema_ind_pso = ema_runner.run_independent(train_prices)
print("EMA INDEPENDENT RUN:::")
print(f"best param   : {ema_ind_pso.get_best_params()}")
print(f"best fitness : {ema_ind_pso.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_ind_pso.get_best_params(), train_prices, ema_runner.get_signals_independent
)
plot_equity_and_purchases(ema_ind_pso, train_prices, short, long, buy_at, sell_at, "Train performance independent ema")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    ema_ind_pso.get_best_params(), test_prices, ema_runner.get_signals_independent
)
plot_equity_and_purchases(ema_ind_pso, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test performance independent ema", equity_curve=equity_curve_test)