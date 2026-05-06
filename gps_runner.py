from utilities.data_loader import load_data
import runners.gps.ema as ema_runner

from runners.aos.common import re_evaluate
from runners.aos.plot import plot_equity_and_purchases

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

# --- EMA independent alpha ---
ema_ind_gps = ema_runner.run_independent(train_prices, initial_step_size = 1,
    tolerance = 1e-5, decay_rate = 0.5, max_iterations=50, D = [
        [25, 0, 0, 0], [-25, 0, 0, 0], 
        [0, 100, 0, 0], [0, -100, 0, 0], 
        [0, 0, 0.5, 0], [0, 0, -0.5, 0], 
        [0, 0, 0, 0.5], [0, 0, 0, -0.5]])
print("EMA INDEPENDENT RUN:::")
print(f"best param   : {ema_ind_gps.get_best_params()}")
print(f"best fitness : {ema_ind_gps.best_fitness}")
print("=" * 64)

short, long, buy_at, sell_at, _, _ = re_evaluate(
    ema_ind_gps.get_best_params(), train_prices, ema_runner.get_signals_independent
)
plot_equity_and_purchases(ema_ind_gps, train_prices, short, long, buy_at, sell_at, "Train performance independent ema")

short_test, long_test, buy_at_test, sell_at_test, equity_curve_test, cash_test = re_evaluate(
    ema_ind_gps.get_best_params(), test_prices, ema_runner.get_signals_independent
)
plot_equity_and_purchases(ema_ind_gps, test_prices,
    short_test, long_test, buy_at_test, sell_at_test,
    title="Test performance independent ema", equity_curve=equity_curve_test)