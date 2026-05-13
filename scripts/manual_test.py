import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_loader import load_data

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

from optimizer.aos import AOS
from optimizer.evaluator import evaluate
from utilities.filters import lma_filter, ema_filter, wma, sma_filter

prices = train_prices

def sma_fitness(candidate):
    short_n, long_n = candidate
    long = (wma(prices, long_n, sma_filter(long_n)))
    short = (wma(prices, short_n, sma_filter(short_n)))
    cash, buy_at, sell_at, equity_curve = evaluate(train_prices, short, long)
    return cash

aos = AOS(10, 1, 5)
aos.run(sma_fitness, bounds = {
    "short_window": (2, 50),
    "long_window":  (51, 200)
})
print(aos.param_names)
print(aos.param_ranges[0])
print(aos.candidate_solutions)
print(aos.energy)