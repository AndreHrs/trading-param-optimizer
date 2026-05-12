from optimizer.evaluator import evaluate


def re_evaluate(best_params, prices, get_signals_fn):
    short, long = get_signals_fn(best_params, prices)
    cash, buy_at, sell_at, equity_curve = evaluate(prices, short, long)
    return short, long, buy_at, sell_at, equity_curve, cash
