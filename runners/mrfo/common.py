from optimizer.evaluator import evaluate

def re_evaluate(best_params, prices, get_signals_fn):
    """
    Rebuild signals from best_params and run evaluate to retrieve full trade history.
    """

    short, long = get_signals_fn(best_params, prices)

    cash, buy_at, sell_at, equity_curve = evaluate(prices, short, long)

    return short, long, buy_at, sell_at, equity_curve, cash