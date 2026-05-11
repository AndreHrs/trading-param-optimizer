from optimizer.evaluator import evaluate


def re_evaluate(best_params, prices, get_signals_fn):
    """Rebuild signals from best_params and run evaluate to retrieve the full trade history.

    Args:
        best_params (dict): named parameters from sos.get_best_params()
        prices (np.ndarray): price series to build signals on and evaluate against
        get_signals_fn (Callable(best_params, prices) -> tuple(short, long)]): strategy-specific function=

    Returns:
        tuple: (short, long, buy_at, sell_at, equity_curve, cash)
    """
    short, long = get_signals_fn(best_params, prices)
    cash, buy_at, sell_at, equity_curve = evaluate(prices, short, long)
    return short, long, buy_at, sell_at, equity_curve, cash
