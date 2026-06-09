"""
MLflow pyfunc wrapper for a trading strategy model.

The "model" is a set of optimizer-found best_params + a named strategy.
predict() accepts a single-column DataFrame of prices and returns a DataFrame
with the backtest results.
"""

import json
import mlflow.pyfunc
import numpy as np
import pandas as pd

from runners.shared import (
    get_signals_sma,
    get_signals_lma,
    get_signals_ema_shared,
    get_signals_ema_independent,
    get_signals_weighted,
    re_evaluate,
)

SIGNAL_FNS = {
    "sma":              get_signals_sma,
    "lma":              get_signals_lma,
    "ema_shared":       get_signals_ema_shared,
    "ema_independent":  get_signals_ema_independent,
    "weighted":         get_signals_weighted,
}


class TradingStrategyModel(mlflow.pyfunc.PythonModel):
    """
    Wraps a strategy + best_params as an MLflow pyfunc model.

    Artifacts expected at log time:
        "best_params" -> path to a JSON file containing the best_params dict

    Model input (predict):
        pandas DataFrame with a single column "price" (float, chronological order)

    Model output:
        pandas DataFrame with columns:
            final_cash, equity_curve, buy_at, sell_at, short, long
    """

    def load_context(self, context):
        with open(context.artifacts["best_params"]) as f:
            artifact = json.load(f)
        # artifact format: {"strategy": "...", "best_params": {...}}
        self.strategy = artifact["strategy"]
        self.best_params = artifact["best_params"]

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        prices = np.array(model_input["price"], dtype=float)

        sig_fn = SIGNAL_FNS[self.strategy]
        short, long, buy_at, sell_at, equity_curve, final_cash = re_evaluate(
            self.best_params, prices, sig_fn
        )

        return pd.DataFrame([{
            "final_cash":   final_cash,
            "equity_curve": equity_curve,
            "buy_at":       buy_at,
            "sell_at":      sell_at,
            "short":        short.tolist(),
            "long":         long.tolist(),
        }])


mlflow.models.set_model(TradingStrategyModel())
