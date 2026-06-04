"""
Demonstrates the full MLflow Model Registry flow:
  1. Load a Production model by name
  2. Run predict() on the test price series
  3. Print final_cash so you can compare against the source run

Usage:
    python load_and_test.py                         # defaults to pso / ema_shared
    python load_and_test.py --algo abo --strategy sma
"""

import argparse
import mlflow.pyfunc
import pandas as pd

from utilities.data_loader import load_data

TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(TRACKING_URI)


def load_and_predict(algo, strategy, split="test"):
    model_name = f"{algo}-{strategy}"
    model_uri = f"models:/{model_name}/Production"

    print(f"Loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    _, _, test_prices, _ = load_data("./data/BTC-Daily.csv")
    train_prices, _, _, _ = load_data("./data/BTC-Daily.csv")

    prices = test_prices if split == "test" else train_prices
    prices_df = pd.DataFrame({"price": prices})

    print(f"Running predict() on {split} prices ({len(prices)} timesteps)...")
    result = model.predict(prices_df)

    print(f"\nResult for {model_name} ({split}):")
    print(f"  final_cash : {result['final_cash'].iloc[0]:.2f}")
    print(f"  buy  trades: {len(result['buy_at'].iloc[0])}")
    print(f"  sell trades: {len(result['sell_at'].iloc[0])}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",     default="pso")
    parser.add_argument("--strategy", default="ema_shared")
    parser.add_argument("--split",    default="test", choices=["train", "test"])
    args = parser.parse_args()

    load_and_predict(args.algo, args.strategy, args.split)
