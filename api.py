"""
FastAPI serving layer for registered trading strategy models.

Models are loaded from the MLflow Model Registry at startup (once) and reused
across all requests. The MLflow tracking server only needs to be reachable at
startup time, not during inference.

Usage:
    uvicorn api:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from io import StringIO
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from experiment_runner import ALGO_LIST, STRATEGY_LIST

TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(TRACKING_URI)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    prices: list[float]


class PredictResponse(BaseModel):
    final_cash: float
    buy_count: int
    sell_count: int
    equity_curve: list[float]


# ---------------------------------------------------------------------------
# Load all Production models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    failed = []

    for algo in ALGO_LIST:
        for strategy in STRATEGY_LIST:
            key = f"{algo}-{strategy}"
            uri = f"models:/{key}/Production"
            try:
                app.state.models[key] = mlflow.pyfunc.load_model(uri)
                print(f"  Loaded {key}")
            except Exception as e:
                failed.append(key)
                print(f"  Skipped {key}: {e}")

    if failed:
        print(f"\nWarning: {len(failed)} model(s) not loaded: {failed}")
    else:
        print(f"\nAll {len(app.state.models)} models loaded.")

    yield  # app runs here

    app.state.models.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Trading Strategy Optimizer API",
    description="Serves best-found strategy params from the MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
def list_models() -> dict[str, Any]:
    """List all successfully loaded (algo, strategy) model keys."""
    return {"loaded_models": sorted(app.state.models.keys())}

@app.post("/predict/{algo}/{strategy}", response_model=PredictResponse)
def predict(algo: str, strategy: str, body: PredictRequest) -> PredictResponse:
    """Run backtest using a JSON price list."""
    return _run_predict(algo, strategy, body.prices)


@app.post("/predict-csv/{algo}/{strategy}", response_model=PredictResponse)
async def predict_csv(
    algo: str,
    strategy: str,
    file: UploadFile = File(...),
    price_column: str = Form("close"),
) -> PredictResponse:
    """
    Same as /predict but accepts a CSV file upload instead of a JSON price list.
    The price_column form field specifies which column to use (default: "close").
    """
    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode()))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    if price_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Column '{price_column}' not found. Available: {list(df.columns)}",
        )

    return _run_predict(algo, strategy, df[price_column].tolist())


def _run_predict(algo: str, strategy: str, prices: list[float]) -> PredictResponse:
    """
    Run the registered Production model for (algo, strategy) against the
    provided price series and return the backtest result.
    """
    key = f"{algo}-{strategy}"
    model = app.state.models.get(key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"No Production model found for '{key}'.")

    prices_df = pd.DataFrame({"price": prices})
    result = model.predict(prices_df)

    row = result.iloc[0]
    return PredictResponse(
        final_cash=float(row["final_cash"]),
        buy_count=len(row["buy_at"]),
        sell_count=len(row["sell_at"]),
        equity_curve=[float(v) for v in row["equity_curve"]],
    )
