# Trading Parameter Optimizer

> **Disclaimer:** Originally developed as part of [CITS4404 at UWA](https://github.com/AndreHrs/CITS4404_2026_Team6). This repository is a manual duplicate of that work as GitHub does not allow forking your own repository. This version is extended with MLflow experiment tracking, model version control, and a FastAPI serving layer.

The project optimises parameters for Bitcoin trading strategies using nature-inspired algorithms, with experiment tracking via MLflow.

---

## What it does

Six optimisers are benchmarked against five moving-average crossover strategies on historical BTC/USD data:

| Optimisers |
|---|
| Particle Swarm Optimisation (PSO) |
| Atomic Orbital Search (AOS) |
| Manta Ray Foraging Optimisation (MRFO) |
| Symbiotic Organisms Search (SOS) |
| African Buffalo Optimisation (ABO) |
| Generalised Pattern Search (GPS) — baseline |


| Strategy | Parameters | Count |
|---|---|---|
| Double SMA crossover | `short_window`, `long_window` | 2 |
| Double LMA crossover | `short_window`, `long_window` | 2 |
| Double EMA crossover (shared α) | `short_window`, `long_window`, `alpha` | 3 |
| Double EMA crossover (independent α) | `short_window`, `long_window`, `alpha_short`, `alpha_long` | 4 |
| Weighted MA combination | `sma_weight_short`, `sma_weight_long`, `sma_short_window`, `sma_long_window`, `lma_weight_short`, `lma_weight_long`, `lma_short_window`, `lma_long_window`, `ema_weight_short`, `ema_weight_long`, `ema_short_window`, `ema_long_window`, `ema_alpha_short`, `ema_alpha_long` | 14 |

Performance is measured by final USD value after simulating trades with a 3% fee on a $1000 starting balance.

---

## Directory Structure

```
.
├── experiment_runner.py      Runs all algorithm/strategy combinations
├── register_best_models.py   Finds the best run per (algo, strategy) and registers it in the MLflow Model Registry
├── trading_model.py          MLflow pyfunc wrapper for loading and running a registered strategy
├── load_and_test.py          Loads a registered model from the registry and runs it on new data
├── notebook.ipynb            Post-processing: loads results, generates tables and figures
├── api.py                    FastAPI serving layer — loads Production models and exposes prediction endpoints
├── serve.sh                  Starts MLflow + FastAPI servers together
├── run.sh                    Convenience script: installs deps then runs the experiment runner
├── install.sh                Sets up conda or venv from requirements.txt
├── requirements.txt          Python dependencies
│
├── optimizer/                Algorithm implementations and shared evaluator
├── runners/                  Per-algorithm runner modules
├── utilities/                Data loading, CSV export, CPU-core pinning, filters
├── scripts/                  Standalone per-algorithm scripts and manual test utilities
│
├── data/                     Historical BTC/USD price data (daily and hourly CSVs)
├── results/                  Output CSVs from the experiment runner
├── mlartifacts/              MLflow artifact store
├── mlflow.db                 MLflow tracking database (SQLite)
└── report/                   LaTeX source and compiled PDF from the original project
```

---

## Setup

Python 3.12+ required.

**With conda:**

```bash
conda create -n trading python=3.12 -y
conda activate trading
pip install -r requirements.txt
```

**With venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running experiments

### 1. Start the MLflow server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

The tracking UI is then available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 2. Run all combinations

```bash
python experiment_runner.py
```

Configuration flags (number of runs, population size, which strategies to include) are set at the top of `experiment_runner.py`. Each run is logged to MLflow automatically.

### 3. Register the best models

```bash
python register_best_models.py
```

This queries MLflow for the highest-scoring run per (algo, strategy) pair and registers it in the Model Registry under a name like `pso-sma`, promoted to Production.

### 4. Load and run a registered model

```bash
python load_and_test.py
```

Loads the Production version of a registered model and runs it on test data.

### 5. Analyse results

Open `notebook.ipynb` and run all cells. It reads the `results/` CSVs and produces the summary tables and figures.

---

## Serving the API

After experiments are run and models are registered (steps 1–3 above), you can serve predictions via HTTP.

### Quick start

```bash
./serve.sh
```

This handles environment activation, starts the MLflow tracking server in the background, waits for it to be ready, then launches the FastAPI server. Both are stopped cleanly on Ctrl+C.

| Service | URL |
|---|---|
| FastAPI | http://127.0.0.1:8000 |
| Interactive API docs | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

### Endpoints

**`GET /models`** — list all loaded (algo, strategy) model keys.

**`POST /predict/{algo}/{strategy}`** — run a backtest from a JSON price list.

**`POST /predict-csv/{algo}/{strategy}`** — run a backtest from a CSV file upload.

Valid values:

| Parameter | Values |
|---|---|
| `{algo}` | `pso`, `abo`, `mrfo`, `sos`, `aos`, `gps` |
| `{strategy}` | `sma`, `lma`, `ema_shared`, `ema_independent`, `weighted` |

### curl examples

List loaded models:
```bash
curl http://localhost:8000/models
```

Predict from a CSV file (`close` column, default):
```bash
curl -X POST "http://localhost:8000/predict-csv/pso/sma" \
  -F "file=@data/BTC-Daily.csv" \
  -F "price_column=close"
```

Predict from a JSON price list:
```bash
curl -X POST "http://localhost:8000/predict/pso/sma" \
  -H "Content-Type: application/json" \
  -d '{"prices": [43185.48, 43178.98, 37712.68, 39146.66]}'
```

Example response:
```json
{
  "final_cash": 1243.57,
  "buy_count": 12,
  "sell_count": 11,
  "equity_curve": [1000.0, 1012.3, ...]
}
```

The interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs) let you explore and try all endpoints directly in the browser.

---

## Automated experiment run (Linux)

```bash
./run.sh
```

Handles environment setup and launches the experiment runner. Tested on Linux (Arch); untested on macOS or Windows/WSL.
