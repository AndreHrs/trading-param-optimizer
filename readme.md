# Trading Parameter Optimizer

A personal continuation of [CITS4404_2026_Team6](https://github.com/AndreHrs/CITS4404_2026_Team6). GitHub doesn't allow forking your own repository, so this is a manual copy with ongoing development layered on top.

The project optimises parameters for Bitcoin trading strategies using nature-inspired algorithms, with experiment tracking via MLflow.

---

## What it does

Six optimisers are benchmarked against five moving-average crossover strategies on historical BTC/USD data:

| Optimisers | Strategies |
|---|---|
| Particle Swarm Optimisation (PSO) | Double SMA crossover |
| Atomic Orbital Search (AOS) | EMA/SMA crossover |
| Manta Ray Foraging Optimisation (MRFO) | Triple MA crossover |
| Symbiotic Organisms Search (SOS) | MACD-based |
| African Buffalo Optimisation (ABO) | Weighted MA combination |
| Generalised Pattern Search (GPS) — baseline | |

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

## Automated run (Linux)

```bash
./run.sh
```

Handles environment setup and launches the experiment runner. Tested on Linux (Arch); untested on macOS or Windows/WSL.
