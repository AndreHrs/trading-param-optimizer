"""
Experiment runner: runs all 6 optimizers x 5 strategies x N_RUNS trials x (2 if both RUN_RANDOM and RUN_FIXED is true).
Note: Currently all algorithms must start with same Initial population size due to the code limitation, especially
on fixed populations
"""

import json
import os
import time
import importlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from utilities.data_loader import load_data
from utilities.csv_exporter import COLUMNS
from utilities.pin_p_cores import get_worker_count, worker_init

import mlflow

# ===============
# CONFIGURATIONS:
# ===============
RUN_RANDOM = True
RUN_FIXED = True

N_RUNS = 5
GLOBAL_INITIAL_POPULATION_SIZE = 100
RESULTS_DIR = "results"
RANDOM_CSV = os.path.join(RESULTS_DIR, "random_runs.csv")
FIXED_CSV = os.path.join(RESULTS_DIR, "fixed_runs.csv")
FIXED_POP_FILE = os.path.join(RESULTS_DIR, "fixed_populations.npz")

HYPERPARAMS = {
    "pso":  {"pop_size": GLOBAL_INITIAL_POPULATION_SIZE,
                "max_iter": 50, "w_max": 0.9, "w_min": 0.4, "c1": 2, "c2": 2, "max_vel_frac": 0.1},
    "abo":  {"pop_size": GLOBAL_INITIAL_POPULATION_SIZE, 
                "max_iter": 50, "lp1": 0.5,   "lp2": 0.5,   "stagnation_limit": 10},
    "mrfo": {"pop_size": GLOBAL_INITIAL_POPULATION_SIZE, 
                "max_iter": 50, "somersault_range": 2.0},
    "sos":  {"pop_size": GLOBAL_INITIAL_POPULATION_SIZE, 
                "max_iter": 50},
    "aos":  {"pop_size": GLOBAL_INITIAL_POPULATION_SIZE, 
                "max_iter": 50, "n_shells": 5},
    "gps":  {"initial_step_size": 1, "tolerance": 1e-5, "decay_rate": 0.5, "max_iterations": 50},
}


ALGO_LIST = ["pso", "abo", "mrfo", "sos", "aos", "gps"]
STRATEGY_LIST = ["sma", "lma", "ema_shared", "ema_independent", "weighted"]

STRATEGY_DIMS = {
    "sma":             2,
    "lma":             2,
    "ema_shared":      3,
    "ema_independent": 4,
    "weighted":        14,
}

# Hyperparams keys passed directly as kwargs to each algo run function.
PASS_THROUGH_KWARGS = {
    "pso": ["pop_size", "max_iter", "w_max", "w_min", "c1", "c2", "max_vel_frac"],
    "abo": ["pop_size", "max_iter", "lp1", "lp2"],
    "mrfo": ["pop_size", "max_iter"],
    "sos": ["pop_size", "max_iter"],
    "aos": ["pop_size", "max_iter", "n_shells"],
    "gps": ["initial_step_size", "tolerance", "decay_rate", "max_iterations"],
}

# =======
# HELPERS
# =======

def _get_runner(algo, strategy):
    """Return (mod, run_fn, get_signals_fn, re_evaluate_fn) for an algo+strategy pair."""
    mod_base = strategy.split("_")[0]          # sma | lma | ema | weighted

    # we can use importlib to programmatically import the file since all the file share the same name
    mod      = importlib.import_module(f"runners.{algo}.{mod_base}")
    common   = importlib.import_module(f"runners.{algo}.common")
    suffix   = "" if "_" not in strategy else f"_{strategy.split('_', 1)[1]}"
    return mod, getattr(mod, f"run{suffix}"), getattr(mod, f"get_signals{suffix}"), common.re_evaluate

def _get_strategy_bounds(strategy):
    mod = importlib.import_module(f"runners.pso.{strategy.split('_')[0]}")
    if strategy == "ema_shared":
        return mod.BOUNDS_SHARED
    if strategy == "ema_independent":
        return mod.BOUNDS_INDEPENDENT
    return mod.BOUNDS

def make_gps_directions(n_dims):
    eye = np.eye(n_dims)
    return list(eye) + list(-eye)

def run_single(algo, strategy, prices, hp_override=None, initial_population=None, initial_position=None, seed=None):
    _, run_fn, _, _ = _get_runner(algo, strategy)
    hp     = {**HYPERPARAMS[algo], **(hp_override or {})}
    kwargs = {k: hp[k] for k in PASS_THROUGH_KWARGS[algo] if k in hp}
    if seed is not None:
        kwargs["seed"] = seed

    if algo == "mrfo" and "somersault_range" in hp:
        kwargs["somersault"] = hp["somersault_range"]

    if algo == "gps":
        kwargs["D"] = make_gps_directions(STRATEGY_DIMS[strategy])
        if initial_position is not None:
            kwargs["initial_position"] = initial_position
    elif initial_population is not None:
        kwargs["initial_population"] = initial_population

    return run_fn(prices, **kwargs)


def _get_initialization(start_mode, algo, strategy, run_id):
    """Return (initial_population, initial_position) matching the batch runner."""
    if start_mode == "random":
        return None, None
    if start_mode == "fixed":
        if not os.path.exists(FIXED_POP_FILE):
            generate_fixed_populations()
        pop = np.load(FIXED_POP_FILE)[f"{strategy}_{run_id}"]
        if algo == "gps":
            return None, pop[0]
        return pop, None
    raise ValueError(f"start_mode must be 'fixed' or 'random', got {start_mode!r}")


def reproduce_result(algo, strategy, run_id, start_mode, data_path="./data/BTC-Daily.csv"):
    """
    Re-run one experiment with the same initialization as the batch runner.

    Parameters
    ----------
    algo : str
        Optimizer name (e.g. "aos", "pso", "mrfo").
    strategy : str
        Signal strategy (e.g. "ema_shared", "sma", "weighted").
    run_id : int
        Run index (0 .. N_RUNS-1). Used as RNG seed for random starts.
    start_mode : str
        "fixed" or "random" — must match the CSV row you want to reproduce.
    data_path : str
        Path to the price CSV (default matches the batch runner).

    Returns
    -------
    dict with keys:
        optimizer, history, train_prices, test_prices,
        short, long, buy_at, sell_at, equity_curve, final_cash,
        short_test, long_test, buy_at_test, sell_at_test,
        test_equity_curve, test_final_cash,
        best_params, best_fitness, algo, strategy, run_id, start_mode
    """
    if algo not in ALGO_LIST:
        raise ValueError(f"Unknown algo {algo!r}. Choose from {ALGO_LIST}")
    if strategy not in STRATEGY_LIST:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {STRATEGY_LIST}")

    train_p, _, test_p, _ = load_data(data_path)
    initial_population, initial_position = _get_initialization(start_mode, algo, strategy, run_id)

    opt = run_single(
        algo, strategy, train_p,
        initial_population=initial_population,
        initial_position=initial_position,
        seed=run_id,
    )

    _, _, sig_fn, reeval_fn = _get_runner(algo, strategy)
    best_params = opt.get_best_params()
    short, long, buy_at, sell_at, equity_curve, final_cash = reeval_fn(best_params, train_p, sig_fn)
    short_test, long_test, buy_at_test, sell_at_test, test_equity_curve, test_final_cash = reeval_fn(
        best_params, test_p, sig_fn
    )

    return {
        "optimizer": opt,
        "history": opt.history,
        "train_prices": train_p,
        "test_prices": test_p,
        "short": short,
        "long": long,
        "buy_at": buy_at,
        "sell_at": sell_at,
        "equity_curve": equity_curve,
        "final_cash": final_cash,
        "short_test": short_test,
        "long_test": long_test,
        "buy_at_test": buy_at_test,
        "sell_at_test": sell_at_test,
        "test_equity_curve": test_equity_curve,
        "test_final_cash": test_final_cash,
        "best_params": best_params,
        "best_fitness": opt.best_fitness,
        "algo": algo,
        "strategy": strategy,
        "run_id": run_id,
        "start_mode": start_mode,
    }


def plot_reproduce_result(result, split="both"):
    """
    Plot train and/or test results from reproduce_result().

    Parameters
    ----------
    result : dict
        Output of reproduce_result().
    split : str
        "train", "test", or "both" (default).
    """
    import importlib

    plot_fn = importlib.import_module(f"runners.{result['algo']}.plot").plot_equity_and_purchases
    label = f"{result['algo']} {result['strategy']} run {result['run_id']} ({result['start_mode']})"

    if split in ("train", "both"):
        plot_fn(
            result["optimizer"],
            result["train_prices"],
            result["short"],
            result["long"],
            result["buy_at"],
            result["sell_at"],
            title=f"{label} — train",
        )

    if split in ("test", "both"):
        plot_fn(
            result["optimizer"],
            result["test_prices"],
            result["short_test"],
            result["long_test"],
            result["buy_at_test"],
            result["sell_at_test"],
            title=f"{label} — test",
            equity_curve=result["test_equity_curve"],
        )


def collect_row(opt, algo, strategy, run_id, start_mode, train_prices, test_prices, hyperparams_used):
    _, _, sig_fn, reeval_fn = _get_runner(algo, strategy)
    best_params = opt.get_best_params()
    _, _, _, _, equity_curve, final_cash = reeval_fn(best_params, train_prices, sig_fn)
    _, _, _, _, test_equity_curve, test_final_cash = reeval_fn(best_params, test_prices,  sig_fn)
    return {
        "algo": algo,
        "strategy": strategy,
        "run_id": run_id,
        "start_mode": start_mode,
        "final_cash": final_cash,
        "best_fitness": opt.best_fitness,
        "epoch_count": opt.epoch_count,
        "runtime_ms": opt.time,
        "equity_curve": equity_curve,
        "test_final_cash": test_final_cash,
        "test_equity_curve": test_equity_curve,
        "best_params": best_params,
        "hyperparams": hyperparams_used,
        "early_stopped": getattr(opt, "early_stop", False),
    }


# ==================
# PARALLEL HELPERS
# ==================
# train_prices / test_prices are set in _main() before parallel runs; forked workers inherit them.
train_prices = None
test_prices = None


def _run_task(args):
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("/trading-optimizer")

    algo, strategy, run_id, mode, initial_population, initial_position = args
    opt = run_single(algo, strategy, train_prices,
                     initial_population=initial_population,
                     initial_position=initial_position,
                     seed=run_id)
    row = collect_row(opt, algo, strategy, run_id, mode, train_prices, test_prices, HYPERPARAMS[algo])

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_params({
            "algo": algo,
            "strategy": strategy,
            "run_id": run_id,
            "start_mode": mode,
            **HYPERPARAMS[algo]   # pop_size, max_iter, etc.
        })
        mlflow.log_metrics({
            "final_cash": row["final_cash"],
            "test_final_cash": row["test_final_cash"],
            "best_fitness": row["best_fitness"],
            "runtime_ms": row["runtime_ms"],
        })
    return row

def _run_parallel(tasks):
    n_workers = get_worker_count()
    total = len(tasks)
    print(f"  Workers: {n_workers}  |  Tasks: {total}")
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init) as ex:
        futures = {ex.submit(_run_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            row = fut.result()
            rows.append(row)
            print(f"[{i}/{total}] {t[3]:6s} | {t[0]:4s} | {t[1]:16s} | run {t[2]}", flush=True)
    return rows

def _save_results(rows, csv_path):
    df = pd.DataFrame(rows)
    for col in ("equity_curve", "test_equity_curve", "best_params", "hyperparams"):
        df[col] = df[col].apply(json.dumps)
    df[COLUMNS].to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")


# ================================
# RUNNER (RANDOM INITIALIZATION)
# ================================
def run_random():
    tasks = [
        (algo, strategy, run_id, "random", None, None)
        for algo in ALGO_LIST
        for strategy in STRATEGY_LIST
        for run_id in range(N_RUNS)
    ]
    _save_results(_run_parallel(tasks), RANDOM_CSV)


# ================================
# RUNNER (FIXED POPULATIONS)
# ================================
def generate_fixed_populations():
    populations = {}
    for strategy in STRATEGY_LIST:
        bounds = _get_strategy_bounds(strategy)
        lows   = np.array([v[0] for v in bounds.values()])
        highs  = np.array([v[1] for v in bounds.values()])
        n_dims = STRATEGY_DIMS[strategy]
        for run_id in range(N_RUNS):
            seed = run_id * 100 + abs(hash(strategy)) % 100
            rng  = np.random.default_rng(seed=seed)
            pop  = lows + rng.random((GLOBAL_INITIAL_POPULATION_SIZE, n_dims)) * (highs - lows)
            populations[f"{strategy}_{run_id}"] = pop
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(FIXED_POP_FILE, **populations)
    print(f"Fixed populations saved to {FIXED_POP_FILE}")
    return populations

def run_fixed():
    if not os.path.exists(FIXED_POP_FILE):
        print("Generating fixed populations...")
        generate_fixed_populations()
    else:
        npz_check = np.load(FIXED_POP_FILE)
        first_key = next(iter(npz_check))
        if npz_check[first_key].shape[0] != GLOBAL_INITIAL_POPULATION_SIZE:
            print("Saved numpy initial points did not match, regenerating new one")
            generate_fixed_populations()

    npz   = np.load(FIXED_POP_FILE)
    tasks = []
    for algo in ALGO_LIST:
        for strategy in STRATEGY_LIST:
            for run_id in range(N_RUNS):
                pop = npz[f"{strategy}_{run_id}"]
                if algo == "gps":
                    tasks.append((algo, strategy, run_id, "fixed", None, pop[0]))
                else:
                    tasks.append((algo, strategy, run_id, "fixed", pop, None))
    _save_results(_run_parallel(tasks), FIXED_CSV)


def _main():
    global train_prices, test_prices
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_prices, _, test_prices, _ = load_data("./data/BTC-Daily.csv")

    if RUN_RANDOM:
        print(f"=== Random runs ({len(ALGO_LIST)} algos x {len(STRATEGY_LIST)} strategies x {N_RUNS} runs) ===")
        _t_random = time.perf_counter()
        run_random()
        _elapsed_random = time.perf_counter() - _t_random
        print(f"Done. Results in {RANDOM_CSV}")
        print(f"Random runs time: {_elapsed_random / 60:.1f} min ({_elapsed_random:.1f} s)")

    if RUN_FIXED:
        print(f"\n=== Fixed-population runs ===")
        _t_fixed = time.perf_counter()
        run_fixed()
        _elapsed_fixed = time.perf_counter() - _t_fixed
        print(f"Done. Results in {FIXED_CSV}")
        print(f"Fixed runs time: {_elapsed_fixed / 60:.1f} min ({_elapsed_fixed:.1f} s)")


if __name__ == "__main__":
    _main()