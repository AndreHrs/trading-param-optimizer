"""
Experiment runner: runs all 6 optimizers x 5 strategies x N_RUNS trials x (2 if both RUN_RANDOM and RUN_FIXED is true).
Note: Currently all algorithms must start with same Initial population size due to the code limitation, especially
on fixed populations
"""

import os
import importlib
import numpy as np

from utilities.data_loader import load_data
from utilities.csv_exporter import append_result

# ===============
# CONFIGURATIONS:
# ===============
RUN_RANDOM = True
RUN_FIXED = True

N_RUNS = 30
GLOBAL_INITIAL_POPULATION_SIZE = 100
RESULTS_DIR = "results"
RANDOM_CSV = os.path.join(RESULTS_DIR, "random_runs.csv")
FIXED_CSV = os.path.join(RESULTS_DIR, "fixed_runs.csv")
FIXED_POP_FILE = os.path.join(RESULTS_DIR, "fixed_populations.npz")

HYPERPARAMS = {
    "pso":  {"pop_size": GLOBAL_INITIAL_POPULATION_SIZE, 
                "max_iter": 50, "w_max": 0.9, "w_min": 0.4, "c1": 2, "c2": 2},
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
    "pso": ["pop_size", "max_iter", "w_max", "w_min", "c1", "c2"],
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

def run_single(algo, strategy, prices, hp_override=None, initial_population=None, initial_position=None):
    _, run_fn, _, _ = _get_runner(algo, strategy)
    hp     = {**HYPERPARAMS[algo], **(hp_override or {})}
    kwargs = {k: hp[k] for k in PASS_THROUGH_KWARGS[algo] if k in hp}

    if algo == "mrfo" and "somersault_range" in hp:
        kwargs["somersault"] = hp["somersault_range"]

    if algo == "gps":
        kwargs["D"] = make_gps_directions(STRATEGY_DIMS[strategy])
        if initial_position is not None:
            kwargs["initial_position"] = initial_position
    elif initial_population is not None:
        kwargs["initial_population"] = initial_population

    return run_fn(prices, **kwargs)


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

# ================================
# RUNNER (RANDOM INITIALIZATION)
# ================================
def run_random(train_prices, test_prices):
    total = len(ALGO_LIST) * len(STRATEGY_LIST) * N_RUNS
    done  = 0
    for algo in ALGO_LIST:
        for strategy in STRATEGY_LIST:
            for run_id in range(N_RUNS):
                done += 1
                print(f"[{done}/{total}] random | {algo:4s} | {strategy:16s} | run {run_id}", flush=True)
                opt = run_single(algo, strategy, train_prices)
                row = collect_row(opt, algo, strategy, run_id, "random", train_prices, test_prices, HYPERPARAMS[algo])
                append_result(RANDOM_CSV, row)


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

def run_fixed(train_prices, test_prices):
    if not os.path.exists(FIXED_POP_FILE):
        print("Generating fixed populations...")
        generate_fixed_populations()

    npz   = np.load(FIXED_POP_FILE)
    total = len(ALGO_LIST) * len(STRATEGY_LIST) * N_RUNS
    done  = 0

    for algo in ALGO_LIST:
        for strategy in STRATEGY_LIST:
            for run_id in range(N_RUNS):
                done += 1
                pop = npz[f"{strategy}_{run_id}"]
                print(f"[{done}/{total}] fixed  | {algo:4s} | {strategy:16s} | run {run_id}", flush=True)

                if algo == "gps":
                    opt = run_single(algo, strategy, train_prices, initial_position=pop[0])
                else:
                    opt = run_single(algo, strategy, train_prices,
                                     initial_population=pop)

                row = collect_row(opt, algo, strategy, run_id, "fixed", train_prices, test_prices, HYPERPARAMS[algo])
                append_result(FIXED_CSV, row)


os.makedirs(RESULTS_DIR, exist_ok=True)
train_prices, _, test_prices, _ = load_data("./data/BTC-Daily.csv")

if RUN_RANDOM:
    print(f"=== Random runs ({len(ALGO_LIST)} algos x {len(STRATEGY_LIST)} strategies x {N_RUNS} runs) ===")
    run_random(train_prices, test_prices)
    print(f"Done. Results in {RANDOM_CSV}")

if RUN_FIXED:
    print(f"\n=== Fixed-population runs ===")
    run_fixed(train_prices, test_prices)
    print(f"Done. Results in {FIXED_CSV}")
