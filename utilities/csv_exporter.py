import csv
import json
import os

COLUMNS = [
    "algo", "strategy", "run_id", "start_mode",
    "final_cash", "best_fitness", "epoch_count", "runtime_ms",
    "equity_curve", "test_final_cash", "test_equity_curve",
    "best_params", "hyperparams", "early_stopped",
]


def append_result(filepath, row_dict):
    """Append one result row to a CSV file. Creates file with header if new."""
    file_exists = os.path.isfile(filepath)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        row = dict(row_dict)
        row["equity_curve"]      = json.dumps(row["equity_curve"])
        row["test_equity_curve"] = json.dumps(row["test_equity_curve"])
        row["best_params"]       = json.dumps(row["best_params"])
        row["hyperparams"]       = json.dumps(row["hyperparams"])
        writer.writerow(row)
