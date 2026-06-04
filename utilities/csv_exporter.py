import csv
import json
import os

COLUMNS = [
    "algo", "strategy", "run_id", "start_mode", "best_fitness",
    "final_cash", "test_final_cash", "epoch_count", "runtime_ms", "early_stopped",
]