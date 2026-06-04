"""
Queries the MLflow experiment for the best run per (algo, strategy) combination
(by test_final_cash), then registers each as a pyfunc model in the Model Registry
and promotes it to Production.

Usage:
    python register_best_models.py

Requires the MLflow tracking server to be running at http://127.0.0.1:5000/
"""

import json
import os
import tempfile

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from experiment_runner import ALGO_LIST, STRATEGY_LIST
from trading_model import TradingStrategyModel

TRACKING_URI = "http://127.0.0.1:5000/"
EXPERIMENT_NAME = "/trading-optimizer"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def get_experiment_id(name):
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise RuntimeError(f"Experiment {name!r} not found. Run the experiment first.")
    return exp.experiment_id


def find_best_run(experiment_id, algo, strategy):
    """Return the MLflow Run with the highest test_final_cash for this (algo, strategy)."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"params.algo = '{algo}' AND params.strategy = '{strategy}'",
        order_by=["metrics.test_final_cash DESC"],
    )

    # Exclude registration runs (have source_run_id set and no real artifacts)
    original_runs = [r for r in runs if "source_run_id" not in r.data.params]
    if not original_runs:
        return None
    return original_runs[0]


def register_model(run, algo, strategy, experiment_id):
    model_name = f"{algo}-{strategy}"

    # Load best_params from the source run's artifact store
    best_params = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/best_params.json")

    # Write a combined artifact file: strategy name + best_params
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "best_params.json")
        with open(artifact_path, "w") as f:
            json.dump({"strategy": strategy, "best_params": best_params}, f)

        # Log a new run that contains the pyfunc model
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"register-{model_name}") as reg_run:
            mlflow.log_params({"algo": algo, "strategy": strategy, "source_run_id": run.info.run_id})
            mlflow.log_metric("test_final_cash", run.data.metrics["test_final_cash"])

            mlflow.pyfunc.log_model(
                name=f"model-{model_name}",
                python_model=TradingStrategyModel(),
                artifacts={"best_params": artifact_path},
                code_paths=["runners/", "optimizer/", "utilities/"],
            )

            model_uri = f"runs:/{reg_run.info.run_id}/model-{model_name}"

    # Register in the Model Registry
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"  Registered {model_name} version {mv.version}")

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"  Promoted {model_name} v{mv.version} -> Production")
    return mv


def main():
    experiment_id = get_experiment_id(EXPERIMENT_NAME)
    print(f"Experiment ID: {experiment_id}\n")

    for algo in ALGO_LIST:
        for strategy in STRATEGY_LIST:
            print(f"{algo} / {strategy}")
            run = find_best_run(experiment_id, algo, strategy)
            if run is None:
                print(f"  No runs found, skipping.")
                continue
            print(f"  Best run: {run.info.run_id}  test_final_cash={run.data.metrics['test_final_cash']:.2f}")
            register_model(run, algo, strategy, experiment_id)


if __name__ == "__main__":
    main()
