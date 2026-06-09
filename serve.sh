#!/bin/bash

set -e

MARKER=".installed"
MLFLOW_PORT=5000
API_PORT=8000

echo "============================================"
echo "  Trading Optimizer — API + MLflow Server"
echo "============================================"

if [ ! -f "$MARKER" ]; then
    echo "[Serve] Not installed yet — running installer first..."
    echo ""
    ./install.sh
    echo ""
fi

ENV_TYPE=$(grep "^env_type=" "$MARKER" | cut -d= -f2)
ENV_NAME=$(grep "^env_name=" "$MARKER" | cut -d= -f2)

echo "[Serve] Environment : $ENV_TYPE ($ENV_NAME)"

if [ "$ENV_TYPE" = "conda" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    source .venv/bin/activate
fi

cleanup() {
    echo ""
    echo "[Serve] Shutting down..."
    kill "$MLFLOW_PID" 2>/dev/null || true
    kill "$API_PID" 2>/dev/null || true
    wait "$MLFLOW_PID" "$API_PID" 2>/dev/null || true
    echo "[Serve] Done."
}
trap cleanup EXIT INT TERM

echo "[Serve] Starting MLflow server on http://127.0.0.1:$MLFLOW_PORT ..."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port "$MLFLOW_PORT" &
MLFLOW_PID=$!

echo "[Serve] Waiting for MLflow to be ready..."
for i in $(seq 1 20); do
    if curl -sf "http://127.0.0.1:$MLFLOW_PORT/health" > /dev/null 2>&1; then
        echo "[Serve] MLflow is up."
        break
    fi
    sleep 1
done

echo "[Serve] Starting FastAPI on http://127.0.0.1:$API_PORT ..."
uvicorn api:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

echo ""
echo "  MLflow UI  : http://127.0.0.1:$MLFLOW_PORT"
echo "  API        : http://127.0.0.1:$API_PORT"
echo "  API docs   : http://127.0.0.1:$API_PORT/docs"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

wait "$API_PID"
