#!/bin/bash

set -e

ENV_NAME="cits4404"
MARKER=".installed"
FORCE=false

for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
    esac
done

echo "============================================"
echo "  CITS4404 Team 6 — Installer"
echo "============================================"

if [ -f "$MARKER" ] && [ "$FORCE" = false ]; then
    echo "[Install] Already installed. Run with --force to reinstall."
    exit 0
fi

# --- Python version check ---
echo "[Install] Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "[Error] Python 3.10+ is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "[Install] Python $PYTHON_VERSION OK"

# --- Environment setup ---
USE_CONDA=false

if command -v conda &>/dev/null; then
    USE_CONDA=true
    echo "[Install] conda found — setting up environment '$ENV_NAME'"

    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
        echo "[Install] Conda environment '$ENV_NAME' already exists — reusing"
    else
        echo "[Install] Creating conda environment '$ENV_NAME' with Python 3.12..."
        conda create -n "$ENV_NAME" python=3.12 -y
    fi

    conda activate "$ENV_NAME"
    echo "[Install] Conda environment '$ENV_NAME' activated"
else
    echo "[Install] conda not found — falling back to venv (.venv)"

    if [ ! -d ".venv" ]; then
        echo "[Install] Creating .venv..."
        python3 -m venv .venv
    fi

    source .venv/bin/activate
    echo "[Install] venv activated"
fi

# --- Dependencies ---
echo "[Install] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

# --- Write marker ---
{
    echo "env_type=$([ "$USE_CONDA" = true ] && echo conda || echo venv)"
    echo "env_name=$ENV_NAME"
} > "$MARKER"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Environment : $([ "$USE_CONDA" = true ] && echo "conda ($ENV_NAME)" || echo "venv (.venv)")"
echo "  Run experiments : python experiment_runner.py"
echo "============================================"
