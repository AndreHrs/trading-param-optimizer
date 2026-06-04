# CITS4404 Team 6 — Bitcoin Trading Bot Optimisation

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Reproducing Results](#reproducing-results)
  - [1. Automated setup and run (recommended)](#1-automated-setup-and-run-recommended)
  - [2. Manual setup (if not using `run.sh`)](#2-manual-setup-if-not-using-runsh)
  - [3. Run the experiment runner](#3-run-the-experiment-runner)
  - [4. Generate figures and tables](#4-generate-figures-and-tables)
- [AI Usage Disclaimer](#ai-usage-disclaimer)
- [Development Setup](#development-setup)

---

## Project Overview

This is a CITS4404 (Artificial Intelligence and Adaptive Systems) university project that builds and evaluates Bitcoin trading bots using nature-inspired optimisation algorithms. Six optimisers are compared: Atomic Orbital Search (AOS), Particle Swarm Optimisation (PSO), Manta Ray Foraging Optimisation (MRFO), Symbiotic Organisms Search (SOS), and African Buffalo Optimisation (ABO), with Generalised Pattern Search (GPS) serving as a classical local-search baseline. Each algorithm optimises the parameters of one of five moving-average crossover strategies — ranging from a simple 2-dimensional double-SMA crossover to a 14-dimensional weighted MA combination — evaluated on historical BTC/USD price data. Performance is measured by final USD holdings after simulating trades with a 3% transaction fee on a fixed starting balance of $1000.

---

## Directory Structure

```
.
├── experiment_runner.py   Main entry point — runs all algorithm/strategy combinations
├── notebook.ipynb         Post-processing notebook: loads CSVs, generates tables and figures
├── notebook.py            Plain-Python equivalent of the notebook originally for jupytext use
├── run.sh                 Convenience script: installs dependencies then runs the experiment runner
├── install.sh             Installer: sets up conda or venv environment from requirements.txt
├── requirements.txt       Python package dependencies
├── manual_calculation.ods Spreadsheet used for manual verification of results
│
├── optimizer/             Algorithm implementations (AOS, PSO, MRFO, SOS, ABO, GPS, shared evaluator)
├── runners/               Per-algorithm runner modules wiring optimisers to strategies
├── utilities/             Shared utilities: data loading, CSV export, CPU-core pinning, filters
├── scripts/               Standalone per-algorithm runner scripts and manual test utilities
│
├── data/                  Historical BTC/USD price data (daily and hourly CSVs)
├── results/               Output CSVs and summary tables produced by the experiment runner
├── report/                LaTeX source, bibliography, section files, figures, and compiled PDF
├── notes/                 Developer notes
└── not_used/              Archived code not included in the final submission
```

---

## Reproducing Results

### 1. Automated setup and run (recommended)

> **Note:** `run.sh` and `install.sh` have only been tested on Linux (Arch). Usage on macOS or Windows (using WSL) is untested, proceed with caution.

A convenience script handles environment setup and launches the runner in one step:

```bash
./run.sh
```

`run.sh` calls `install.sh` on first use (creating a `conda` or `.venv` environment and installing `requirements.txt`), then activates the environment and runs `experiment_runner.py`. Subsequent calls skip the install step unless `.installed` is missing.

You can pass arguments through to the runner:

```bash
./run.sh --some-flag value
```

### 2. Manual setup (if not using `run.sh`)

Python 3.10 or later is required.

**With conda:**

```bash
conda create -n cits4404 python=3.12 -y
conda activate cits4404
pip install -r requirements.txt
```

**With venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the experiment runner

The main entry point is `experiment_runner.py`. Before running, open the file and set the configuration flags at the top of the script (e.g. number of runs, population size, strategies to include).

Then run:

```bash
python experiment_runner.py
```

This produces two output files in `results/`:

| File | Contents |
|------|----------|
| `results/fixed_runs.csv` | Results from fixed-seed (deterministic) runs |
| `results/random_runs.csv` | Results from random-seed runs |

### 4. Generate figures and tables

After the experiment runner completes, open `notebook.ipynb` and run all cells top-to-bottom. The notebook reads the CSV files above and produces the tables and figures reported in the paper.

---

## AI Usage Disclaimer

AI assistance was used in the following specific, non-algorithmic capacities only:

- Refactoring manual Python loops to be NumPy-compliant (to benefit from Numpy vectorization)
- Implementing multithreading in the experiment runner (due to number of trials exercised)
- Appending new arguments to existing functions across 20+ separate files
- Refactoring to remove code duplication
- Restructuring this README

All algorithmic logic, experimental design, and analysis are the authors' own work.

---

## Development Setup

> The content below is preserved from the original project coding guide.

---

### Coding Guide

This document describes the coding standards and development workflow used in this repository. The goal is to maintain consistent code style, clean collaboration, and predictable releases.

---

### Naming Conventions

Follow these naming conventions across the repository:

| Type                        | Convention              | Example            |
| --------------------------- | ----------------------- | ------------------ |
| Variables                   | snake_case              | data_frame         |
| Functions                   | snake_case              | load_dataset()     |
| Constants                   | PascalCase              | ModelConfig        |
| Public Classes / Interfaces | PascalCase              | DataProcessor      |
| Private/Internal Functions  | _underscored_snake_case | _internal_helper() |

Keeping naming consistent makes the code easier to read and maintain.

> Unused variables can be replaced with just underscore like this ` _, b = (0, 2.36)`

---

### Development Workflow

This repository follows a Git Flow–inspired workflow to maintain stability while allowing active development.

**Branch Flow**

dev → feature → PR → dev → release PR → main

---

### Repository Branch Structure

#### main

Stable, production-ready code. Always deployable or demo ready. Protected branch. No direct pushes allowed.

#### dev

Integration branch for development. All feature work is merged here first. Protected branch.

#### Feature Branches

Feature branches are created from dev.

Naming format:

```
feature/#<ticketNo>-<short-description>
```

Example:

```
feature/#42-risk-classification
```

---

### Development Workflow Steps

#### 1. Start a Feature

Pull the latest dev branch and create a feature branch.

```bash
git checkout dev
git pull origin dev
git checkout -b feature/#ticketNo-short-description
```

#### 2. Work on the Feature

- Commit regularly.
- Use clear commit messages.
- Ensure code runs locally before pushing.

#### 3. Sync With Latest dev

Before creating a Pull Request, update your branch with the latest changes from dev.

```bash
git checkout dev
git pull origin dev
git checkout feature/#<ticketNo>-<feature-name>
git merge dev
```

#### 4. Create Pull Request

Open a Pull Request: `feature/#<ticketNo>-<feature-name> → dev`

The PR description should include what was implemented and any notes for reviewers.

#### 5. Code Review

At least one team member approval is required. Address comments before merging.

#### 6. Merge Feature

Use Merge Commit when merging into dev. After merging, delete the feature branch (optional but recommended to keep the repository clean).
