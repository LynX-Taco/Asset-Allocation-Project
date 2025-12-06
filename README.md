# 🚀 Asset Allocation — V2 Enhanced Model Demo

This repository now documents the **Version 2 (V2) MINLP workflow** for asset allocation. The demo aligns with the Colab notebook that walks through installing solvers, pulling the V2 branch, running the Pyomo model, and saving results for backtesting.

## What V2 Includes
- Solver installation snippets for GLPK (default) and optional IPOPT/Bonmin.
- How to load the **`v2-enhanced`** branch from GitHub in Colab.
- Running the Pyomo MINLP model with:
  - Binary activation variables
  - Linking constraints
  - Cardinality constraint
  - Sector constraints
- Saving and reading V2 outputs
- Template hooks for backtesting and risk scenarios

## Repository Layout (V2)
```
Asset-Allocation-Project/
└── portfolio_pipeline/
    └── v2/
        ├── model/model_minlp.py
        ├── runner_v2.py
        ├── outputs/
        └── notebooks/v2_demo.ipynb  ← Colab walkthrough
```

## Quickstart in Colab
Follow these cells in a Colab notebook to mirror the V2 demo.

### 1) Install Python dependencies and GLPK
```bash
%%bash
set -e
cd /content  # Ensure we are in a stable directory
echo "Installing Pyomo, yfinance and GLPK..."
pip install -q pyomo yfinance nbformat
apt-get update -qq
apt-get install -y -qq glpk-utils rsync
python - <<'PY'
import pyomo.environ as pyo
import pyomo.version
print('Pyomo', pyomo.version.version)
PY
```

### 2) Clone the repo and check out the V2 branch
```bash
%%bash
set -e
cd /content
rm -rf /content/Asset-Allocation-Project
# clone
git clone https://github.com/LynX-Taco/Asset-Allocation-Project.git /content/Asset-Allocation-Project
cd /content/Asset-Allocation-Project
# ensure branch exists remotely; if not, create it locally
git fetch --all || true
git checkout v2-enhanced || git checkout -b v2-enhanced
pwd
ls -la
```

### 3) Ensure the V2 structure exists
```bash
%%bash
set -e
cd /content/Asset-Allocation-Project
mkdir -p portfolio_pipeline/v2/model
mkdir -p portfolio_pipeline/v2/backtest
mkdir -p portfolio_pipeline/v2/outputs/v2_run_example
```

### 4) Run the V2 runner
```bash
%%bash
set -e
cd /content/Asset-Allocation-Project
python portfolio_pipeline/v2/runner_v2.py \
  --solver glpk \
  --choose-at-least 1 \
  --min-alloc 0.01 \
  --max-alloc 0.8 \
  --start-date 2024-01-01 \
  --end-date 2025-07-31

ls -la portfolio_pipeline/v2/outputs/v2_run_example
if [ -f portfolio_pipeline/v2/outputs/v2_run_example/allocations.csv ]; then
  sed -n '1,50p' portfolio_pipeline/v2/outputs/v2_run_example/allocations.csv
fi
```

### 5) Backtesting hook
If `portfolio_pipeline/v2/backtest/backtest.py` is present, the runner will attempt a simple backtest using the saved allocations. Results (CSV and optional PNG) are written next to the allocations file.

## Notes on Solvers
- **GLPK** is installed in the quickstart and works for the demo MINLP.
- **IPOPT** or **Bonmin** can be installed separately if you want nonlinear or more advanced MINLP support (see `model_bonmin.py` references in the V2 notebook).

## Local Development
If you are working locally instead of Colab:
1. Install dependencies: `pip install -r requirements.txt` (plus any solver binaries you need).
2. Run `python portfolio_pipeline/v2/runner_v2.py --solver glpk` from the repository root.
3. Inspect outputs under `portfolio_pipeline/v2/outputs/` for allocation CSVs and any backtest artifacts.

This README now aligns with the V2 enhanced model instructions and replaces the prior mean-variance-only guidance.
