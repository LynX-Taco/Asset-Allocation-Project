# Version2 (V2) for BDM

## 🚀 Asset Allocation — *V2 Enhanced Model Demo*
This repository documents the **V2 MINLP workflow** for asset allocation. The accompanying Colab notebook shows how to install solvers, pull the `v2-enhanced` branch, and run the Pyomo model end-to-end — from building binary-linked positions to exporting outputs that feed a simple backtest helper.

### ✔ What's included
- Minimal Pyomo MINLP with binary activation variables, linking constraints, cardinality, and sector rules.
- Runner that exposes the V2 parameters (`--min-alloc`, `--max-alloc`, `--choose-at-least`, `--tickers`, `--start-date`, `--end-date`, `--solver`).
- Robust backtest helper that consumes the allocations CSV and saves portfolio value time series + plot.
- Optional Bonmin variant for mixed-integer nonlinear risk frontiers.

---

## 📌 Repository requirements and layout
This notebook/guide expects access to the repository with the V2 folders already present:

```
Asset-Allocation-Project/
└── portfolio_pipeline/
    └── v2/
        ├── model/
        │   ├── model_minlp.py
        │   └── model_bonmin.py           # optional, Bonmin-only
        ├── backtest/backtest.py          # optional but recommended
        ├── runner_v2.py
        ├── outputs/
        └── notebooks/v2_demo.ipynb       # you are here in Colab
```

Key defaults from the code:
- `runner_v2.py`: `min_alloc=0.02`, `max_alloc=0.20`, `choose_at_least=2`, default tickers `AAPL MSFT GOOGL AMZN`, solver `glpk`.
- `model_minlp.py`: budget equals 1, linking constraints enforce bounds only when a binary is active, sector constraints require at least one per declared sector.
- `backtest/backtest.py`: downloads prices with `auto_adjust=True`, normalizes weights for available tickers, writes `backtest_results.csv` and optional PNG next to the allocations.

---

## 🧭 Quickstart in Colab
Mirror these cells in Colab to recreate the V2 demo environment.

### 1) Install Python deps and GLPK
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

### 2) Clone the repo and switch to the V2 branch
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

### 3) Prepare the V2 folder structure (idempotent)
```bash
%%bash
set -e
cd /content/Asset-Allocation-Project
mkdir -p portfolio_pipeline/v2/model
mkdir -p portfolio_pipeline/v2/backtest
mkdir -p portfolio_pipeline/v2/outputs/v2_run_example
```

### 4) Run the V2 runner with the new parameters
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
If `portfolio_pipeline/v2/backtest/backtest.py` exists, `runner_v2.py` will attempt to load it and run `run_backtest_from_alloc`, producing `backtest_results.csv` and (optionally) `backtest_value.png` alongside the allocations file.

---

## 🧰 Solver notes
- **GLPK**: installed in the quickstart and works for the demo MINLP.
- **Bonmin/IPOPT**: optional installs if you want nonlinear MINLP. See `model_bonmin.py` for Bonmin-specific helper functions and risk frontier generation.

---

## 🛠 Local development
1. Install dependencies: `pip install -r requirements.txt` plus any solver binaries you need.
2. Run the demo: `python portfolio_pipeline/v2/runner_v2.py --solver glpk --min-alloc 0.02 --max-alloc 0.20 --choose-at-least 2` (override parameters as desired).
3. Inspect outputs under `portfolio_pipeline/v2/outputs/` for allocation CSVs, backtest artifacts, and any Bonmin frontier plots.

This README now mirrors the V2 enhanced model parameters and notebook-driven workflow.
