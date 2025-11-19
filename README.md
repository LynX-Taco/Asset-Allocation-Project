# Asset Allocation Project

This repository walks you through running a mean-variance portfolio optimization workflow with your own tickers. It explains how to set up dependencies, fetch historical data from Yahoo Finance, sweep risk limits with IPOPT, and visualize both the efficient frontier and allocation profiles.

## Quick start (local)
1) **Clone and enter the repo**
```bash
git clone https://github.com/your-username/asset-allocation-project.git
cd asset-allocation-project
```

2) **Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\\Scripts\\Activate.ps1  # Windows PowerShell
```

3) **Install dependencies**
```bash
python -m pip install -r requirements.txt
```

4) **Get IPOPT via IDAES** (installs the solver to `./bin`)
```bash
python -m idaes get-extensions --to ./bin
```

5) **Run with your own tickers**
```bash
TICKERS="AAPL MSFT NVDA AMZN GOOGL"
python main.py --ipopt ./bin/ipopt --start 2022-01-01 --end 2024-01-01 --tickers ${TICKERS}
```

## Running in Google Colab
1) Upload or clone the repo inside Colab.
2) Install dependencies and IPOPT (goes to `/content/bin` by default):
```python
!pip install -r requirements.txt
!python -m idaes get-extensions --to /content/bin
```
3) Point the CLI to the IPOPT binary:
```python
!python main.py --tickers GE KO NVDA --start-date 2020-01-01 --end-date 2024-01-01 --ipopt-path /content/bin/ipopt
```
The script displays two Matplotlib figures: the efficient frontier and allocation-by-risk chart.

## CLI usage
The core options available in `main.py`:
```bash
python main.py \
  --tickers GE KO NVDA \
  --start-date 2020-01-01 \
  --end-date 2024-01-01 \
  --n-points 250 \
  --ipopt-path ./bin/ipopt
```
- `--tickers`: space-separated equities.
- `--start-date` / `--end-date`: inclusive date range pulled from Yahoo Finance.
- `--n-points`: number of variance caps to trace the efficient frontier (higher = denser curve).
- `--ipopt-path`: path to the IPOPT solver installed via `idaes get-extensions`.

## What the pipeline does
1) Downloads daily prices for the tickers you provide.
2) Converts prices to daily returns, then to monthly compounded returns.
3) Builds a Pyomo optimization model to maximize expected return under a budget constraint and a rolling variance cap.
4) Sweeps risk levels with IPOPT to trace the efficient frontier and record allocations for each cap.
5) Plots the frontier (risk vs. expected return) and allocation paths (weights vs. risk).

## Notes
- If the solver is not found on your PATH, pass the explicit location with `--ipopt-path`.
- IPOPT binaries installed by IDAES live in `~/.idaes/bin` by default; the examples above use a local `./bin/ipopt` or `/content/bin/ipopt` path for clarity.
- To change tickers, simply update the `--tickers` argument with a space-separated list.
