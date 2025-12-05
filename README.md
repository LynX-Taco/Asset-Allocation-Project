# Asset Allocation Project

This repository walks you through running a mean-variance portfolio optimization workflow with your own tickers. It explains how to set up dependencies, fetch historical data from Yahoo Finance, sweep risk limits with IPOPT, and visualize both the efficient frontier and allocation profiles.

The runnable code now lives in the **Portfolio Pipeline** folder, which includes:
- `main.py`: CLI entry point for downloading data, optimizing, and plotting.
- `requirements.txt`: Python dependencies (Pyomo, idaes-pse, yfinance, etc.).
- `.gitignore`: Local artifacts to keep out of version control.

## Quick start (local)
1) **Clone and enter the repo**
```bash
!git clone https://github.com/your-username/asset-allocation-project.git
!cd asset-allocation-project
```
2) **Install dependencies**
```bash
!pip install -r "Portfolio Pipeline/requirements.txt"
```

3) **Get IPOPT via IDAES** (installs the solver to `./bin`)
```bash
!idaes get-extensions --to ./bin
```

4) **Run with your own tickers**
```bash
TICKERS="AAPL MSFT NVDA AMZN GOOGL"
python "Portfolio Pipeline/main.py" --ipopt-path ./bin/ipopt --start-date 2022-01-01 --end-date 2024-01-01 --tickers "${TICKERS}"
```

## Running in Google Colab
1) Upload or clone the repo inside Colab.
2) Install dependencies and IPOPT (goes to `/content/bin` by default):
```python
!pip install -r "Portfolio Pipeline/requirements.txt"
!python -m idaes get-extensions --to /content/bin
```
3) Point the CLI to the IPOPT binary:
```python
!python "Portfolio Pipeline/main.py" --tickers GE KO NVDA --start-date 2020-01-01 --end-date 2024-01-01 --ipopt-path /content/bin/ipopt
```
