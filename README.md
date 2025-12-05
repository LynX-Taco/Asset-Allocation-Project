# Asset Allocation Project

This repository walks you through running a mean-variance portfolio optimization workflow with your own tickers. It explains how to set up dependencies, fetch historical data from Yahoo Finance, sweep risk limits with IPOPT, and visualize both the efficient frontier and allocation profiles.

The runnable code now lives in the **Portfolio Pipeline** folder, which includes:
- `main.py`: CLI entry point for downloading data, optimizing, and plotting.
- `requirements.txt`: Python dependencies (Pyomo, idaes-pse, yfinance, etc.).
- `.gitignore`: Local artifacts to keep out of version control.

## Colab File Path
1) **Clone and enter the repo**
```bash
!git clone https://github.com/your-username/Asset-Allocation-Project.git
%cd Asset-Allocation-Project
!ls
```
2) **Install dependencies**
```bash
!pip install -r requirements.txt
!pip install idaes-pse
```

3) **Get IPOPT via IDAES** (installs the solver to `./bin`)
```bash
!idaes get-extensions --to /content/bin
!ls /content/bin

```

4) **Run with your own tickers**
```bash
TICKERS="AAPL MSFT NVDA AMZN GOOGL"

!python3 main.py \
    --ipopt /content/bin/ipopt \
    --start 2022-01-01 \
    --end 2024-01-01 \
    --tickers $TICKERS

```
5) **Printing the Plots**
```bash
   from IPython.display import Image, display

# Adjust path if needed, but this should be correct:

base_path = "/content/Asset-Allocation-Project"

display(Image(f"{base_path}/efficient_frontier.png"))
display(Image(f"{base_path}/allocation_vs_risk.png"))
```

