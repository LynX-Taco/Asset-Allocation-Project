# portfolio_pipeline/v2/portfolio_pipeline.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
# Colab / headless friendly backend
if "google.colab" in sys.modules:
    matplotlib.use("module://matplotlib_inline.backend_inline")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your V2 model + backtest helpers
from portfolio_pipeline.v2.model.model_minlp import (
    build_model,
    solve_model,
    save_results,
)
from portfolio_pipeline.v2.backtest.backtest import run_backtest_from_alloc


# ---------------------------------------------------------------------
# 1) Data: download daily prices and convert to monthly returns (like V1)
# ---------------------------------------------------------------------
def calculate_monthly_returns(tickers, start_date, end_date):
    """
    Download daily prices via yfinance and compute monthly returns.

    Returns
    -------
    monthly_returns : DataFrame
        Index: month-end dates
        Columns: tickers
    """
    prices = {}
    for t in tickers:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                print(f"Warning: no data for {t}")
                continue
            prices[t] = df["Close"]
        except Exception as e:
            print(f"Failed to download {t}: {e}")

    if not prices:
        print("No price data downloaded; check tickers and date range.")
        return None

    daily = pd.DataFrame(prices).dropna(how="all")
    if daily.empty:
        print("Daily price DataFrame is empty after alignment.")
        return None

    daily_returns = daily.pct_change().dropna(how="all")
    monthly_returns = (1.0 + daily_returns).resample("ME").prod() - 1.0
    monthly_returns = monthly_returns.dropna(how="all")
    return monthly_returns


# ---------------------------------------------------------------------
# 2) Optimization: use your V2 MINLP model (GLPK or other MILP solver)
# ---------------------------------------------------------------------
def run_minlp_allocation(
    df_returns,
    solver="glpk",
    min_alloc=0.02,
    max_alloc=0.20,
    choose_at_least=2,
    sectors_map=None,
    out_dir="portfolio_pipeline/v2/outputs/v2_run_example",
):
    """
    Build and solve the V2 MINLP portfolio model using expected returns
    derived from historical data.

    Parameters
    ----------
    df_returns : DataFrame
        Monthly returns, index = dates, columns = tickers.
    solver : str
        Pyomo solver name (e.g., 'glpk').
    min_alloc, max_alloc : float
        Bounds for each selected asset's allocation.
    choose_at_least : int
        Cardinality lower bound (at least this many assets selected).
    sectors_map : dict or None
        Optional sector mapping: {sector_name: [tickers_in_sector]}.
    out_dir : str
        Directory where allocations.csv is written.

    Returns
    -------
    allocations_df : DataFrame
        Columns: ['ticker', 'weight', 'selected']
    alloc_csv_path : pathlib.Path
        Path to the allocations CSV file.
    """
    tickers = df_returns.columns.tolist()
    # Simple expected returns: average monthly return (you could annualize if you like)
    expected_returns = df_returns.mean().to_dict()

    # Build model
    model = build_model(
        tickers=tickers,
        expected_returns=expected_returns,
        sectors_map=sectors_map,
        min_alloc=min_alloc,
        max_alloc=max_alloc,
        choose_at_least=choose_at_least,
    )

    print(f"Built V2 MINLP model with {len(tickers)} tickers. Solving with {solver}...")
    try:
        res = solve_model(model, solver=solver, tee=False)
        try:
            print("Solver status:", res.solver.status, res.solver.termination_condition)
        except Exception:
            print("Solver finished (no status available).")
    except Exception as e:
        print("Solver failed:", type(e).__name__, e)
        return None, None

    # Save allocations to CSV using your helper
    alloc_csv = save_results(model, out_dir=out_dir)
    alloc_csv = Path(alloc_csv)

    # Load into a tidy DataFrame for return to caller
    try:
        allocations_df = pd.read_csv(alloc_csv)
    except Exception as e:
        print("Failed to read allocations CSV:", type(e).__name__, e)
        allocations_df = None

    return allocations_df, alloc_csv


# ---------------------------------------------------------------------
# 3) Optional backtest: use your robust backtest helper
# ---------------------------------------------------------------------
def maybe_run_backtest(alloc_csv_path, start_date, end_date, do_backtest=True):
    """
    Run backtest_from_alloc if requested and possible.

    Returns
    -------
    backtest_df : DataFrame or None
    """
    if not do_backtest:
        print("Backtest skipped by user flag.")
        return None

    try:
        out_csv = run_backtest_from_alloc(
            alloc_csv_path, start=start_date, end=end_date
        )
        print("Backtest saved to:", out_csv)
        backtest_df = pd.read_csv(out_csv, parse_dates=["date"]).set_index("date")
        return backtest_df
    except Exception as e:
        print("Backtest skipped or failed:", type(e).__name__, e)
        return None


# ---------------------------------------------------------------------
# 4) PUBLIC ENTRY POINT used by V2 main.py
# ---------------------------------------------------------------------
def run_portfolio_pipeline_v2(
    tickers,
    start_date,
    end_date,
    solver="glpk",
    min_alloc=0.02,
    max_alloc=0.20,
    choose_at_least=2,
    sectors_map=None,
    do_backtest=True,
):
    """
    One-stop V2 pipeline:
      1) Download prices & compute monthly returns
      2) Solve MINLP asset-allocation model
      3) (Optionally) backtest the resulting allocations

    Parameters are passed straight through to the helpers above.

    Returns
    -------
    mret : DataFrame or None
        Monthly returns.
    allocations_df : DataFrame or None
        Ticker / weight / selected.
    backtest_df : DataFrame or None
        Backtest results if available.
    """
    # Ensure list
    if isinstance(tickers, str):
        tickers_list = [t.strip() for t in tickers.split() if t.strip()]
    else:
        tickers_list = list(tickers)

    print("Starting V2 portfolio pipeline...")
    print("Tickers:", tickers_list)
    print("Return window:", start_date, "to", end_date)

    # 1) Data
    mret = calculate_monthly_returns(tickers_list, start_date, end_date)
    if mret is None or mret.empty:
        print("No valid monthly returns; aborting V2 pipeline.")
        return None, None, None

    print("Monthly returns DataFrame shape:", mret.shape)

    # 2) Optimization (MINLP)
    allocations_df, alloc_csv = run_minlp_allocation(
        df_returns=mret,
        solver=solver,
        min_alloc=min_alloc,
        max_alloc=max_alloc,
        choose_at_least=choose_at_least,
        sectors_map=sectors_map,
    )

    if allocations_df is None:
        print("No allocations produced; aborting backtest.")
        return mret, None, None

    print("Allocations CSV:", alloc_csv)
    print("Allocations preview:")
    print(allocations_df.head())

    # 3) Backtest (optional)
    backtest_df = maybe_run_backtest(
        alloc_csv_path=alloc_csv,
        start_date=start_date,
        end_date=end_date,
        do_backtest=do_backtest,
    )

    return mret, allocations_df, backtest_df
