import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import yfinance as yf
from pyomo.environ import (
    ConcreteModel,
    Var,
    Set,
    Param,
    Constraint,
    NonNegativeReals,
    Objective,
    maximize,
)
from pyomo.opt import SolverFactory, TerminationCondition

import os
import sys

# ------------------------------------------------------------
# Colab-friendly Matplotlib setup (must be before pyplot import)
# ------------------------------------------------------------
in_colab = "google.colab" in sys.modules
if in_colab:
    print("Running in Colab — setting matplotlib inline backend")
    matplotlib.use("module://matplotlib_inline.backend_inline")

import matplotlib.pyplot as plt
from IPython.display import display

# ------------------------------------------------------------
# Default tickers and date range (for demo / __main__)
# ------------------------------------------------------------
tickers_list = [
    "COST", "KO", "TGT", "WMT", "PEP",
    "XOM", "TSLA", "CVX", "PSX", "SLB",
    "HON", "GE", "CAT", "LMT", "FDX",
]  # removed SOB to avoid warnings

start = "2022-01-01"
end = "2024-01-01"


# ------------------------------------------------------------
# Function to calculate monthly returns
# ------------------------------------------------------------
def calculate_monthly_returns(tickers_list, start_date, end_date):
    """
    Download daily prices from Yahoo Finance and compute monthly returns.
    """
    dow_prices = {}
    for t in tickers_list:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                dow_prices[t] = df
            else:
                print(f"Warning: no data returned for {t}")
        except Exception as e:
            print(f"Failed {t}: {e}")

    if not dow_prices:
        print("No stock data was downloaded. Please check tickers and dates.")
        return None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        if not data.empty:
            returns = data["Close"].pct_change().dropna()
            if len(returns) > 1:
                return_data_dict[ticker] = returns

    if not return_data_dict:
        print("No valid stock data available after calculating returns.")
        return None

    daily_returns = pd.concat(
        return_data_dict.values(), axis=1, keys=return_data_dict.keys()
    )
    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    return monthly_returns.dropna()


# ------------------------------------------------------------
# Optimization and plotting – efficient frontier + allocations
# ------------------------------------------------------------
def optimize_and_plot_portfolio(df_returns, ipopt_executable):
    """
    Given a monthly returns DataFrame, build a Markowitz model with Pyomo,
    trace out the efficient frontier for a range of risk levels, and plot:
      1) Efficient Frontier
      2) Asset Allocation vs Risk
    Returns: (df_results, df_allocations)
    """
    # Build model
    m = ConcreteModel()

    assets = df_returns.columns.tolist()
    m.Assets = Set(initialize=assets)

    # Decision variables: portfolio weights
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0, 1))

    # Expected returns
    avg_returns = df_returns.mean().to_dict()
    m.mu = Param(m.Assets, initialize=avg_returns)

    # Covariance matrix
    cov_df = df_returns.cov()
    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=cov_dict)

    # Objective: maximize expected return
    def total_return_rule(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)

    m.objective = Objective(rule=total_return_rule, sense=maximize)

    # Budget constraint: fully invested
    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1

    m.budget = Constraint(rule=budget_constraint_rule)

    print(
        "Pyomo model initialized with sets, variables, parameters, "
        "objective, and budget constraint."
    )

    # Solver: IPOPT
    solver = SolverFactory("ipopt")
    if not solver.available():
        solver = SolverFactory("ipopt", executable=ipopt_executable)

    # Risk grid (variance upper bounds) for efficient frontier
    max_possible_variance = np.max(np.diag(cov_df.values))
    max_risk_for_range = max_possible_variance * 1.5
    min_risk_for_range = 1e-6

    # Use fewer points if you want less noise (e.g., 51–101 instead of 201)
    risk_limits = np.linspace(min_risk_for_range, max_risk_for_range, 101)

    param_analysis = {}
    returns = {}

    print(f"Starting portfolio optimization for {len(risk_limits)} risk levels...")
    for r in risk_limits:
        # Remove old variance constraint if it exists
        if hasattr(m, "variance_constraint"):
            m.del_component(m.variance_constraint)

        def variance_constraint_rule(m):
            return (
                sum(
                    m.Sigma[i, j] * m.x[i] * m.x[j]
                    for i in m.Assets
                    for j in m.Assets
                )
                <= r
            )

        m.variance_constraint = Constraint(rule=variance_constraint_rule)

        result = solver.solve(m)

        # Skip infeasible / weird solutions
        if result.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.other,
        ]:
            continue

        if result.solver.termination_condition in [
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
        ]:
            param_analysis[r] = [m.x[a]() for a in m.Assets]
            returns[r] = m.objective()

    if not returns:
        print("No feasible solutions found for any risk level.")
        return None, None

    # Frontier DataFrame
    df_results = pd.DataFrame(
        {"Risk": list(returns.keys()), "Return": list(returns.values())}
    ).sort_values(by="Risk")

    # Efficient Frontier plot (Colab-friendly)
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Risk"], df_results["Return"], marker="o", linestyle="-")
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.tight_layout()
    # 🔽 Save + show
    plt.savefig("efficient_frontier.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Allocations DataFrame
    df_allocations = pd.DataFrame(param_analysis).T
    df_allocations.columns = assets
    df_allocations["Risk"] = df_allocations.index

    # Asset Allocation vs Risk plot (Colab-friendly)
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(
            df_allocations["Risk"],
            df_allocations[asset],
            marker="o",
            markersize=4,
            label=str(asset),
        )
    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    # 🔽 Save + show
    plt.savefig("allocation_vs_risk.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Portfolio optimization and plotting complete.")
    print("Saved plots as 'efficient_frontier.png' and 'allocation_vs_risk.png'.")
    return df_results, df_allocations


print("Finished defining `optimize_and_plot_portfolio` function.")


# ------------------------------------------------------------
# Full portfolio analysis: downloads prices + optimizes + plots
# ------------------------------------------------------------
def perform_full_portfolio_analysis(tickers_list, start_date, end_date, ipopt_executable):
    """
    One-stop function:
      1) Download prices
      2) Compute monthly returns
      3) Optimize and plot frontier + allocations

    Returns:
        df_returns, df_results, df_allocations
    """
    print("Starting full portfolio analysis...")

    df_returns = calculate_monthly_returns(tickers_list, start_date, end_date)
    if df_returns is None or df_returns.empty:
        print("No valid return data; aborting analysis.")
        return None, None, None

    df_results, df_allocations = optimize_and_plot_portfolio(
        df_returns, ipopt_executable
    )

    return df_returns, df_results, df_allocations


print("Defined `perform_full_portfolio_analysis` function.")


# ------------------------------------------------------------
# Wrapper used by main.py (for CLI-style usage)
# ------------------------------------------------------------
def run_portfolio_pipeline(ipopt_executable, start_date, end_date, tickers):
    """
    Entry point for main.py.

    Args:
        ipopt_executable (str): Path to ipopt executable (e.g., /content/bin/ipopt)
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        tickers (str or list): e.g. "AAPL MSFT NVDA" or ["AAPL","MSFT","NVDA"]

    Returns:
        mret (DataFrame): monthly returns
        frontier (DataFrame): risk/return frontier
        allocs (DataFrame): allocations by risk level
    """
    # Accept tickers as space-separated string or list
    if isinstance(tickers, str):
        tickers_list_local = [t.strip() for t in tickers.split() if t.strip()]
    else:
        tickers_list_local = list(tickers)

    mret, frontier, allocs = perform_full_portfolio_analysis(
        tickers_list_local, start_date, end_date, ipopt_executable
    )

    if frontier is not None and allocs is not None:
        print("Final results and allocations obtained.")
        display(frontier.head())
        display(allocs.head())

    return mret, frontier, allocs


# ------------------------------------------------------------
# Demo block – only runs if you execute THIS file directly
# (does NOT run when imported by main.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    ipopt_executable = "/content/bin/ipopt"  # adjust path if needed

    # Example: run on default tickers_list and date range
    _mret, _df_results, _df_allocations = run_portfolio_pipeline(
        ipopt_executable, start, end, tickers_list
    )


# ---------------------------
# Helper: save V1 outputs to disk (no changes to existing functions)
# ---------------------------
import shutil
import pathlib
import time

def save_v1_outputs(tickers, df_monthly_returns, df_results, df_allocations, output_dir="portfolio_pipeline/v1/outputs/v1_run_example", download_daily=True, start=None, end=None):
    """
    Save V1 pipeline outputs to a single folder.

    Args:
      tickers: list of tickers used (or str)
      df_monthly_returns: DataFrame returned by calculate_monthly_returns(...)
      df_results: DataFrame returned by optimize_and_plot_portfolio(...) (frontier)
      df_allocations: DataFrame returned by optimize_and_plot_portfolio(...) (allocations)
      output_dir: directory to write CSVs and images into
      download_daily: if True, re-download daily Close prices and save them
      start, end: optional date strings used for download (if download_daily True)
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Save monthly returns (we already have it)
    try:
        if df_monthly_returns is not None:
            df_monthly_returns.to_csv(out / "monthly_returns.csv", index=True)
            print("Saved monthly_returns.csv")
    except Exception as e:
        print("Could not save monthly_returns.csv:", e)

    # 2) Optionally re-download daily Close prices and save
    if download_daily:
        try:
            import yfinance as yf
            tickers_list = tickers if isinstance(tickers, (list, tuple)) else [t.strip() for t in str(tickers).split()]
            price_frames = []
            for t in tickers_list:
                try:
                    df_daily = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
                    if not df_daily.empty and "Close" in df_daily.columns:
                        df_close = df_daily["Close"].rename(t)
                        price_frames.append(df_close)
                        # small sleep to be polite
                        time.sleep(0.1)
                except Exception as ex:
                    print(f"Warning: failed to download daily {t}: {ex}")

            if price_frames:
                daily_prices_df = pd.concat(price_frames, axis=1).sort_index()
                daily_prices_df.to_csv(out / "price_data_daily.csv")
                daily_returns_df = daily_prices_df.pct_change().dropna()
                daily_returns_df.to_csv(out / "daily_returns.csv")
                print("Saved daily price CSVs (price_data_daily.csv, daily_returns.csv)")
            else:
                print("No daily price frames downloaded; skipping daily CSVs.")
        except Exception as e:
            print("Could not download/save daily prices:", e)

    # 3) Save frontier and allocations as CSVs
    try:
        if df_results is not None:
            df_results.to_csv(out / "efficient_frontier.csv", index=False)
            print("Saved efficient_frontier.csv")
    except Exception as e:
        print("Could not save efficient_frontier.csv:", e)

    try:
        if df_allocations is not None:
            # if df_allocations has index = risk levels, preserve it
            df_allocations.to_csv(out / "allocations_by_risk.csv", index=True)
            print("Saved allocations_by_risk.csv")
    except Exception as e:
        print("Could not save allocations_by_risk.csv:", e)

    # 4) Move or copy generated plot files if they were saved to CWD
    #    Your functions save 'efficient_frontier.png' and 'allocation_vs_risk.png' in cwd.
    possible_plots = {
        "efficient_frontier.png": out / "efficient_frontier.png",
        "allocation_vs_risk.png": out / "allocation_vs_risk.png",
    }
    for src_name, dest_path in possible_plots.items():
        src = pathlib.Path(src_name)
        if src.exists():
            try:
                # copy rather than move so original remains for interactive view
                shutil.copy2(src, dest_path)
                print(f"Copied {src_name} -> {dest_path}")
            except Exception as e:
                print(f"Could not copy {src_name}: {e}")
        else:
            # if original plot not found, try to recreate from data
            print(f"{src_name} not found in CWD; attempting to replot from data...")

            try:
                if src_name == "efficient_frontier.png" and df_results is not None:
                    plt.figure(figsize=(10,6))
                    plt.plot(df_results["Risk"], df_results["Return"], marker="o", linestyle="-")
                    plt.title("Efficient Frontier")
                    plt.xlabel("Portfolio Risk (Variance)")
                    plt.ylabel("Expected Return")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(dest_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"Re-saved {dest_path} from df_results")
                elif src_name == "allocation_vs_risk.png" and df_allocations is not None:
                    plt.figure(figsize=(12,6))
                    # df_allocations expected to have a 'Risk' column or index
                    if "Risk" in df_allocations.columns:
                        x = df_allocations["Risk"]
                    else:
                        x = df_allocations.index
                    for col in [c for c in df_allocations.columns if c != "Risk"]:
                        plt.plot(x, df_allocations[col], marker="o", markersize=4, label=str(col))
                    plt.title("Asset Allocation as a Function of Portfolio Risk")
                    plt.xlabel("Portfolio Risk (Variance)")
                    plt.ylabel("Proportion Invested")
                    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(dest_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"Re-saved {dest_path} from df_allocations")
                else:
                    print(f"Insufficient data to replot {src_name}.")
            except Exception as e:
                print(f"Failed to replot {src_name}: {e}")

    print(f"\nAll artifacts attempted to be saved in: {out.resolve()}")
    print("List of files in output dir:")
    try:
        for p in sorted(out.iterdir()):
            print(" -", p.name)
    except Exception:
        pass

