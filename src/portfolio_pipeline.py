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
    Returns: df_results, df_allocations
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

    # Use fewer points if you want less noise (e.g., 51 instead of 201)
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
    plt.show()

    print("Portfolio optimization and plotting complete.")
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
    """
    print("Starting full portfolio analysis...")

    df_returns = calculate_monthly_returns(tickers_list, start_date, end_date)
    if df_returns is None or df_returns.empty:
        print("No valid return data; aborting analysis.")
        return None, None

    return optimize_and_plot_portfolio(df_returns, ipopt_executable)


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
        df_results, df_allocations
    """
    # Accept tickers as space-separated string or list
    if isinstance(tickers, str):
        tickers_list_local = [t.strip() for t in tickers.split() if t.strip()]
    else:
        tickers_list_local = list(tickers)

    df_results, df_allocations = perform_full_portfolio_analysis(
        tickers_list_local, start_date, end_date, ipopt_executable
    )

    if df_results is not None and df_allocations is not None:
        print("Final results and allocations obtained.")
        display(df_results.head())
        display(df_allocations.head())

    return df_results, df_allocations


# ------------------------------------------------------------
# Demo block – only runs if you execute THIS file directly
# (does NOT run when imported by main.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    ipopt_executable = "/content/bin/ipopt"  # adjust path if needed

    # Example: run on default tickers_list and date range
    _df_results, _df_allocations = run_portfolio_pipeline(
        ipopt_executable, start, end, tickers_list
    )
