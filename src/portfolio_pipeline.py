import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import yfinance as yf
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition

import os
import sys
import subprocess

# ------------------------------------------------------------
# Ensure plots render correctly in Colab BEFORE importing pyplot
# ------------------------------------------------------------
in_colab = "google.colab" in sys.modules
if in_colab:
    print("Running in Colab — setting matplotlib to inline backend")
    matplotlib.use('module://matplotlib_inline.backend_inline')

import matplotlib.pyplot as plt
from IPython.display import display

plt.ion()  # Enable interactive mode

# ------------------------------------------------------------
# List of tickers and date range (defaults)
# ------------------------------------------------------------
tickers_list = [
    "COST", "KO", "TGT", "WMT", "PEP",
    "XOM", "TSLA", "CVX", "PSX", "SOB",
    "SLB", "HON", "GE", "CAT", "LMT", "FDX"
]
start = '2022-01-01'
end = '2024-01-01'


# ------------------------------------------------------------
# Function to calculate monthly returns
# ------------------------------------------------------------
def calculate_monthly_returns(tickers_list, start_date, end_date):
    dow_prices = {}
    for t in tickers_list:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                interval='1d',
                progress=False,
                auto_adjust=False
            )
            if not df.empty:
                dow_prices[t] = df
            else:
                print(f'Warning: no data returned for {t}')
        except Exception as e:
            print(f'Failed {t}: {e}')

    if not dow_prices:
        print("No stock data was downloaded. Please check tickers and dates.")
        return None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 1:
                return_data_dict[ticker] = returns

    if not return_data_dict:
        print("No valid stock data available after calculating returns.")
        return None

    daily_returns = pd.concat(
        return_data_dict.values(), axis=1, keys=return_data_dict.keys()
    )
    monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
    return monthly_returns.dropna()


# ------------------------------------------------------------
# Optimization and plotting
# ------------------------------------------------------------
def optimize_and_plot_portfolio(df_returns, ipopt_executable):
    # Pyomo model
    m = ConcreteModel()

    assets = df_returns.columns.tolist()
    m.Assets = Set(initialize=assets)

    # Decision variables: weights
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

    # Budget: fully invested
    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1

    m.budget = Constraint(rule=budget_constraint_rule)

    print("Pyomo model initialized with sets, variables, parameters, objective, and budget constraint.")

    # Setup IPOPT solver
    solver = SolverFactory("ipopt")
    if not solver.available():
        solver = SolverFactory("ipopt", executable=ipopt_executable)

    # Risk grid for efficient frontier
    max_possible_variance = np.max(np.diag(cov_df.values))
    max_risk_for_range = max_possible_variance * 1.5
    min_risk_for_range = 1e-6
    risk_limits = np.arange(
        min_risk_for_range,
        max_risk_for_range + 1e-6,
        (max_risk_for_range - min_risk_for_range) / 200,
    )

    param_analysis = {}
    returns = {}

    print(f"Starting portfolio optimization for {len(risk_limits)} risk levels...")
    for r in risk_limits:
        # Remove old variance constraint, if any
        if hasattr(m, 'variance_constraint'):
            m.del_component(m.variance_constraint)

        def variance_constraint_rule(m):
            return sum(
                m.Sigma[i, j] * m.x[i] * m.x[j]
                for i in m.Assets for j in m.Assets
            ) <= r

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

    df_results = pd.DataFrame(
        {'Risk': list(returns.keys()), 'Return': list(returns.values())}
    ).sort_values(by='Risk')

    # Efficient frontier plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Risk'], df_results['Return'], marker='o', linestyle='-')
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    display(plt.gcf())
    plt.close()

    # Allocation dataframe
    df_allocations = pd.DataFrame(param_analysis).T
    df_allocations.columns = assets
    df_allocations['Risk'] = df_allocations.index

    # Asset allocation plot
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(
            df_allocations['Risk'],
            df_allocations[asset],
            label=str(asset),
            marker='o',
            markersize=4,
        )
    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

    print("Portfolio optimization and plotting complete.")
    return df_results, df_allocations


print("Finished defining `optimize_and_plot_portfolio` function.")


# ------------------------------------------------------------
# Perform full portfolio analysis
# ------------------------------------------------------------
def perform_full_portfolio_analysis(tickers_list, start_date, end_date, ipopt_executable):
    print("Starting full portfolio analysis...")
    dow_prices = {}
    for t in tickers_list:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                interval='1d',
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                dow_prices[t] = df
            else:
                print(f"Warning: no data returned for {t}")
        except Exception as e:
            print(f'Failed {t}: {e}')

    if not dow_prices:
        print("No stock data downloaded.")
        return None, None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 1:
            return_data_dict[ticker] = returns

    if not return_data_dict:
        print("No valid return data.")
        return None, None

    df_returns = pd.concat(
        return_data_dict.values(), axis=1, keys=return_data_dict.keys()
    )
    df_returns = (1 + df_returns).resample('ME').prod() - 1
    df_returns = df_returns.dropna()

    if df_returns.empty:
        print("No valid return data after resampling.")
        return None, None

    return optimize_and_plot_portfolio(df_returns, ipopt_executable)


print("Defined `perform_full_portfolio_analysis` function.")


# ------------------------------------------------------------
# Wrapper used by main.py
# ------------------------------------------------------------
def run_portfolio_pipeline(ipopt_path, start_date, end_date, tickers):
    """
    Wrapper so main.py can call the full pipeline.

    Returns:
        (df_results, df_allocations, final_df_results, final_df_allocations)
    """
    df_returns = calculate_monthly_returns(tickers, start_date, end_date)
    df_results = df_allocations = None

    if df_returns is not None:
        df_results, df_allocations = optimize_and_plot_portfolio(df_returns, ipopt_path)

    final_df_results, final_df_allocations = perform_full_portfolio_analysis(
        tickers, start_date, end_date, ipopt_path
    )

    return df_results, df_allocations, final_df_results, final_df_allocations


# ------------------------------------------------------------
# Demo workflow when run directly (NOT when imported)
# ------------------------------------------------------------
if __name__ == "__main__":
    ipopt_executable = "/content/bin/ipopt"  # For Colab

    df_returns = calculate_monthly_returns(tickers_list, start, end)
    if df_returns is not None:
        df_results, df_allocations = optimize_and_plot_portfolio(
            df_returns, ipopt_executable
        )
        print("Functions called successfully and plots generated.")
    else:
        print("No valid return data available; skipping optimization.")

    # Full workflow example
    my_tickers = ['GE', 'KO', 'NVDA']
    my_start_date = '2020-01-01'
    my_end_date = '2024-01-01'

    final_df_results, final_df_allocations = perform_full_portfolio_analysis(
        my_tickers, my_start_date, my_end_date, ipopt_executable
    )

    if final_df_results is not None and final_df_allocations is not None:
        print("Final results and allocations obtained:")
        display(final_df_results.head())
        display(final_df_allocations.head())
    else:
        print("Portfolio analysis did not produce results.")
