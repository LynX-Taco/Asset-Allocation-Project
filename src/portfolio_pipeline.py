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
        max_risk_for_range +_
