"""
Mean-variance portfolio optimization pipeline using Pyomo and IPOPT.

Features
- Downloads daily prices from Yahoo Finance for user-provided tickers.
- Converts prices into monthly compounded returns.
- Builds a long-only, fully-invested optimization model that maximizes expected return
  while sweeping a variance cap to trace the efficient frontier.
- Saves efficient frontier and allocation plots to disk for easy inspection.

Usage (example)
python main.py --tickers "AAPL MSFT NVDA" --start-date 2020-01-01 --end-date 2024-01-01 \
  --ipopt-path ./bin/ipopt --n-points 200 --output-dir outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pyomo.environ import ConcreteModel, Constraint, NonNegativeReals, Objective, Param, Set, Var, maximize
from pyomo.opt import SolverFactory, TerminationCondition

# Use a non-interactive backend so plots render in headless environments.
matplotlib.use("Agg")


def calculate_monthly_returns(tickers: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily prices and convert to monthly compounded returns.

    Args:
        tickers: Iterable of ticker symbols.
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).

    Returns:
        DataFrame of monthly returns indexed by month-end with tickers as columns.
    """
    tickers_list = list(tickers)
    if not tickers_list:
        raise ValueError("No tickers provided.")

    price_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers_list:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker} in the specified date range.")
        price_frames[ticker] = data

    return_data: dict[str, pd.Series] = {}
    for ticker, data in price_frames.items():
        returns = data["Close"].pct_change().dropna()
        if len(returns) < 2:
            raise ValueError(f"Insufficient data to compute returns for ticker {ticker}.")
        return_data[ticker] = returns

    daily_returns = pd.concat(return_data.values(), axis=1, keys=return_data.keys())
    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    return monthly_returns.dropna()


def _build_solver(ipopt_path: Optional[str]):
    solver = SolverFactory("ipopt")
    if solver.available():
        return solver

    if ipopt_path:
        solver = SolverFactory("ipopt", executable=ipopt_path)
        if solver.available():
            return solver

    raise RuntimeError("IPOPT solver is not available. Provide a valid --ipopt-path from idaes get-extensions.")


def optimize_and_plot_portfolio(
    df_returns: pd.DataFrame,
    ipopt_path: Optional[str],
    n_points: int = 200,
    output_dir: str | Path = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize allocations across risk levels and save plots.

    Args:
        df_returns: Monthly returns DataFrame indexed by period with tickers as columns.
        ipopt_path: Optional path to the IPOPT executable (e.g., ./bin/ipopt or /content/bin/ipopt).
        n_points: Number of risk levels to sweep when tracing the efficient frontier.
        output_dir: Directory to store the generated figures.

    Returns:
        Tuple of (frontier DataFrame, allocation DataFrame).
    """
    if df_returns.empty:
        raise ValueError("Monthly returns DataFrame is empty.")

    solver = _build_solver(ipopt_path)

    assets: List[str] = df_returns.columns.tolist()
    model = ConcreteModel()
    model.Assets = Set(initialize=assets)
    model.x = Var(model.Assets, within=NonNegativeReals, bounds=(0, 1))

    avg_returns = df_returns.mean().to_dict()
    model.mu = Param(model.Assets, initialize=avg_returns)

    cov_df = df_returns.cov()
    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    model.Sigma = Param(model.Assets, model.Assets, initialize=cov_dict)

    def total_return_rule(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)

    model.objective = Objective(rule=total_return_rule, sense=maximize)

    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1

    model.budget = Constraint(rule=budget_constraint_rule)

    max_possible_variance = float(np.max(np.diag(cov_df.values)))
    max_risk = max_possible_variance * 1.5
    min_risk = 1e-6
    risk_limits = np.linspace(min_risk, max_risk, num=n_points)

    allocations: dict[float, List[float]] = {}
    returns: dict[float, float] = {}

    for risk_cap in risk_limits:
        if hasattr(model, "variance_constraint"):
            model.del_component(model.variance_constraint)

        def variance_constraint_rule(m):
            return sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets) <= risk_cap

        model.variance_constraint = Constraint(rule=variance_constraint_rule)

        result = solver.solve(model, tee=False)
        termination = result.solver.termination_condition

        if termination in {TerminationCondition.infeasible, TerminationCondition.other}:
            continue

        if termination in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
            allocations[risk_cap] = [model.x[a]() for a in model.Assets]
            returns[risk_cap] = model.objective()

    if not allocations:
        raise RuntimeError("Optimization did not produce any feasible solutions. Check the solver path and data.")

    df_frontier = pd.DataFrame({"Risk": list(returns.keys()), "Return": list(returns.values())}).sort_values("Risk")
    df_allocations = pd.DataFrame(allocations).T
    df_allocations.columns = assets
    df_allocations.index.name = "Risk"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df_frontier["Risk"], df_frontier["Return"], marker="o", linestyle="-")
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    frontier_file = output_path / "efficient_frontier.png"
    plt.savefig(frontier_file, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(df_allocations.index, df_allocations[asset], marker="o", markersize=3, label=asset)
    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    allocation_file = output_path / "allocation_vs_risk.png"
    plt.savefig(allocation_file, bbox_inches="tight")
    plt.close()

    return df_frontier, df_allocations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean-variance portfolio optimization pipeline.")
    parser.add_argument("--tickers", type=str, required=True, help="Space-separated list of tickers (e.g., 'GE KO NVDA')")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--n-points", type=int, default=200, help="Number of risk levels to evaluate (default: 200)")
    parser.add_argument(
        "--ipopt-path",
        type=str,
        default=None,
        help="Path to IPOPT executable installed via 'idaes get-extensions' (e.g., ./bin/ipopt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to write plots (default: outputs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = args.tickers.split()
    df_returns = calculate_monthly_returns(tickers, args.start_date, args.end_date)
    df_frontier, df_allocations = optimize_and_plot_portfolio(
        df_returns=df_returns,
        ipopt_path=args.ipopt_path,
        n_points=args.n_points,
        output_dir=args.output_dir,
    )

    print("Optimization complete. Saved plots to:")
    print(f"  {Path(args.output_dir).resolve() / 'efficient_frontier.png'}")
    print(f"  {Path(args.output_dir).resolve() / 'allocation_vs_risk.png'}")
    print("Frontier preview:")
    print(df_frontier.head())
    print("\nAllocation preview:")
    print(df_allocations.head())


if __name__ == "__main__":
    main()
