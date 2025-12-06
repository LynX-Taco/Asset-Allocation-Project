# main.py (V2)

import argparse
from portfolio_pipeline.v2.portfolio_pipeline import run_portfolio_pipeline_v2

# Defaults (edit if you like)
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN"]
START = "2024-01-01"
END   = "2025-07-31"

def parse_args():
    p = argparse.ArgumentParser(
        description="Run V2 MINLP portfolio allocation (GLPK + backtest)."
    )
    p.add_argument(
        "--solver",
        default="glpk",
        help="Pyomo solver name (default: glpk)",
    )
    p.add_argument(
        "--start",
        default=START,
        help="Start date for return estimation (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end",
        default=END,
        help="End date for return estimation (YYYY-MM-DD)",
    )
    p.add_argument(
        "--tickers",
        nargs="*",
        default=TICKERS,
        help="Universe of tickers (space-separated)",
    )
    p.add_argument(
        "--min-alloc",
        type=float,
        default=0.02,
        help="Minimum allocation per selected asset (e.g., 0.02 = 2%)",
    )
    p.add_argument(
        "--max-alloc",
        type=float,
        default=0.20,
        help="Maximum allocation per selected asset (e.g., 0.20 = 20%)",
    )
    p.add_argument(
        "--choose-at-least",
        type=int,
        default=2,
        help="Cardinality: at least this many assets must be selected",
    )
    p.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip running the backtest step",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mret, allocations, backtest_results = run_portfolio_pipeline_v2(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        solver=args.solver,
        min_alloc=args.min_alloc,
        max_alloc=args.max_alloc,
        choose_at_least=args.choose_at_least,
        do_backtest=not args.no_backtest,
    )

    print("\n=== V2 RUN SUMMARY ===")
    if mret is not None:
        print("Monthly returns shape:", mret.shape)
    else:
        print("Monthly returns: None")

    if allocations is not None:
        print("\nAllocations (head):")
        print(allocations.head())
    else:
        print("\nAllocations: None (solver may have failed)")

    if backtest_results is not None:
        print("\nBacktest results (head):")
        print(backtest_results.head())
    else:
        print("\nBacktest: skipped or failed.")
