#!/usr/bin/env python3
"""
main.py — top-level dispatcher for V1 / V2 pipelines

Usage examples:
  # run V1 (original/basic pipeline)
  python main.py --version v1 --tickers AAPL MSFT NVDA --start 2022-01-01 --end 2024-01-01

  # run V2 (enhanced pipeline — uses Bonmin if available)
  python main.py --version v2 --tickers AAPL MSFT NVDA --start 2024-01-01 --end 2025-07-31 --bonmin-bin /content/idaes_bin/bonmin

Notes:
- This script does NOT hard-import the pipeline modules; it calls the runner scripts
  as subprocesses to avoid import-time side effects and to ensure the env is clean.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent

def find_runner_v1() -> Path | None:
    """Find the v1 runner script in likely locations."""
    candidates = [
        REPO_ROOT / "portfolio_pipeline" / "runner.py",
        REPO_ROOT / "portfolio_pipeline" / "runner_v1.py",
        REPO_ROOT / "src" / "portfolio_pipeline.py",  # fallback if v1 implemented as module
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def find_runner_v2() -> Path | None:
    """Find the v2 runner script."""
    candidates = [
        REPO_ROOT / "portfolio_pipeline" / "v2" / "runner_v2.py",
        REPO_ROOT / "portfolio_pipeline" / "v2" / "runner_bonmin.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def build_basic_args(args: argparse.Namespace) -> List[str]:
    """Build commonly forwarded args for child runners."""
    cmd_args: List[str] = []
    if args.solver:
        cmd_args += ["--solver", args.solver]
    if args.tickers:
        cmd_args += ["--tickers"] + list(args.tickers)
    if args.start:
        cmd_args += ["--start", args.start]
    if args.end:
        cmd_args += ["--end", args.end]
    if args.config:
        cmd_args += ["--config", args.config]
    if args.min_alloc is not None:
        cmd_args += ["--min-alloc", str(args.min_alloc)]
    if args.max_alloc is not None:
        cmd_args += ["--max-alloc", str(args.max_alloc)]
    if args.choose_at_least is not None:
        cmd_args += ["--choose-at-least", str(args.choose_at_least)]
    return cmd_args

def run_subprocess(cmd: List[str], env: dict | None = None, cwd: Path | None = None) -> int:
    """Run a subprocess command, streaming stdout/stderr to the current process."""
    print("Running:", " ".join(shlex_quote_list(cmd)))
    use_env = os.environ.copy()
    if env:
        use_env.update(env)
    proc = subprocess.Popen(cmd, env=use_env, cwd=(str(cwd) if cwd else None))
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        return 130

def shlex_quote_list(lst: List[str]) -> List[str]:
    """Return a shell-quoted representation for printing (not used for execution)."""
    import shlex
    return [shlex.quote(x) for x in lst]

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top-level runner (v1 | v2) for Asset-Allocation-Project")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Which pipeline version to run")
    parser.add_argument("--solver", default=None, help="Solver to use (glpk, ipopt, etc.) for downstream runner")
    parser.add_argument("--bonmin-bin", default=None, help="Path to bonmin executable (if using v2 bonmin)")
    parser.add_argument("--tickers", nargs="+", default=None, help="List of ticker symbols")
    parser.add_argument("--start", default=None, help="Start date for price history (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date for price history (YYYY-MM-DD)")
    parser.add_argument("--config", default=None, help="Optional config file for pipeline (v2 configs/...)")
    parser.add_argument("--min-alloc", type=float, default=None, help="Minimum allocation bound (forwarded)")
    parser.add_argument("--max-alloc", type=float, default=None, help="Maximum allocation bound (forwarded)")
    parser.add_argument("--choose-at-least", type=int, default=None, help="Cardinality (forwarded)")
    parser.add_argument("--cwd", default=str(REPO_ROOT), help="Working directory to run runners from (default repo root)")
    parsed = parser.parse_args(argv)

    os.chdir(parsed.cwd)
    print(f"Working directory: {Path.cwd()}")

    forwarded = build_basic_args(parsed)

    if parsed.version == "v1":
        runner = find_runner_v1()
        if not runner:
            print("ERROR: could not find v1 runner script (looked in portfolio_pipeline/runner.py and src/).", file=sys.stderr)
            return 2
        # If the runner is a python module file, call via python <path>
        cmd = [sys.executable, str(runner)] + forwarded
        return run_subprocess(cmd, cwd=REPO_ROOT)

    # version == v2
    runner = find_runner_v2()
    if not runner:
        print("ERROR: could not find v2 runner (portfolio_pipeline/v2/runner_v2.py or runner_bonmin.py).", file=sys.stderr)
        return 3

    # prepare env: set BONMIN_BIN if provided and ensure PYTHONPATH includes repo
    env = {}
    if parsed.bonmin_bin:
        env["BONMIN_BIN"] = parsed.bonmin_bin
        print("Setting BONMIN_BIN to", parsed.bonmin_bin)
    # ensure repo root is in PYTHONPATH so runners can import package modules
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    repo_str = str(REPO_ROOT)
    if repo_str not in existing_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = repo_str + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    else:
        env["PYTHONPATH"] = existing_pythonpath

    # construct command
    cmd = [sys.executable, str(runner)] + forwarded
    return run_subprocess(cmd, env=env, cwd=REPO_ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
