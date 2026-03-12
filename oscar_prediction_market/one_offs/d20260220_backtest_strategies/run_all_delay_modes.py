"""Run all 3 signal-delay modes and save results.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python oscar_prediction_market/one_offs/\
d20260220_backtest_strategies/run_all_delay_modes.py

    # Fast mode (~20s total):
    uv run python oscar_prediction_market/one_offs/\
d20260220_backtest_strategies/run_all_delay_modes.py --fast
"""

import argparse
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path("storage/d20260220_backtest_strategies/2025")
MODULE = "oscar_prediction_market.one_offs.d20260220_backtest_strategies.run_backtests"

MODES = [
    ("delay_0", ["--signal-delay-days", "0"]),
    ("delay_1", ["--signal-delay-days", "1"]),
    ("inferred_6h", ["--inferred-lag-hours", "6"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Pass --fast to each mode")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Include equal-weight ensemble model (avg of all 4 models)",
    )
    args = parser.parse_args()

    for name, mode_args in MODES:
        result_dir = str(EXP_DIR / f"results_{name}")
        cmd = [sys.executable, "-m", MODULE, "--results-dir", result_dir] + mode_args
        if args.fast:
            cmd.append("--fast")
        if args.ensemble:
            cmd.append("--ensemble")
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"  Results: {result_dir}")
        print(f"  Args: {mode_args}")
        if args.fast:
            print("  Mode: FAST")
        print(f"{'=' * 70}", flush=True)

        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(f"ERROR: {name} failed with code {proc.returncode}")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("All 3 modes complete!")
    print("=" * 70)
    for name, _ in MODES:
        print(f"  {EXP_DIR / f'results_{name}'}/")


if __name__ == "__main__":
    main()
