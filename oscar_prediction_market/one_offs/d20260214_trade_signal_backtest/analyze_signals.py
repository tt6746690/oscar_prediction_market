"""Analyze trade signal backtest results.

Reads backtest_results.json and produces:
- P&L curves over time (mark-to-market)
- Per-nominee P&L breakdown
- Independent vs. multi-outcome Kelly comparison
- Sensitivity analysis across kelly_fraction values
- "What if X wins" settlement table
- Edge accuracy analysis

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260214_trade_signal_backtest.analyze_signals \
        --results-dir storage/d20260214_trade_signal_backtest
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Loading
# ============================================================================


def load_results(results_dir: Path) -> dict:
    """Load backtest results JSON."""
    path = results_dir / "backtest_results.json"
    with open(path) as f:
        return json.load(f)


# ============================================================================
# P&L Analysis
# ============================================================================


def build_wealth_timeseries(results: dict) -> pd.DataFrame:
    """Build a DataFrame of wealth over time for all backtest runs.

    Returns DataFrame with columns: snapshot_date, run_key, model_type,
    bankroll_mode, cash, mtm_value, total_wealth, total_fees, total_trades
    """
    rows = []
    for run_key, run_data in results["backtests"].items():
        for snap in run_data["snapshots"]:
            rows.append(
                {
                    "snapshot_date": snap["snapshot_date"],
                    "run_key": run_key,
                    "model_type": snap["model_type"],
                    "bankroll_mode": snap["bankroll_mode"],
                    "cash": snap["cash"],
                    "mtm_value": snap["mark_to_market_value"],
                    "total_wealth": snap["total_wealth"],
                    "total_fees": snap["total_fees_paid"],
                    "total_trades": snap["total_trades"],
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def build_settlement_table(results: dict) -> pd.DataFrame:
    """Build a "what if X wins" settlement table.

    Returns DataFrame with columns: run_key, winner, final_cash, total_pnl, return_pct
    """
    rows = []
    for run_key, run_data in results["backtests"].items():
        for winner, settlement in run_data.get("settlements", {}).items():
            rows.append(
                {
                    "run_key": run_key,
                    "winner": winner,
                    "final_cash": settlement["final_cash"],
                    "total_pnl": settlement["total_pnl"],
                    "return_pct": settlement["return_pct"],
                }
            )
    return pd.DataFrame(rows)


def build_position_history(results: dict) -> pd.DataFrame:
    """Build position history across snapshots.

    Returns DataFrame with: snapshot_date, run_key, nominee, contracts, avg_cost
    """
    rows = []
    for run_key, run_data in results["backtests"].items():
        for snap in run_data["snapshots"]:
            for nominee, pos in snap.get("positions", {}).items():
                rows.append(
                    {
                        "snapshot_date": snap["snapshot_date"],
                        "run_key": run_key,
                        "nominee": nominee,
                        "contracts": pos["contracts"],
                        "avg_cost": pos["avg_cost"],
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


# ============================================================================
# Plotting
# ============================================================================


def plot_wealth_curves(wealth_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot total wealth over time for each run."""
    if wealth_df.empty:
        logger.warning("No wealth data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    initial_bankroll = wealth_df["total_wealth"].iloc[0] if not wealth_df.empty else 1000

    # By model type
    ax = axes[0]
    for run_key, group in wealth_df.groupby("run_key"):
        if "fixed" in str(run_key):
            ax.plot(
                group["snapshot_date"],
                group["total_wealth"],
                marker="o",
                label=str(run_key),
            )
    ax.axhline(y=initial_bankroll, color="gray", linestyle="--", alpha=0.5, label="Initial")
    ax.set_title("Fixed Bankroll — Wealth Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Wealth ($)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    # Dynamic bankroll
    ax = axes[1]
    for run_key, group in wealth_df.groupby("run_key"):
        if "dynamic" in str(run_key):
            ax.plot(
                group["snapshot_date"],
                group["total_wealth"],
                marker="o",
                label=str(run_key),
            )
    ax.axhline(y=initial_bankroll, color="gray", linestyle="--", alpha=0.5, label="Initial")
    ax.set_title("Dynamic Bankroll — Wealth Over Time")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = output_dir / "wealth_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_settlement_heatmap(settlement_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot what-if settlement as a heatmap."""
    if settlement_df.empty:
        logger.warning("No settlement data to plot")
        return

    # Filter to one bankroll mode for clarity
    fixed_df = settlement_df[settlement_df["run_key"].str.contains("fixed")]
    if fixed_df.empty:
        fixed_df = settlement_df

    pivot = fixed_df.pivot_table(
        index="winner", columns="run_key", values="return_pct", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_fixed", "") for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                color = "white" if abs(val) > 30 else "black"
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Hypothetical P&L if Nominee X Wins (% return)")
    plt.colorbar(im, ax=ax, label="Return (%)")
    plt.tight_layout()
    path = output_dir / "settlement_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_position_evolution(position_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot position sizes over time for each run."""
    if position_df.empty:
        logger.warning("No position data to plot")
        return

    run_keys = position_df["run_key"].unique()

    for run_key in run_keys:
        run_data = position_df[position_df["run_key"] == run_key]
        if run_data.empty:
            continue

        pivot = run_data.pivot_table(
            index="snapshot_date",
            columns="nominee",
            values="contracts",
            aggfunc="first",
            fill_value=0,
        )

        fig, ax = plt.subplots(figsize=(14, 6))
        pivot.plot.bar(stacked=True, ax=ax)
        ax.set_title(f"Position Evolution — {run_key}")
        ax.set_xlabel("Snapshot Date")
        ax.set_ylabel("Contracts")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()

        path = output_dir / f"positions_{run_key}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", path)


# ============================================================================
# Summary Printing
# ============================================================================


def print_summary(results: dict) -> None:
    """Print a concise summary of backtest results."""
    config = results.get("config", {})
    initial = config.get("bankroll_dollars", 1000)

    print("\n" + "=" * 80)
    print("TRADE SIGNAL BACKTEST SUMMARY")
    print("=" * 80)
    print(f"  Initial bankroll: ${initial:.0f}")
    print(f"  Kelly fraction: {config.get('kelly_fraction', 0.25):.0%}")
    print(f"  Min edge: {config.get('min_edge', 0.05):.0%}")
    print(f"  Spread penalty mode: {config.get('spread_penalty_mode', 'fixed')}")

    for run_key, run_data in results["backtests"].items():
        snapshots = run_data["snapshots"]
        if not snapshots:
            continue

        last = snapshots[-1]
        total_wealth = last["total_wealth"]
        pnl = total_wealth - initial
        n_trades = last["total_trades"]
        fees = last["total_fees_paid"]

        print(f"\n  --- {run_key} ---")
        print(f"  Final wealth: ${total_wealth:.2f}  (PnL: ${pnl:+.2f}, {pnl / initial:+.1%})")
        print(f"  Trades: {n_trades}  |  Fees: ${fees:.2f}")

        positions = last.get("positions", {})
        if positions:
            print("  Open positions:")
            for nominee, pos in positions.items():
                outlay = pos["contracts"] * pos["avg_cost"]
                print(
                    f"    {nominee}: {pos['contracts']} contracts "
                    f"@ ${pos['avg_cost']:.2f} (${outlay:.2f})"
                )

        # Settlement summary
        settlements = run_data.get("settlements", {})
        if settlements:
            best_winner = max(settlements.items(), key=lambda x: x[1]["total_pnl"])
            worst_winner = min(settlements.items(), key=lambda x: x[1]["total_pnl"])
            print("\n  What-if settlement:")
            print(
                f"    Best case:  {best_winner[0]} wins → "
                f"${best_winner[1]['total_pnl']:+.2f} ({best_winner[1]['return_pct']:+.1f}%)"
            )
            print(
                f"    Worst case: {worst_winner[0]} wins → "
                f"${worst_winner[1]['total_pnl']:+.2f} ({worst_winner[1]['return_pct']:+.1f}%)"
            )

    print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Analyze trade signal backtest results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing backtest_results.json",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    # Build analysis DataFrames
    wealth_df = build_wealth_timeseries(results)
    settlement_df = build_settlement_table(results)
    position_df = build_position_history(results)

    # Print summary
    print_summary(results)

    # Generate plots
    plot_wealth_curves(wealth_df, results_dir)
    plot_settlement_heatmap(settlement_df, results_dir)
    plot_position_evolution(position_df, results_dir)

    # Export analysis CSVs
    if not wealth_df.empty:
        wealth_df.to_csv(results_dir / "wealth_timeseries.csv", index=False)
    if not settlement_df.empty:
        settlement_df.to_csv(results_dir / "settlement_scenarios.csv", index=False)
    if not position_df.empty:
        position_df.to_csv(results_dir / "position_history.csv", index=False)

    logger.info("Analysis complete. Outputs in %s", results_dir)


if __name__ == "__main__":
    main()
