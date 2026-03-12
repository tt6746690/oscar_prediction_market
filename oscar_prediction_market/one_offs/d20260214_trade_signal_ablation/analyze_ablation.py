"""Analyze trade signal ablation results.

Reads ablation_results.json and produces:
- Parameter sensitivity analysis (mean return by each parameter level)
- Best/worst configs table
- Heatmaps of return by parameter pairs
- Marginal effect of each parameter
- Settlement analysis under different winners

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260214_trade_signal_ablation.analyze_ablation \
        --results-dir storage/d20260214_trade_signal_ablation/results \
        --output-dir storage/d20260214_trade_signal_ablation
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Loading
# ============================================================================


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load ablation results into a DataFrame."""
    path = results_dir / "ablation_summary.csv"
    if path.exists():
        return pd.read_csv(path)

    # Fallback: load from JSON
    json_path = results_dir / "ablation_results.json"
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        rows.append(
            {
                "config_id": r["config_id"],
                "model_type": r["model_type"],
                "fee_type": r["fee_type"],
                "kelly_fraction": r["kelly_fraction"],
                "min_edge": r["min_edge"],
                "sell_edge_threshold": r["sell_edge_threshold"],
                "min_price": r["min_price"],
                "kelly_mode": r.get("kelly_mode", "multi_outcome"),
                "bankroll_mode": r["bankroll_mode"],
                "final_wealth": r["final_wealth"],
                "total_return_pct": r["total_return_pct"],
                "total_fees_paid": r["total_fees_paid"],
                "total_trades": r["total_trades"],
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Parameter Sensitivity
# ============================================================================


def parameter_sensitivity(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute mean return for each level of each parameter.

    Returns dict of {param_name: DataFrame with level, mean_return, median, count}.
    """
    params = [
        "model_type",
        "fee_type",
        "kelly_fraction",
        "min_edge",
        "sell_edge_threshold",
        "min_price",
        "kelly_mode",
    ]
    results: dict[str, pd.DataFrame] = {}

    for param in params:
        grouped = (
            df.groupby(param)
            .agg(
                mean_return=("total_return_pct", "mean"),
                median_return=("total_return_pct", "median"),
                std_return=("total_return_pct", "std"),
                mean_fees=("total_fees_paid", "mean"),
                mean_trades=("total_trades", "mean"),
                profitable=("total_return_pct", lambda x: (x > 0).sum()),
                count=("total_return_pct", "count"),
            )
            .reset_index()
        )
        results[param] = grouped

    return results


def print_sensitivity(sensitivity: dict[str, pd.DataFrame]) -> None:
    """Print parameter sensitivity tables."""
    print(f"\n{'=' * 80}")
    print("Parameter Sensitivity Analysis")
    print(f"{'=' * 80}")

    for param, df in sensitivity.items():
        print(f"\n--- {param} ---")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


# ============================================================================
# Interaction Analysis
# ============================================================================


def top_configs(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N configs by return."""
    cols = [
        "config_id",
        "model_type",
        "fee_type",
        "kelly_fraction",
        "min_edge",
        "sell_edge_threshold",
        "min_price",
        "kelly_mode",
        "total_return_pct",
        "total_fees_paid",
        "total_trades",
    ]
    return df.nlargest(n, "total_return_pct")[cols]


def interaction_heatmap(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = "total_return_pct",
) -> pd.DataFrame:
    """Compute mean metric for each combination of two parameters."""
    pivot = df.pivot_table(
        index=param1,
        columns=param2,
        values=metric,
        aggfunc="mean",
    )
    return pivot


# ============================================================================
# Plotting
# ============================================================================


def plot_parameter_bars(
    sensitivity: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Bar charts of mean return by parameter level."""
    n_params = len(sensitivity)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (param, df) in enumerate(sensitivity.items()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        x = [str(v) for v in df[param]]
        y = df["mean_return"]
        colors = ["green" if v > 0 else "red" for v in y]
        ax.bar(x, y, color=colors, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Mean Return by {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Mean Return (%)")
        ax.tick_params(axis="x", rotation=45)

    # Hide unused axes
    for idx in range(len(sensitivity), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = output_dir / "parameter_sensitivity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_interaction_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmaps of key parameter interactions."""
    interactions = [
        ("model_type", "fee_type"),
        ("kelly_fraction", "min_edge"),
        ("min_edge", "min_price"),
        ("sell_edge_threshold", "min_edge"),
        ("model_type", "kelly_mode"),
        ("kelly_mode", "fee_type"),
    ]

    n_interactions = len(interactions)
    n_cols = 3
    n_rows = (n_interactions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 6 * n_rows))

    for idx, (p1, p2) in enumerate(interactions):
        ax = axes[idx // n_cols][idx % n_cols]
        pivot = interaction_heatmap(df, p1, p2)

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"Mean Return: {p1} × {p2}")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 20 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = output_dir / "interaction_heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_return_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of returns across all configs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax = axes[0]
    ax.hist(df["total_return_pct"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="breakeven")
    ax.axvline(
        x=df["total_return_pct"].median(),
        color="orange",
        linestyle="-",
        alpha=0.7,
        label=f"median={df['total_return_pct'].median():.1f}%",
    )
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Count")
    ax.set_title("Return Distribution Across All Configs")
    ax.legend()

    # By model type
    ax = axes[1]
    for mt in df["model_type"].unique():
        subset = df[df["model_type"] == mt]
        ax.hist(subset["total_return_pct"], bins=30, alpha=0.5, label=mt, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Count")
    ax.set_title("Return Distribution by Model Type")
    ax.legend()

    plt.tight_layout()
    path = output_dir / "return_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_fees_vs_return(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of fees paid vs return, colored by fee_type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for fee_type in df["fee_type"].unique():
        subset = df[df["fee_type"] == fee_type]
        ax.scatter(
            subset["total_fees_paid"],
            subset["total_return_pct"],
            alpha=0.3,
            label=fee_type,
            s=20,
        )

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Total Fees Paid ($)")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Fees vs Return (each dot = one config)")
    ax.legend()

    plt.tight_layout()
    path = output_dir / "fees_vs_return.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


# ============================================================================
# Settlement Analysis
# ============================================================================


def settlement_analysis(results_dir: Path, output_dir: Path) -> None:
    """Analyze hypothetical settlements for top configs."""
    json_path = results_dir / "ablation_results.json"
    if not json_path.exists():
        logger.warning("No ablation_results.json found, skipping settlement analysis")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Find top 10 configs by return
    sorted_results = sorted(data["results"], key=lambda x: -x["total_return_pct"])
    top_results = sorted_results[:10]

    print(f"\n{'=' * 80}")
    print("Settlement Analysis (top 10 configs)")
    print(f"{'=' * 80}")

    for r in top_results:
        if not r.get("settlements"):
            continue
        print(f"\n  {r['config_id']} (MtM return: {r['total_return_pct']:+.1f}%)")
        for winner, s in sorted(r["settlements"].items(), key=lambda x: -x[1]["return_pct"]):
            if abs(s["return_pct"]) > 1:  # Only show non-trivial settlements
                print(f"    if {winner:30s} wins: {s['return_pct']:+.1f}%")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    df = load_results(results_dir)
    logger.info("Loaded %d results", len(df))

    if df.empty:
        logger.error("No results to analyze")
        return

    # Parameter sensitivity
    sensitivity = parameter_sensitivity(df)
    print_sensitivity(sensitivity)

    # Top configs
    print(f"\n{'=' * 80}")
    print("Top 20 Configs by Return")
    print(f"{'=' * 80}")
    print(top_configs(df, 20).to_string(index=False))

    # Profitable config analysis
    profitable = df[df["total_return_pct"] > 0]
    print(f"\n{'=' * 80}")
    print(
        f"Profitable Configs: {len(profitable)}/{len(df)} ({100 * len(profitable) / len(df):.0f}%)"
    )
    print(f"{'=' * 80}")
    if not profitable.empty:
        print("\nProfile of profitable configs (mean values):")
        for param in [
            "model_type",
            "fee_type",
            "kelly_fraction",
            "min_edge",
            "sell_edge_threshold",
            "min_price",
        ]:
            print(f"  {param}: {profitable[param].value_counts().to_dict()}")

    # Plots
    plot_parameter_bars(sensitivity, output_dir)
    plot_interaction_heatmaps(df, output_dir)
    plot_return_distribution(df, output_dir)
    plot_fees_vs_return(df, output_dir)

    # Settlement analysis
    settlement_analysis(results_dir, output_dir)

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
