"""Cross-year analysis plots for buy-and-hold backtest.

Generates plots comparing 2024 and 2025 backtest results side-by-side:

1. Config profitability scatter (2024 P&L vs 2025 P&L, colored by model)
2. Cross-year rank correlation (per-model rank scatter plots, by P&L)
3. Category edge comparison (grouped bars: category × year)

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.analyze_cross_year_plots
"""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from oscar_prediction_market.one_offs.analysis_utils.style import (
    apply_style,
    get_model_color,
    get_model_display,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    BUY_HOLD_EXP_DIR,
    YEAR_CONFIGS,
)

apply_style()

PLOTS_DIR = BUY_HOLD_EXP_DIR / "cross_year_plots"
BANKROLL = 1000.0

CONFIG_PARAMS = [
    "model_type",
    "config_label",
    "fee_type",
    "kelly_fraction",
    "buy_edge_threshold",
    "kelly_mode",
    "bankroll_mode",
    "allowed_directions",
]

CATEGORY_DISPLAY = {
    "best_picture": "Best Picture",
    "directing": "Directing",
    "actor_leading": "Lead Actor",
    "actress_leading": "Lead Actress",
    "actor_supporting": "Supp. Actor",
    "actress_supporting": "Supp. Actress",
    "original_screenplay": "Orig. Screenplay",
    "animated_feature": "Animated Feature",
    "cinematography": "Cinematography",
}

MODEL_ORDER = ["clogit", "lr", "gbt", "cal_sgbt", "avg_ens", "clog_sgbt"]


def _save_fig(fig: matplotlib.figure.Figure, filename: str, *, dpi: int = 150) -> None:
    """Save figure to PLOTS_DIR."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def _portfolio_pnl(agg: pd.DataFrame) -> pd.DataFrame:
    """Sum total_pnl across categories per config to get portfolio P&L."""
    group_cols = [c for c in CONFIG_PARAMS if c in agg.columns]
    sum_cols = ["total_pnl", "total_fees", "total_trades"]
    if "capital_deployed" in agg.columns:
        sum_cols.append("capital_deployed")
    portfolio = agg.groupby(group_cols, as_index=False)[sum_cols].sum()
    portfolio = portfolio.rename(columns={"total_pnl": "portfolio_pnl"})
    return portfolio


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aggregate P&L for both years.

    Returns:
        (agg_2024, agg_2025) — filtered to fixed bankroll.
    """
    agg_2024 = pd.read_csv(YEAR_CONFIGS[2024].results_dir / "aggregate_pnl.csv")
    agg_2024 = agg_2024[agg_2024["bankroll_mode"] == "fixed"]

    agg_2025 = pd.read_csv(YEAR_CONFIGS[2025].results_dir / "aggregate_pnl.csv")
    agg_2025 = agg_2025[agg_2025["bankroll_mode"] == "fixed"]

    return agg_2024, agg_2025


def plot_config_profitability_scatter(port_2024: pd.DataFrame, port_2025: pd.DataFrame) -> None:
    """Plot 1: Scatter of 2024 P&L vs 2025 P&L for each model×config, colored by model.

    Each dot is one config. Quadrants show:
    - Q1 (top-right): profitable in both years
    - Q2 (top-left): 2025-only profitable
    - Q3 (bottom-left): neither profitable
    - Q4 (bottom-right): 2024-only profitable
    """
    merge_cols = [c for c in CONFIG_PARAMS if c in port_2024.columns and c in port_2025.columns]
    merged = port_2024.merge(port_2025, on=merge_cols, suffixes=("_2024", "_2025"))

    fig, ax = plt.subplots(figsize=(10, 8))

    for model in MODEL_ORDER:
        subset = merged[merged["model_type"] == model]
        if subset.empty:
            continue
        ax.scatter(
            subset["portfolio_pnl_2024"],
            subset["portfolio_pnl_2025"],
            c=get_model_color(model),
            label=get_model_display(model),
            alpha=0.4,
            s=15,
            edgecolors="none",
        )

    # Quadrant lines
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    label_kwargs: dict[str, Any] = {
        "fontsize": 9,
        "alpha": 0.5,
        "ha": "center",
        "va": "center",
        "style": "italic",
    }
    ax.text(xlim[1] * 0.6, ylim[1] * 0.9, "Both profitable", **label_kwargs)
    ax.text(xlim[0] * 0.4, ylim[1] * 0.9, "2025 only", **label_kwargs)

    ax.set_xlabel("2024 Portfolio P&L ($)")
    ax.set_ylabel("2025 Portfolio P&L ($)")
    ax.set_title("Cross-Year Config Profitability: 2024 vs 2025 P&L")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # Annotate counts
    both = ((merged["portfolio_pnl_2024"] > 0) & (merged["portfolio_pnl_2025"] > 0)).sum()
    total = len(merged)
    ax.annotate(
        f"{both}/{total} ({100 * both / total:.1f}%) profitable in both years",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
    )

    fig.tight_layout()
    _save_fig(fig, "cross_year_profitability_scatter.png")


def plot_rank_correlation(port_2024: pd.DataFrame, port_2025: pd.DataFrame) -> None:
    """Plot 2: Per-model scatter of 2024 P&L rank vs 2025 P&L rank.

    A 2x3 grid of subplots (one per model). Ranks are computed from portfolio
    P&L (higher P&L = lower rank number = better).
    """
    merge_cols = [c for c in CONFIG_PARAMS if c in port_2024.columns and c in port_2025.columns]
    merged = port_2024.merge(port_2025, on=merge_cols, suffixes=("_2024", "_2025"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()

    for i, model in enumerate(MODEL_ORDER):
        ax = axes_flat[i]
        subset = merged[merged["model_type"] == model].copy()
        if subset.empty:
            continue

        # Compute P&L ranks within this model (1 = best)
        subset["rank_2024"] = subset["portfolio_pnl_2024"].rank(ascending=False).astype(int)
        subset["rank_2025"] = subset["portfolio_pnl_2025"].rank(ascending=False).astype(int)

        color = get_model_color(model)
        ax.scatter(
            subset["rank_2024"],
            subset["rank_2025"],
            c=color,
            alpha=0.3,
            s=10,
            edgecolors="none",
        )

        # Diagonal reference line
        max_rank = max(subset["rank_2024"].max(), subset["rank_2025"].max())
        ax.plot([1, max_rank], [1, max_rank], "k--", alpha=0.3, linewidth=0.8)

        # Spearman rho
        rho, pval = stats.spearmanr(subset["rank_2024"], subset["rank_2025"])
        ax.set_title(f"{get_model_display(model)}\nρ = {rho:.3f}", fontsize=10)
        ax.set_xlabel("2024 P&L Rank")
        ax.set_ylabel("2025 P&L Rank")

        # Invert axes so rank 1 is top-left
        ax.invert_xaxis()
        ax.invert_yaxis()

    fig.suptitle("Cross-Year P&L Rank Correlation by Model", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "cross_year_rank_correlation.png")


def plot_category_edge_comparison(agg_2024: pd.DataFrame, agg_2025: pd.DataFrame) -> None:
    """Plot 3: Grouped bar chart showing total P&L by category for each year.

    Highlights the anti-correlation: 2024 profits from Actress Leading / Original
    Screenplay, while 2025 profits from Directing / Best Picture.
    """
    # Sum P&L across all configs per category
    cat_2024 = agg_2024.groupby("category")["total_pnl"].sum().reset_index()
    cat_2024["year"] = "2024"
    cat_2025 = agg_2025.groupby("category")["total_pnl"].sum().reset_index()
    cat_2025["year"] = "2025"

    combined = pd.concat([cat_2024, cat_2025])

    # Order categories by total |P&L| descending
    cat_order = (
        combined.groupby("category")["total_pnl"]
        .apply(lambda x: x.abs().sum())
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cat_order))
    width = 0.35

    vals_2024 = []
    vals_2025 = []
    for cat in cat_order:
        v24 = cat_2024.loc[cat_2024["category"] == cat, "total_pnl"]
        vals_2024.append(v24.iloc[0] if len(v24) > 0 else 0)
        v25 = cat_2025.loc[cat_2025["category"] == cat, "total_pnl"]
        vals_2025.append(v25.iloc[0] if len(v25) > 0 else 0)

    ax.bar(
        x - width / 2,
        [v / 1000 for v in vals_2024],
        width,
        label="2024",
        color="#3274a1",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        [v / 1000 for v in vals_2025],
        width,
        label="2025",
        color="#e1812c",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c) for c in cat_order], rotation=35, ha="right")
    ax.set_ylabel("Total P&L ($K, sum across all 3,528 configs)")
    ax.set_title("Category Edge is Anti-Correlated Across Years")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save_fig(fig, "cross_year_category_edge.png")


def main() -> None:
    """Generate all cross-year analysis plots."""
    print("=" * 70)
    print("Cross-Year Analysis Plots")
    print(f"Output: {PLOTS_DIR}")
    print("=" * 70)

    agg_2024, agg_2025 = load_data()
    port_2024 = _portfolio_pnl(agg_2024)
    port_2025 = _portfolio_pnl(agg_2025)

    print("\n--- 1. Config Profitability Scatter ---")
    plot_config_profitability_scatter(port_2024, port_2025)

    print("\n--- 2. Cross-Year Rank Correlation ---")
    plot_rank_correlation(port_2024, port_2025)

    print("\n--- 3. Category Edge Comparison ---")
    plot_category_edge_comparison(agg_2024, agg_2025)

    n_plots = len(list(PLOTS_DIR.glob("*.png")))
    print(f"\nDone! {n_plots} plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
