"""Visualization script for d0305 config selection sweep results.

Generates plots summarizing the targeted 27-config × 6-model sweep,
covering model comparison, config sensitivity, EV reliability, and
risk-return tradeoffs. Outputs to storage/d20260305_config_selection_sweep/plots/.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.d20260305_config_selection_sweep.plot_results
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.analysis_utils.style import (
    apply_style,
    get_model_color,
    get_model_display,
)

apply_style()

EXP_DIR = Path("storage/d20260305_config_selection_sweep")
PLOTS_DIR = EXP_DIR / "plots"

# Canonical model ordering (best to worst by combined P&L)
MODEL_ORDER = ["avg_ensemble", "cal_sgbt", "clogit_cal_sgbt_ensemble", "clogit", "lr", "gbt"]


def _save_fig(fig: matplotlib.figure.Figure, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _load_cross_year() -> pd.DataFrame:
    df = pd.read_csv(EXP_DIR / "cross_year_scenario_scores.csv")
    # Normalize total_pnl -> actual_pnl for consistency
    renames = {}
    for col in list(df.columns):
        if "total_pnl" in col and col.replace("total_pnl", "actual_pnl") not in df.columns:
            renames[col] = col.replace("total_pnl", "actual_pnl")
    if renames:
        df = df.rename(columns=renames)
    return df


def _short(model: str) -> str:
    return get_model_display(model)


# ============================================================================
# Plot 1: Model Tier Overview — Combined P&L by model (box + strip)
# ============================================================================


def plot_model_pnl_overview(df: pd.DataFrame) -> None:
    """Box plot of combined P&L by model, showing all 27 configs per model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]
    positions = range(len(models))

    for i, model in enumerate(models):
        vals = df[df["model_type"] == model]["avg_actual_pnl"].to_numpy() * 2  # combined = avg * 2
        color = get_model_color(model)
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.3, "edgecolor": color},
            whiskerprops={"color": color},
            capprops={"color": color},
            medianprops={"color": color, "linewidth": 2},
            flierprops={"markeredgecolor": color},
        )
        # Overlay individual points
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            [i] * len(vals) + jitter,
            vals,
            c=color,
            alpha=0.5,
            s=20,
            zorder=3,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([_short(m) for m in models], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Combined P&L (2024 + 2025, $)")
    ax.set_title("Model Tier Overview: Combined P&L Across All 27 Configs")
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, PLOTS_DIR / "model_pnl_overview.png")


# ============================================================================
# Plot 2: Edge Threshold Sensitivity by Model
# ============================================================================


def plot_edge_sensitivity(df: pd.DataFrame) -> None:
    """Line plot: combined P&L vs edge_threshold, one line per model, averaged over KF."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]

    for model in models:
        m = df[df["model_type"] == model]
        grouped = m.groupby("buy_edge_threshold")["avg_actual_pnl"].mean() * 2
        ax.plot(
            grouped.index,
            grouped.to_numpy(),
            marker="o",
            color=get_model_color(model),
            label=_short(model),
            linewidth=2,
            markersize=5,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Edge Threshold")
    ax.set_ylabel("Mean Combined P&L ($)")
    ax.set_title("Edge Threshold Sensitivity (Averaged Over Kelly Fractions)")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.3)

    _save_fig(fig, PLOTS_DIR / "edge_sensitivity.png")


# ============================================================================
# Plot 3: Kelly Fraction Sensitivity by Model
# ============================================================================


def plot_kelly_sensitivity(df: pd.DataFrame) -> None:
    """Line plot: combined P&L vs kelly_fraction, one line per model, averaged over edge."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]

    for model in models:
        m = df[df["model_type"] == model]
        grouped = m.groupby("kelly_fraction")["avg_actual_pnl"].mean() * 2
        ax.plot(
            grouped.index,
            grouped.to_numpy(),
            marker="s",
            color=get_model_color(model),
            label=_short(model),
            linewidth=2,
            markersize=6,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Kelly Fraction")
    ax.set_ylabel("Mean Combined P&L ($)")
    ax.set_title("Kelly Fraction Sensitivity (Averaged Over Edge Thresholds)")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.set_xticks([0.05, 0.15, 0.25])
    ax.grid(alpha=0.3)

    _save_fig(fig, PLOTS_DIR / "kelly_sensitivity.png")


# ============================================================================
# Plot 4: 2024 vs 2025 P&L Scatter (per-config, colored by model)
# ============================================================================


def plot_cross_year_scatter(df: pd.DataFrame) -> None:
    """Scatter plot of 2024 vs 2025 P&L. Each point is one model+config."""
    fig, ax = plt.subplots(figsize=(8, 7))

    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]

    for model in models:
        m = df[df["model_type"] == model]
        ax.scatter(
            m["actual_pnl_2024"],
            m["actual_pnl_2025"],
            c=get_model_color(model),
            label=_short(model),
            alpha=0.6,
            s=40,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("2024 P&L ($)")
    ax.set_ylabel("2025 P&L ($)")
    ax.set_title("Cross-Year P&L: Each Point = One Model + Config")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.3)

    # Shade profitable-in-both quadrant
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(
        [0, max(xlim[1], 1)],
        0,
        max(ylim[1], 1),
        alpha=0.05,
        color="green",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    _save_fig(fig, PLOTS_DIR / "cross_year_scatter.png")


# ============================================================================
# Plot 5: EV vs Actual PnL (per model, showing the anti-correlation)
# ============================================================================


def plot_ev_vs_actual(df: pd.DataFrame) -> None:
    """Scatter: avg EV vs avg actual P&L per config, faceted by model (2x3 grid)."""
    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        m = df[df["model_type"] == model]
        color = get_model_color(model)
        ax.scatter(
            m["avg_ev_pnl_blend"],
            m["avg_actual_pnl"],
            c=color,
            alpha=0.7,
            s=35,
            edgecolors="white",
            linewidths=0.5,
        )

        # Best-fit line
        if len(m) > 2:
            from scipy import stats

            slope, intercept, r, p, _ = stats.linregress(m["avg_ev_pnl_blend"], m["avg_actual_pnl"])
            x_range = np.linspace(m["avg_ev_pnl_blend"].min(), m["avg_ev_pnl_blend"].max(), 50)
            ax.plot(x_range, slope * x_range + intercept, "--", color=color, alpha=0.5)

            # Spearman
            rho, _ = stats.spearmanr(m["avg_ev_pnl_blend"], m["avg_actual_pnl"])
            ax.text(
                0.05,
                0.95,
                f"ρ = {rho:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                fontweight="bold",
            )

        ax.set_title(_short(model), fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        if i >= 3:
            ax.set_xlabel("Mean EV ($)")
        if i % 3 == 0:
            ax.set_ylabel("Mean Actual P&L ($)")

    fig.suptitle(
        "EV vs Actual P&L (Per Config) — Higher EV ≠ Higher Returns",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, PLOTS_DIR / "ev_vs_actual.png")


# ============================================================================
# Plot 6: Config Heatmap — Edge × KF with combined P&L
# ============================================================================


def plot_config_heatmap(df: pd.DataFrame) -> None:
    """Heatmaps of combined P&L for each model: edge_threshold × kelly_fraction.

    Uses a shared colorscale across all models with vmin=0 so that
    red = $0 (break-even) and green = max profit.  This makes colors
    directly comparable between subplots.
    """
    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]

    # Shared colorscale: vmin=0, vmax = max across all models
    global_max = max((df[df["model_type"] == m]["avg_actual_pnl"].max() * 2) for m in models)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        m = df[df["model_type"] == model]

        pivot = m.pivot_table(
            values="avg_actual_pnl",
            index="buy_edge_threshold",
            columns="kelly_fraction",
            aggfunc="first",
        )
        pivot = pivot * 2  # combined = avg * 2

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            interpolation="nearest",
            vmin=0,
            vmax=global_max,
        )

        # Annotate cells
        for ii in range(pivot.shape[0]):
            for jj in range(pivot.shape[1]):
                val = pivot.values[ii, jj]
                # Dark text on light background, white on dark
                brightness = val / global_max if global_max > 0 else 0.5
                text_color = "white" if brightness > 0.75 or brightness < 0.25 else "black"
                ax.text(
                    jj,
                    ii,
                    f"${val:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=8)
        ax.set_title(_short(model), fontsize=11, fontweight="bold")

        if i >= 3:
            ax.set_xlabel("Kelly Fraction")
        if i % 3 == 0:
            ax.set_ylabel("Edge Threshold")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Combined P&L (2024+2025): Edge Threshold × Kelly Fraction",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, PLOTS_DIR / "config_heatmap.png")


# ============================================================================
# Plot 7: Risk-Return (CVaR-5% vs Combined P&L)
# ============================================================================


def plot_risk_return(df: pd.DataFrame) -> None:
    """Scatter: avg CVaR-5% vs avg actual P&L, colored by model."""
    fig, ax = plt.subplots(figsize=(9, 6))

    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]
    cvar_col = "avg_cvar_5" if "avg_cvar_5" in df.columns else None

    if cvar_col is None:
        print("  Skipping risk-return plot: no avg_cvar_5 column")
        plt.close(fig)
        return

    for model in models:
        m = df[df["model_type"] == model]
        ax.scatter(
            m[cvar_col],
            m["avg_actual_pnl"],
            c=get_model_color(model),
            label=_short(model),
            alpha=0.6,
            s=40,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("Avg CVaR-5% ($) — Higher = Less Risky")
    ax.set_ylabel("Avg Actual P&L ($)")
    ax.set_title("Risk vs Return: CVaR-5% vs Actual P&L")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.grid(alpha=0.3)

    _save_fig(fig, PLOTS_DIR / "risk_return.png")


# ============================================================================
# Plot 8: Cross-Year Rank Correlation Bar Chart
# ============================================================================


def plot_rank_correlation(df: pd.DataFrame) -> None:
    """Bar chart of Spearman ρ (2024 rank vs 2025 rank) per model."""
    from scipy import stats

    models = [m for m in MODEL_ORDER if m in df["model_type"].unique()]

    rhos = []
    for model in models:
        m = df[df["model_type"] == model]
        rho, _ = stats.spearmanr(m["actual_pnl_2024"], m["actual_pnl_2025"])
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [get_model_color(m) for m in models]
    bars = ax.bar(range(len(models)), rhos, color=colors, alpha=0.8, edgecolor="white")

    for bar, rho in zip(bars, rhos, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rho + (0.03 if rho >= 0 else -0.06),
            f"{rho:.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_short(m) for m in models], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman ρ (2024 vs 2025 Ranking)")
    ax.set_title("Config Rank Stability Across Years")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, PLOTS_DIR / "rank_correlation.png")


def main() -> None:
    print("Loading data...")
    df = _load_cross_year()
    print(f"  {len(df)} configs loaded")
    print(f"  Models: {sorted(df['model_type'].unique())}")
    print(f"  Columns: {len(df.columns)}")
    print()

    print("Generating plots...")
    plot_model_pnl_overview(df)
    plot_edge_sensitivity(df)
    plot_kelly_sensitivity(df)
    plot_cross_year_scatter(df)
    plot_ev_vs_actual(df)
    plot_config_heatmap(df)
    plot_risk_return(df)
    plot_rank_correlation(df)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
