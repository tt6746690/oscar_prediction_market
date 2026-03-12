"""Comprehensive analysis and plotting for buy-and-hold backtest results.

Generates portfolio-focused analysis plots:

1.  P&L by category (grouped bars, per model)
2.  BH vs Rebalancing comparison (grouped bars)
3.  Entry-point × category P&L heatmap
4.  Cumulative P&L by entry window (line chart)
5.  Parameter sensitivity (portfolio-level P&L by parameter, grouped by model)
6.  Config neighborhood heatmap (kf × edge)
7.  Risk profile (portfolio P&L distribution histograms per model)
8.  Model comparison (total P&L bar chart across categories)
9.  Win rate by entry point
10. Per-category model edge trajectories (9 × timing plots)
11. Model vs market divergence (histogram + time series)
12. Portfolio P&L by entry point (marginal per-entry contribution)
13. Portfolio capital deployment (utilization and ROI)

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.analyze_plots --ceremony-year 2025
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.data.awards_calendar import CALENDARS
from oscar_prediction_market.one_offs.analysis_utils.style import (
    apply_style,
    get_model_color,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    YEAR_CONFIGS,
)
from oscar_prediction_market.trading.temporal_model import (
    get_post_nomination_snapshot_dates,
)

apply_style()

# ============================================================================
# Constants (defaults for 2025; overridden by main() based on --ceremony-year)
# ============================================================================

BACKTEST_EXP_DIR = Path("storage/d20260220_backtest_strategies")
EXP_DIR = Path("storage/d20260225_buy_hold_backtest")
RESULTS_DIR = EXP_DIR / "2025" / "results"
PLOTS_DIR = EXP_DIR / "2025" / "plots"

REBALANCING_RESULTS_DIR = BACKTEST_EXP_DIR / "2025" / "results_inferred_6h"

CALENDAR = CALENDARS[2025]
BANKROLL = 1000.0

MODEL_DISPLAY = {
    "lr": "Logistic Regression",
    "clogit": "Conditional Logit",
    "gbt": "Gradient Boosting",
    "cal_sgbt": "Cal. Softmax GBT",
    "avg_ensemble": "Avg Ensemble (4)",
}

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

# Event labels by snapshot date
_SNAP_INFO = None


def _get_snap_labels() -> dict[str, str]:
    global _SNAP_INFO
    if _SNAP_INFO is None:
        _SNAP_INFO = get_post_nomination_snapshot_dates(CALENDAR)
    return {str(d): " + ".join(e) for d, e in _SNAP_INFO}


def _save_fig(fig: matplotlib.figure.Figure, filename: str, *, dpi: int = 150) -> None:
    """Apply tight_layout, save to PLOTS_DIR, close, and print confirmation."""
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {filename}")


def _get_category_blue_shades(categories: list[str]) -> dict[str, str]:
    """Return light-to-dark blue shades for the given categories."""
    n = len(categories)
    cmap = plt.cm.Blues  # type: ignore[attr-defined]
    return {cat: cmap(0.3 + 0.6 * i / max(n - 1, 1)) for i, cat in enumerate(categories)}


# ============================================================================
# Data loading
# ============================================================================


def load_data() -> dict[str, pd.DataFrame]:
    """Load all result CSVs."""
    dfs: dict[str, pd.DataFrame] = {}
    for name in ["entry_pnl", "aggregate_pnl", "model_accuracy", "model_vs_market"]:
        path = RESULTS_DIR / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Derive snapshot_date from snapshot_key (e.g. "2025-02-08_dga" -> "2025-02-08")
            if "snapshot_key" in df.columns and "snapshot_date" not in df.columns:
                df["snapshot_date"] = df["snapshot_key"].str[:10]
            dfs[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
        else:
            print(f"  WARNING: {path} not found")
    return dfs


def load_rebalancing_data() -> pd.DataFrame | None:
    """Load rebalancing (d20260220) results for comparison."""
    path = REBALANCING_RESULTS_DIR / "daily_pnl.csv"
    if path.exists():
        df = pd.read_csv(path)
        print(f"  Loaded rebalancing P&L: {len(df)} rows")
        return df
    print(f"  WARNING: Rebalancing results not found at {path}")
    return None


# ============================================================================
# 1. P&L by Category (grouped bars)
# ============================================================================


def plot_pnl_by_category(agg_df: pd.DataFrame) -> None:
    """Show best aggregate P&L per category per model type."""
    active = agg_df[agg_df["entries_with_trades"] > 0]
    best = active.groupby(["model_type", "category"])["total_pnl"].max().reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    cats = sorted(best["category"].unique())
    x = np.arange(len(cats))
    model_types = sorted(best["model_type"].unique())
    width = 0.8 / max(len(model_types), 1)

    for i, mt in enumerate(model_types):
        subset = best[best["model_type"] == mt].set_index("category").reindex(cats)
        ax.bar(
            x + i * width,
            subset["total_pnl"].fillna(0),
            width,
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
            alpha=0.8,
        )

    ax.set_xticks(x + width * (len(model_types) - 1) / 2)
    ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c) for c in cats], rotation=45, ha="right")
    ax.set_ylabel("Best Aggregate P&L ($)")
    ax.set_title("Buy-and-Hold: Best P&L by Category × Model", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    _save_fig(fig, "pnl_by_category.png")


# ============================================================================
# 2. BH vs Rebalancing Comparison (grouped bars)
# ============================================================================


def plot_bh_vs_rebalancing(agg_df: pd.DataFrame) -> None:
    """Compare buy-hold vs rebalancing P&L per category (ensemble only)."""
    rb_df = load_rebalancing_data()
    if rb_df is None:
        return

    # Buy-hold: best ensemble config per category
    bh_ens = agg_df[(agg_df["model_type"] == "avg_ensemble") & (agg_df["entries_with_trades"] > 0)]
    bh_best = bh_ens.groupby("category")["total_pnl"].max().reset_index()
    bh_best.columns = ["category", "bh_pnl"]

    # Rebalancing: best ensemble config per category
    rb_ens = rb_df[(rb_df["model_type"] == "avg_ensemble") & (rb_df["total_trades"] > 0)]
    if "total_pnl" in rb_ens.columns:
        rb_best = rb_ens.groupby("category")["total_pnl"].max().reset_index()
        rb_best.columns = ["category", "rb_pnl"]
    else:
        print("  Rebalancing CSV missing total_pnl — skipping comparison")
        return

    merged = pd.merge(bh_best, rb_best, on="category", how="outer").fillna(0)
    cats = sorted(merged["category"].unique())
    merged = merged.set_index("category").reindex(cats)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cats))
    width = 0.35

    ax.bar(
        x - width / 2, merged["bh_pnl"], width, label="Buy-and-Hold", color="#3274a1", alpha=0.85
    )
    ax.bar(
        x + width / 2,
        merged["rb_pnl"],
        width,
        label="Rebalancing (d20260220)",
        color="#e1812c",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c) for c in cats], rotation=45, ha="right")
    ax.set_ylabel("Best Ensemble P&L ($)")
    ax.set_title(
        "Buy-and-Hold vs Rebalancing: Best Ensemble Config per Category\n"
        "(Inferred+6h timing, $1,000 bankroll per category)",
        fontsize=13,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    # Annotate totals
    bh_total = merged["bh_pnl"].sum()
    rb_total = merged["rb_pnl"].sum()
    ax.text(
        0.98,
        0.97,
        f"BH total: ${bh_total:+,.0f}\nRB total: ${rb_total:+,.0f}\nΔ: ${bh_total - rb_total:+,.0f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    _save_fig(fig, "bh_vs_rebalancing.png")


# ============================================================================
# 3. Entry-Point × Category P&L Heatmap
# ============================================================================


def plot_entry_category_heatmap(entry_df: pd.DataFrame) -> None:
    """Heatmap: entry_point (rows) × category (cols), P&L for ensemble reference config."""
    # Use ensemble with a reasonable config
    ens = entry_df[entry_df["model_type"] == "avg_ensemble"].copy()
    if ens.empty:
        ens = entry_df.copy()

    # Pick the most common profitable config for a representative heatmap
    active = ens[ens["total_trades"] > 0]
    if active.empty:
        print("  No active ensemble trades for heatmap")
        return

    # Group by config_label, sum total_pnl across all entries, pick best
    config_pnl = active.groupby("config_label")["total_pnl"].sum()
    best_config = str(config_pnl.idxmax())
    ref = active[active["config_label"] == best_config]

    categories = sorted(ref["category"].unique())
    snapshots = sorted(ref["entry_snapshot"].unique())
    snap_labels = _get_snap_labels()

    pivot = ref.pivot_table(
        index="entry_snapshot", columns="category", values="total_pnl", aggfunc="first"
    )
    pivot = pivot.reindex(index=snapshots, columns=categories).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    if vmax == 0:
        vmax = 1

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [CATEGORY_DISPLAY.get(c, c) for c in categories], rotation=45, ha="right", fontsize=10
    )
    ax.set_yticks(range(len(snapshots)))
    y_labels = []
    for s in snapshots:
        event = snap_labels.get(s, "")
        y_labels.append(f"{s[5:]}  {event[:30]}")
    ax.set_yticklabels(y_labels, fontsize=9)

    # Annotate cells
    for i in range(len(snapshots)):
        for j in range(len(categories)):
            val = pivot.values[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"${val:+.0f}", ha="center", va="center", fontsize=8, color=color)

    # Row totals
    row_totals = pivot.sum(axis=1)
    for i, total in enumerate(row_totals):
        ax.text(
            len(categories) + 0.3,
            i,
            f"${total:+.0f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Column totals
    col_totals = pivot.sum(axis=0)
    for j, total in enumerate(col_totals):
        ax.text(
            j,
            len(snapshots) + 0.3,
            f"${total:+.0f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    fig.colorbar(im, ax=ax, label="P&L ($)", shrink=0.8)
    ax.set_title(
        f"Entry-Point × Category P&L Heatmap (Ensemble, best config)\nConfig: {best_config[:60]}",
        fontsize=12,
    )
    _save_fig(fig, "entry_category_heatmap.png")


# ============================================================================
# 4. Cumulative P&L by Entry Window
# ============================================================================


def plot_cumulative_pnl_by_entry(entry_df: pd.DataFrame) -> None:
    """Line chart: cumulative P&L as entries are added, one line per model."""
    # Use a reasonable reference config (best aggregate ensemble config)
    ens_agg = entry_df.groupby(["model_type", "config_label"])["total_pnl"].sum()
    best_ens_config: str = (
        str(ens_agg.xs("avg_ensemble", level=0).idxmax())
        if "avg_ensemble" in entry_df["model_type"].values
        else str(ens_agg.groupby(level=0).apply(lambda x: x.idxmax()).iloc[0][1])
    )  # type: ignore[arg-type]  # pandas overloads

    ref = entry_df[entry_df["config_label"] == best_ens_config].copy()
    if ref.empty:
        ref = entry_df.copy()
        best_ens_config = str(ref.groupby("config_label")["total_pnl"].sum().idxmax())
        ref = entry_df[entry_df["config_label"] == best_ens_config]

    models = sorted(ref["model_type"].unique())
    snapshots = sorted(ref["entry_snapshot"].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    for mt in models:
        mt_data = ref[ref["model_type"] == mt]
        # Sum across categories per entry point
        per_entry = mt_data.groupby("entry_snapshot")["total_pnl"].sum()
        per_entry = per_entry.reindex(snapshots).fillna(0)
        cumul = per_entry.cumsum()
        ax.plot(
            range(len(snapshots)),
            cumul.values,
            marker="o",
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
            linewidth=2,
        )

    snap_labels = _get_snap_labels()
    x_labels = [f"{s[5:]}\n{snap_labels.get(s, '')[:25]}" for s in snapshots]
    ax.set_xticks(range(len(snapshots)))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_xlabel("Entry Point")
    ax.set_title(
        f"Cumulative P&L as Entries Are Added\n(Config: {best_ens_config[:50]})",
        fontsize=12,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    _save_fig(fig, "cumulative_pnl_by_entry.png")


# ============================================================================
# 5. Parameter Sensitivity (portfolio-level)
# ============================================================================


def plot_parameter_sensitivity(agg_df: pd.DataFrame) -> None:
    """Multi-panel parameter sensitivity: portfolio-level mean P&L by parameter value.

    For each parameter, bars are grouped by model_type showing mean portfolio P&L
    at each parameter value. Portfolio P&L = sum of per-category P&L per config.
    """
    df = agg_df[agg_df["bankroll_mode"] == "fixed"].copy()
    if df.empty:
        df = agg_df.copy()

    params = [
        ("fee_type", "Fee Type"),
        ("kelly_fraction", "Kelly Fraction"),
        ("buy_edge_threshold", "Edge Threshold"),
        ("kelly_mode", "Kelly Mode"),
        ("allowed_directions", "Trading Side"),
    ]

    # Compute portfolio P&L: sum across categories per config
    group_cols = [
        "model_type",
        "config_label",
        "fee_type",
        "kelly_fraction",
        "buy_edge_threshold",
        "kelly_mode",
        "allowed_directions",
    ]
    available_cols = [c for c in group_cols if c in df.columns]
    has_capital = "capital_deployed" in df.columns
    agg_dict: dict[str, tuple[str, str]] = {"portfolio_pnl": ("total_pnl", "sum")}
    if has_capital:
        agg_dict["capital_deployed"] = ("capital_deployed", "sum")

    portfolio = df.groupby(available_cols).agg(**agg_dict).reset_index()

    models = sorted(portfolio["model_type"].unique())
    n_params = len(params)

    fig, axes = plt.subplots(n_params, 1, figsize=(16, 4 * n_params))
    if n_params == 1:
        axes = [axes]

    sensitivity_rows: list[dict] = []

    for ax_idx, (param, param_label) in enumerate(params):
        if param not in portfolio.columns:
            continue
        ax = axes[ax_idx]
        vals = sorted(portfolio[param].unique(), key=str)
        x = np.arange(len(vals))
        n_models = len(models)
        width = 0.8 / max(n_models, 1)

        for i, mt in enumerate(models):
            mt_df = portfolio[portfolio["model_type"] == mt]
            means = []
            for v in vals:
                subset = mt_df[mt_df[param] == v]
                means.append(subset["portfolio_pnl"].mean() if len(subset) > 0 else 0)
            ax.bar(
                x + i * width,
                means,
                width=width,
                label=MODEL_DISPLAY.get(mt, mt),
                color=get_model_color(mt),
                alpha=0.8,
                edgecolor="white",
                linewidth=0.3,
            )
            for v, mean_pnl in zip(vals, means, strict=True):
                v_df = mt_df[mt_df[param] == v]
                sensitivity_rows.append(
                    {
                        "parameter": param,
                        "value": str(v),
                        "model_type": mt,
                        "mean_portfolio_pnl": round(mean_pnl, 2),
                        "n_configs": len(v_df),
                        "pct_profitable": round(
                            (v_df["portfolio_pnl"] > 0).sum() / max(len(v_df), 1) * 100, 1
                        ),
                    }
                )

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_ylabel("Mean Portfolio P&L ($)")
        ax.set_title(param_label, fontsize=12)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=8, ncol=min(5, n_models), loc="upper right")

    fig.suptitle("Parameter Sensitivity: Mean Portfolio P&L by Parameter Value", fontsize=14)
    _save_fig(fig, "parameter_sensitivity.png")

    if sensitivity_rows:
        pd.DataFrame(sensitivity_rows).to_csv(
            RESULTS_DIR / "parameter_sensitivity.csv", index=False
        )
        print(f"  Saved parameter_sensitivity.csv ({len(sensitivity_rows)} rows)")


# ============================================================================
# 6. Config Neighborhood Heatmap
# ============================================================================


def plot_config_neighborhood(agg_df: pd.DataFrame) -> None:
    """Heatmap: kf × edge → aggregate P&L, faceted by (model, allowed_directions)."""
    df = agg_df[(agg_df["fee_type"] == "maker") & (agg_df["bankroll_mode"] == "fixed")].copy()
    if df.empty:
        df = agg_df[agg_df["bankroll_mode"] == "fixed"].copy()
    if df.empty:
        print("  No fixed-bankroll data for neighborhood heatmap")
        return

    # Aggregate across categories per config
    agg = (
        df.groupby(
            [
                "model_type",
                "config_label",
                "kelly_fraction",
                "buy_edge_threshold",
                "kelly_mode",
                "allowed_directions",
            ]
        )
        .agg(agg_pnl=("total_pnl", "sum"))
        .reset_index()
    )

    models = sorted(agg["model_type"].unique())
    sides = sorted(agg["allowed_directions"].unique())
    kfs = sorted(agg["kelly_fraction"].unique())
    edges = sorted(agg["buy_edge_threshold"].unique())

    n_models = len(models)
    n_sides = len(sides)

    fig, axes = plt.subplots(
        n_models,
        n_sides,
        figsize=(5 * n_sides, 4 * n_models),
        squeeze=False,
    )

    for i, mt in enumerate(models):
        for j, side in enumerate(sides):
            ax = axes[i, j]
            sub = agg[(agg["model_type"] == mt) & (agg["allowed_directions"] == side)]

            pivot_data = np.full((len(kfs), len(edges)), np.nan)
            for ki, kf in enumerate(kfs):
                for ei, edge in enumerate(edges):
                    cell = sub[(sub["kelly_fraction"] == kf) & (sub["buy_edge_threshold"] == edge)]
                    if not cell.empty:
                        pivot_data[ki, ei] = cell["agg_pnl"].mean()

            vmax = max(abs(np.nanmin(pivot_data)), abs(np.nanmax(pivot_data)), 1)
            im = ax.imshow(
                pivot_data,
                cmap="RdYlGn",
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )

            ax.set_xticks(range(len(edges)))
            ax.set_xticklabels([f"{e:.2f}" for e in edges], fontsize=7)
            ax.set_yticks(range(len(kfs)))
            ax.set_yticklabels([f"{k:.2f}" for k in kfs], fontsize=7)

            if i == n_models - 1:
                ax.set_xlabel("Edge Threshold")
            if j == 0:
                ax.set_ylabel("Kelly Fraction")
            ax.set_title(f"{MODEL_DISPLAY.get(mt, mt)}\nSide={side}", fontsize=9)

            for ki in range(len(kfs)):
                for ei in range(len(edges)):
                    val = pivot_data[ki, ei]
                    if not np.isnan(val):
                        color = "white" if abs(val) > vmax * 0.55 else "black"
                        ax.text(
                            ei, ki, f"${val:.0f}", ha="center", va="center", fontsize=6, color=color
                        )

            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        "Config Neighborhood: Aggregate P&L (all categories)\n"
        "Fee=maker, bankroll=fixed, avg across kelly_mode",
        fontsize=14,
    )
    _save_fig(fig, "config_neighborhood_heatmap.png")


# ============================================================================
# 7. Risk Profile (portfolio P&L distribution)
# ============================================================================


def plot_risk_profile(agg_df: pd.DataFrame) -> None:
    """Histogram: portfolio P&L distribution per model.

    Portfolio P&L = sum of per-category P&L for each (model, config) pair.
    Shows the distribution across configs for each model.
    """
    df = agg_df[agg_df["bankroll_mode"] == "fixed"].copy()
    if df.empty:
        df = agg_df.copy()

    has_capital = "capital_deployed" in df.columns

    # Compute portfolio P&L: sum across categories per (model, config)
    agg_dict: dict[str, tuple[str, str]] = {
        "portfolio_pnl": ("total_pnl", "sum"),
        "n_categories": ("total_pnl", "count"),
    }
    if has_capital:
        agg_dict["capital_deployed"] = ("capital_deployed", "sum")

    portfolio = df.groupby(["model_type", "config_label"]).agg(**agg_dict).reset_index()

    if has_capital:
        portfolio["roi"] = np.where(
            portfolio["capital_deployed"] > 0,
            portfolio["portfolio_pnl"] / portfolio["capital_deployed"] * 100,
            0.0,
        )

    models = sorted(portfolio["model_type"].unique())
    n_models = min(len(models), 6)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    flat_axes: list[Any] = list(np.asarray(axes).flatten()) if n_models > 1 else [axes]

    risk_rows: list[dict] = []

    for i, mt in enumerate(models[:n_models]):
        ax = flat_axes[i]
        mt_df = portfolio[portfolio["model_type"] == mt]
        pnls = mt_df["portfolio_pnl"]
        color = get_model_color(mt)

        ax.hist(pnls, bins=25, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
        ax.axvline(
            x=pnls.mean(),
            color="blue",
            linestyle="-",
            linewidth=1.5,
            label=f"Mean: ${pnls.mean():.0f}",
        )
        ax.axvline(
            x=pnls.median(),
            color="green",
            linestyle="-",
            linewidth=1.5,
            label=f"Median: ${pnls.median():.0f}",
        )

        n_total = len(pnls)
        n_profit = (pnls > 0).sum()
        n_loss = (pnls < 0).sum()
        ax.set_title(
            f"{MODEL_DISPLAY.get(mt, mt)}\n"
            f"P(profit)={n_profit / n_total * 100:.0f}%, "
            f"P(loss)={n_loss / n_total * 100:.0f}%",
            fontsize=11,
        )
        ax.set_xlabel("Portfolio P&L ($)")
        ax.set_ylabel("# Configs")
        ax.legend(fontsize=7)

        row: dict[str, Any] = {
            "model_type": mt,
            "n_configs": n_total,
            "prob_profit": round(n_profit / n_total * 100, 1),
            "prob_loss": round(n_loss / n_total * 100, 1),
            "mean_pnl": round(pnls.mean(), 2),
            "median_pnl": round(pnls.median(), 2),
            "best_pnl": round(pnls.max(), 2),
            "worst_pnl": round(pnls.min(), 2),
            "p10_pnl": round(float(pnls.quantile(0.1)), 2),
            "p90_pnl": round(float(pnls.quantile(0.9)), 2),
        }
        if has_capital:
            roi = mt_df["roi"]
            row["mean_roi_pct"] = round(roi.mean(), 2)
            row["median_roi_pct"] = round(roi.median(), 2)
        risk_rows.append(row)

    # Hide unused axes
    for j in range(n_models, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle("Risk Profile: Portfolio P&L Distribution by Model", fontsize=14)
    _save_fig(fig, "risk_profile.png")

    if risk_rows:
        pd.DataFrame(risk_rows).to_csv(RESULTS_DIR / "risk_profile.csv", index=False)
        print(f"  Saved risk_profile.csv ({len(risk_rows)} rows)")


# ============================================================================
# 8. Model Comparison Bar Chart
# ============================================================================


def plot_model_comparison(agg_df: pd.DataFrame) -> None:
    """Stacked bar: total best-config P&L per model, colored by category."""
    active = agg_df[agg_df["entries_with_trades"] > 0]
    idx = active.groupby(["category", "model_type"])["total_pnl"].idxmax()
    best = active.loc[idx]

    models = sorted(best["model_type"].unique())
    categories = sorted(best["category"].unique())
    cat_colors = _get_category_blue_shades(categories)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(models))
    bottoms_pos = np.zeros(len(models))
    bottoms_neg = np.zeros(len(models))

    for cat in categories:
        vals = []
        for mt in models:
            sub = best[(best["model_type"] == mt) & (best["category"] == cat)]
            vals.append(sub["total_pnl"].values[0] if len(sub) > 0 else 0)
        vals_arr = np.array(vals)

        pos_vals = np.where(vals_arr > 0, vals_arr, 0)
        neg_vals = np.where(vals_arr < 0, vals_arr, 0)

        ax.bar(
            x,
            pos_vals,
            bottom=bottoms_pos,
            color=cat_colors[cat],
            label=CATEGORY_DISPLAY.get(cat, cat),
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            x, neg_vals, bottom=bottoms_neg, color=cat_colors[cat], edgecolor="white", linewidth=0.3
        )

        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    # Net total annotations
    for i, mt in enumerate(models):
        total = best[best["model_type"] == mt]["total_pnl"].sum()
        y_pos = max(bottoms_pos[i], 0) + 200
        ax.text(i, y_pos, f"${total:+,.0f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("P&L ($)")
    ax.set_title(
        "Model Comparison: Sum of Best-Config P&L Across Categories\n(stacked by category)",
        fontsize=13,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    _save_fig(fig, "model_comparison.png")


# ============================================================================
# 9. Win Rate by Entry Point
# ============================================================================


def plot_win_rate_by_entry(entry_df: pd.DataFrame) -> None:
    """Bar chart: win rate (% of active entries with P&L > 0) by entry point."""
    active = entry_df[entry_df["total_trades"] > 0].copy()
    if active.empty:
        return

    snapshots = sorted(active["entry_snapshot"].unique())
    snap_labels = _get_snap_labels()

    win_rates = []
    avg_pnls = []
    n_entries = []
    for snap in snapshots:
        snap_data = active[active["entry_snapshot"] == snap]
        win_rates.append((snap_data["total_pnl"] > 0).mean() * 100)
        avg_pnls.append(snap_data["total_pnl"].mean())
        n_entries.append(len(snap_data))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(snapshots))

    bars = ax1.bar(x, win_rates, color="#3274a1", alpha=0.8, label="Win Rate %")
    ax1.set_ylabel("Win Rate (%)", color="#3274a1")
    ax1.tick_params(axis="y", labelcolor="#3274a1")

    # Annotate bars with win rate
    for bar, wr in zip(bars, win_rates, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{wr:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Secondary axis: average P&L
    ax2 = ax1.twinx()
    ax2.plot(x, avg_pnls, "o-", color="#e1812c", linewidth=2, markersize=8, label="Avg P&L")
    ax2.set_ylabel("Average P&L ($)", color="#e1812c")
    ax2.tick_params(axis="y", labelcolor="#e1812c")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    x_labels = [f"{s[5:]}\n{snap_labels.get(s, '')[:20]}" for s in snapshots]
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax1.set_title("Win Rate & Average P&L by Entry Point", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    _save_fig(fig, "win_rate_by_entry.png")


# ============================================================================
# 10. Per-Category Model Edge Trajectories
# ============================================================================


def plot_per_category_timing(mvm_df: pd.DataFrame) -> None:
    """Per-category timing analysis: model edge + per-contract profit over season."""
    winners = mvm_df[mvm_df["is_winner"]].copy()
    if winners.empty:
        print("  No winner data for timing plots")
        return

    categories = sorted(winners["category"].unique())
    models = sorted(winners["model_type"].unique())

    # Multi-panel: one subplot per category
    n_cats = len(categories)
    ncols = 3
    nrows = (n_cats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    flat_axes: list[Any] = list(np.asarray(axes).flatten())

    for ci, cat in enumerate(categories):
        ax = flat_axes[ci]
        cat_winners = winners[winners["category"] == cat]

        for mt in models:
            sub = cat_winners[cat_winners["model_type"] == mt].sort_values("snapshot_date")
            if sub.empty:
                continue
            edge = (sub["model_prob"] - sub["market_prob"]) * 100
            ax.plot(
                range(len(sub)),
                edge,
                marker="o",
                label=MODEL_DISPLAY.get(mt, mt),
                color=get_model_color(mt),
                linewidth=1.5,
            )

        # Per-contract profit on secondary axis
        ax2 = ax.twinx()
        first_mt = cat_winners["model_type"].iloc[0]
        mkt = cat_winners[cat_winners["model_type"] == first_mt].sort_values("snapshot_date")
        profit = 100 - mkt["market_prob"].values * 100
        ax2.fill_between(
            range(len(mkt)),
            profit,
            alpha=0.12,
            color="green" if profit.mean() > 0 else "red",
        )
        ax2.set_ylabel("Profit/contract (¢)", fontsize=8)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Model Edge (pp)", fontsize=9)
        ax.set_title(CATEGORY_DISPLAY.get(cat, cat), fontsize=11)
        dates = mkt["snapshot_date"].values
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=7)

        if ci == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper left")

    for j in range(n_cats, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle(
        "Per-Category Timing: Model Edge & Per-Contract Profit Over Season",
        fontsize=14,
    )
    _save_fig(fig, "per_category_timing.png")

    # Also plot per-contract profit bar chart (model-independent)
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = sorted(winners["snapshot_date"].unique())
    for cat in categories:
        cat_data = winners[
            (winners["category"] == cat)
            & (winners["model_type"] == winners[winners["category"] == cat]["model_type"].iloc[0])
        ].sort_values("snapshot_date")
        if cat_data.empty:
            continue
        profit = (1.0 - cat_data["market_prob"].values) * 100
        ax.plot(
            range(len(cat_data)),
            profit,
            marker="s",
            linewidth=2,
            label=CATEGORY_DISPLAY.get(cat, cat),
        )

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right")
    ax.set_xlabel("Entry Date")
    ax.set_ylabel("Profit per Contract (¢)")
    ax.set_title("Per-Contract Profit on Winner by Entry Date", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    _save_fig(fig, "entry_date_profit.png")


# ============================================================================
# 11. Model vs Market Divergence
# ============================================================================


def plot_model_vs_market_divergence(mvm_df: pd.DataFrame) -> None:
    """Two-panel: (L) divergence histogram winners vs non-winners, (R) mean divergence over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: divergence distribution
    winners = mvm_df[mvm_df["is_winner"]]
    losers = mvm_df[~mvm_df["is_winner"]]

    if not winners.empty:
        ax1.hist(
            winners["divergence"],
            bins=30,
            alpha=0.6,
            color="green",
            label="Winners",
            density=True,
        )
    if not losers.empty:
        ax1.hist(
            losers["divergence"],
            bins=30,
            alpha=0.4,
            color="red",
            label="Non-winners",
            density=True,
        )

    # Percentile markers for winners
    if not winners.empty:
        p67 = winners["divergence"].quantile(0.33)  # 67th pctile (above this)
        p90 = winners["divergence"].quantile(0.10)
        ax1.axvline(x=p67, color="blue", linestyle=":", alpha=0.7, label=f"67th pctile: {p67:.2f}")
        ax1.axvline(x=p90, color="blue", linestyle="--", alpha=0.7)

    ax1.set_xlabel("Model − Market Probability")
    ax1.set_ylabel("Density")
    ax1.set_title("Model vs Market Divergence Distribution", fontsize=12)
    ax1.legend(fontsize=8)

    # Panel B: mean winner divergence over time
    dates = sorted(mvm_df["snapshot_date"].unique())
    models = sorted(mvm_df["model_type"].unique())

    for mt in models:
        mt_winners = winners[winners["model_type"] == mt]
        if mt_winners.empty:
            continue
        mean_div = mt_winners.groupby("snapshot_date")["divergence"].mean()
        mean_div = mean_div.reindex(dates).fillna(0)
        ax2.plot(
            range(len(dates)),
            mean_div.values,
            marker="o",
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
            linewidth=2,
        )

    ax2.set_xticks(range(len(dates)))
    snap_labels = _get_snap_labels()
    ax2.set_xticklabels(
        [f"{d[5:]}\n{snap_labels.get(d, '')[:15]}" for d in dates],
        fontsize=7,
        rotation=45,
        ha="right",
    )
    ax2.set_ylabel("Mean Winner Divergence (model − market)")
    ax2.set_title("Winner Edge Over Time", fontsize=12)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    fig.suptitle("Model vs Market: Edge Analysis", fontsize=14)
    _save_fig(fig, "model_vs_market_divergence.png")


# ============================================================================
# 12. Portfolio P&L by Entry Point
# ============================================================================


def plot_portfolio_pnl_by_entry(entry_df: pd.DataFrame) -> None:
    """Grouped bar chart: portfolio P&L at each entry point, grouped by model.

    For each config, computes portfolio P&L (sum across categories) per entry point,
    then shows the mean across configs for each (model, entry) pair.
    """
    active = entry_df[entry_df["total_trades"] > 0].copy()
    if active.empty:
        print("  No active trades for portfolio entry analysis")
        return

    has_capital = "capital_deployed" in active.columns

    # Portfolio P&L per (model, config, entry): sum across categories
    agg_dict: dict[str, tuple[str, str]] = {"portfolio_pnl": ("total_pnl", "sum")}
    if has_capital:
        agg_dict["capital_deployed"] = ("capital_deployed", "sum")

    portfolio = (
        active.groupby(["model_type", "config_label", "entry_snapshot"])
        .agg(**agg_dict)
        .reset_index()
    )

    # Mean across configs for each (model, entry)
    mean_portfolio = (
        portfolio.groupby(["model_type", "entry_snapshot"])
        .agg(mean_pnl=("portfolio_pnl", "mean"), median_pnl=("portfolio_pnl", "median"))
        .reset_index()
    )

    models = sorted(mean_portfolio["model_type"].unique())
    snapshots = sorted(mean_portfolio["entry_snapshot"].unique())
    snap_labels = _get_snap_labels()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(snapshots))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    for i, mt in enumerate(models):
        mt_df = mean_portfolio[mean_portfolio["model_type"] == mt].set_index("entry_snapshot")
        mt_df = mt_df.reindex(snapshots)
        ax.bar(
            x + i * width,
            mt_df["mean_pnl"].fillna(0),
            width=width,
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
            alpha=0.8,
        )

    x_labels = [f"{s[5:]}\n{snap_labels.get(s, '')[:25]}" for s in snapshots]
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Mean Portfolio P&L ($)")
    ax.set_xlabel("Entry Point")
    ax.set_title(
        "Portfolio P&L by Entry Point\n(mean across configs, portfolio = sum across categories)",
        fontsize=13,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    _save_fig(fig, "portfolio_pnl_by_entry.png")


# ============================================================================
# 14. Portfolio Capital Deployment
# ============================================================================


def plot_portfolio_capital_deployment(entry_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
    """Capital deployment analysis: utilization rate and ROI comparison.

    Compares capital allocated (bankroll * n_entries * n_categories) vs actually
    deployed, and shows ROI on deployed vs allocated capital.
    """
    if "capital_deployed" not in agg_df.columns:
        print("  capital_deployed column not available — skipping capital deployment plot")
        return

    df = agg_df[agg_df["bankroll_mode"] == "fixed"].copy()
    if df.empty:
        df = agg_df.copy()

    n_categories = df["category"].nunique()
    n_entries = entry_df["entry_snapshot"].nunique() if not entry_df.empty else 1

    # Portfolio-level: sum across categories per (model, config)
    portfolio = (
        df.groupby(["model_type", "config_label"])
        .agg(
            portfolio_pnl=("total_pnl", "sum"),
            capital_deployed=("capital_deployed", "sum"),
        )
        .reset_index()
    )

    # Total capital allocated per config = bankroll * n_entries * n_categories
    total_allocated = BANKROLL * n_entries * n_categories
    portfolio["capital_allocated"] = total_allocated
    portfolio["utilization"] = np.where(
        portfolio["capital_allocated"] > 0,
        portfolio["capital_deployed"] / portfolio["capital_allocated"] * 100,
        0.0,
    )
    portfolio["roi_deployed"] = np.where(
        portfolio["capital_deployed"] > 0,
        portfolio["portfolio_pnl"] / portfolio["capital_deployed"] * 100,
        0.0,
    )
    portfolio["roi_allocated"] = np.where(
        portfolio["capital_allocated"] > 0,
        portfolio["portfolio_pnl"] / portfolio["capital_allocated"] * 100,
        0.0,
    )

    models = sorted(portfolio["model_type"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Utilization by model
    ax = axes[0]
    util_by_model = portfolio.groupby("model_type")["utilization"].agg(["mean", "median"])
    x = np.arange(len(models))
    ax.bar(
        x,
        [util_by_model.loc[mt, "mean"] for mt in models],
        color=[get_model_color(mt) for mt in models],
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(mt, mt) for mt in models], fontsize=8, rotation=30, ha="right"
    )
    ax.set_ylabel("Mean Utilization (%)")
    ax.set_title("Capital Utilization by Model", fontsize=11)
    ax.set_ylim(0, 100)

    # Panel B: ROI on deployed capital by model
    ax = axes[1]
    roi_d = portfolio.groupby("model_type")["roi_deployed"].agg(["mean", "median"])
    ax.bar(
        x,
        [roi_d.loc[mt, "mean"] for mt in models],
        color=[get_model_color(mt) for mt in models],
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(mt, mt) for mt in models], fontsize=8, rotation=30, ha="right"
    )
    ax.set_ylabel("Mean ROI on Deployed (%)")
    ax.set_title("ROI on Deployed Capital", fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel C: Scatter of ROI vs utilization (all configs)
    ax = axes[2]
    for mt in models:
        mt_df = portfolio[portfolio["model_type"] == mt]
        ax.scatter(
            mt_df["utilization"],
            mt_df["roi_deployed"],
            alpha=0.3,
            s=15,
            color=get_model_color(mt),
            label=MODEL_DISPLAY.get(mt, mt),
        )
    ax.set_xlabel("Capital Utilization (%)")
    ax.set_ylabel("ROI on Deployed Capital (%)")
    ax.set_title("ROI vs Utilization", fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(
        f"Capital Deployment Analysis\n"
        f"(Allocated: ${total_allocated:,.0f} per config = "
        f"${BANKROLL:.0f} × {n_entries} entries × {n_categories} categories)",
        fontsize=13,
    )
    _save_fig(fig, "portfolio_capital_deployment.png")


# ============================================================================
# Summary Tables (printed, not plotted)
# ============================================================================


def print_summary_tables(agg_df: pd.DataFrame, entry_df: pd.DataFrame) -> None:
    """Print comprehensive summary tables to stdout."""
    print("\n" + "=" * 80)
    print("SUMMARY: Best P&L per Model × Category (cherry-picked best config)")
    print("=" * 80)

    active = agg_df[agg_df["entries_with_trades"] > 0]
    idx = active.groupby(["category", "model_type"])["total_pnl"].idxmax()
    best = active.loc[idx]

    models = sorted(best["model_type"].unique())
    categories = sorted(best["category"].unique())

    print(f"\n{'Category':<22}", end="")
    for mt in models:
        print(f" {mt:>12}", end="")
    print()
    print("-" * (22 + 13 * len(models)))

    for cat in categories:
        print(f"{CATEGORY_DISPLAY.get(cat, cat):<22}", end="")
        for mt in models:
            sub = best[(best["category"] == cat) & (best["model_type"] == mt)]
            if not sub.empty:
                pnl = sub["total_pnl"].values[0]
                print(f" ${pnl:>+10.0f}", end="")
            else:
                print(f" {'—':>12}", end="")
        print()

    print(f"{'TOTAL':<22}", end="")
    for mt in models:
        total = best[best["model_type"] == mt]["total_pnl"].sum()
        print(f" ${total:>+10.0f}", end="")
    print()

    # % profitable configs
    print(f"\n{'% Profitable Configs':<22}", end="")
    for mt in models:
        mt_df = agg_df[agg_df["model_type"] == mt]
        pct = (mt_df["total_pnl"] > 0).sum() / max(len(mt_df), 1) * 100
        print(f" {pct:>11.1f}%", end="")
    print()

    # Entry point summary
    print("\n" + "=" * 80)
    print("SUMMARY: Entry Point Performance (all models, all configs with trades)")
    print("=" * 80)
    active_entries = entry_df[entry_df["total_trades"] > 0]
    snap_labels = _get_snap_labels()
    for snap in sorted(active_entries["entry_snapshot"].unique()):
        snap_data = active_entries[active_entries["entry_snapshot"] == snap]
        wr = (snap_data["total_pnl"] > 0).mean() * 100
        avg = snap_data["total_pnl"].mean()
        best_val = snap_data["total_pnl"].max()
        worst_val = snap_data["total_pnl"].min()
        events = snap_labels.get(snap, "")
        print(
            f"  {snap}  {events:<30} win_rate={wr:>5.1f}%  "
            f"avg=${avg:>+7.0f}  best=${best_val:>+7.0f}  worst=${worst_val:>+7.0f}"
        )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all analyses and generate plots."""
    global RESULTS_DIR, PLOTS_DIR, REBALANCING_RESULTS_DIR, CALENDAR, _SNAP_INFO

    parser = argparse.ArgumentParser(description="Buy-and-hold analysis plots")
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=2025,
        choices=sorted(YEAR_CONFIGS.keys()),
        help="Ceremony year to analyze (default: 2025).",
    )
    args = parser.parse_args()

    year_config = YEAR_CONFIGS[args.ceremony_year]
    RESULTS_DIR = year_config.results_dir
    PLOTS_DIR = year_config.plots_dir
    REBALANCING_RESULTS_DIR = BACKTEST_EXP_DIR / str(args.ceremony_year) / "results_inferred_6h"
    CALENDAR = year_config.calendar
    _SNAP_INFO = None  # reset cached snap info

    print("=" * 70)
    print(f"Buy-and-Hold Backtest: {args.ceremony_year} Comprehensive Analysis")
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots:   {PLOTS_DIR}")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_data()

    entry_df = dfs.get("entry_pnl")
    agg_df = dfs.get("aggregate_pnl")
    mvm_df = dfs.get("model_vs_market")

    if entry_df is None or agg_df is None:
        print("ERROR: Missing required CSVs (entry_pnl, aggregate_pnl)")
        return

    print("\n--- 1. P&L by Category ---")
    plot_pnl_by_category(agg_df)

    print("\n--- 2. BH vs Rebalancing (grouped bars) ---")
    plot_bh_vs_rebalancing(agg_df)

    print("\n--- 3. Entry × Category P&L Heatmap ---")
    plot_entry_category_heatmap(entry_df)

    print("\n--- 4. Cumulative P&L by Entry ---")
    plot_cumulative_pnl_by_entry(entry_df)

    print("\n--- 5. Parameter Sensitivity ---")
    plot_parameter_sensitivity(agg_df)

    print("\n--- 6. Config Neighborhood Heatmap ---")
    plot_config_neighborhood(agg_df)

    print("\n--- 7. Risk Profile ---")
    plot_risk_profile(agg_df)

    print("\n--- 8. Model Comparison ---")
    plot_model_comparison(agg_df)

    print("\n--- 9. Win Rate by Entry ---")
    plot_win_rate_by_entry(entry_df)

    if mvm_df is not None:
        print("\n--- 10. Per-Category Timing ---")
        plot_per_category_timing(mvm_df)

        print("\n--- 11. Model vs Market Divergence ---")
        plot_model_vs_market_divergence(mvm_df)

    print("\n--- 12. Portfolio P&L by Entry ---")
    plot_portfolio_pnl_by_entry(entry_df)

    print("\n--- 13. Portfolio Capital Deployment ---")
    plot_portfolio_capital_deployment(entry_df, agg_df)

    print("\n--- Summary Tables ---")
    print_summary_tables(agg_df, entry_df)

    print(f"\nAnalysis complete! {len(list(PLOTS_DIR.glob('*.png')))} plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
