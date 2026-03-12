"""Analyze multi-category backtest results for 2025 ceremony.

Reads CSV results from ``run_backtests.py`` and produces:

1. Model accuracy heatmaps (Brier score per model × category × snapshot)
2. Classification vs trading performance scatter
3. Model vs market divergence analysis
4. Per-precursor P&L breakdown
5. Return decomposition (selection / timing / sizing alpha)
6. Trading parameter grid analysis
7. Spread & liquidity report
8. Best config selection (pre-commitment for 2026)
9. Timing analysis — P&L by entry date
10. Detailed model performance — probability trajectories, divergence heatmaps,
    feature importance, calibration

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260220_backtest_strategies.analyze

    # Use a specific results dir:
    uv run python -m ... analyze --results-dir results_delay_0
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from oscar_prediction_market.data.awards_calendar import CALENDARS
from oscar_prediction_market.one_offs.analysis_utils.style import (
    apply_style,
    get_model_color,
)
from oscar_prediction_market.trading.temporal_model import (
    get_post_nomination_snapshot_dates,
)

apply_style()

# ============================================================================
# Constants
# ============================================================================

EXP_DIR = Path("storage/d20260220_backtest_strategies")
# Default to inferred+6h results — the most realistic execution model.
# Override via --results-dir CLI flag.
RESULTS_DIR = EXP_DIR / "2025" / "results_inferred_6h"
PLOTS_DIR = EXP_DIR / "2025" / "plots"
CALENDAR = CALENDARS[2025]

# Short name → display name for plots.
# Intentionally different from analysis.style.MODEL_DISPLAY — these use longer
# names suited to this experiment's standalone figures.
MODEL_DISPLAY = {
    "lr": "Logistic Regression",
    "clogit": "Conditional Logit",
    "gbt": "Gradient Boosting",
    "cal_sgbt": "Cal. Softmax GBT",
    "avg_ensemble": "Avg Ensemble (4)",
}

# Category display names.
# Intentionally different from analysis.style.CATEGORY_DISPLAY for some entries
# (shorter abbreviations suited to compact multi-panel figures).
CATEGORY_DISPLAY = {
    "best_picture": "Best Picture",
    "directing": "Directing",
    "actor_leading": "Actor",
    "actress_leading": "Actress",
    "actor_supporting": "Supp. Actor",
    "actress_supporting": "Supp. Actress",
    "original_screenplay": "Orig. Screenplay",
    "animated_feature": "Animated Feature",
    "cinematography": "Cinematography",
}


def _save_fig(fig: matplotlib.figure.Figure, filename: str, *, dpi: int = 150) -> None:
    """Apply tight_layout, save to PLOTS_DIR, close, and print confirmation."""
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {filename}")


# ============================================================================
# Data loading
# ============================================================================


def load_results() -> dict[str, pd.DataFrame]:
    """Load all result CSVs into a dict."""
    dfs = {}
    for name in [
        "daily_pnl",
        "model_accuracy",
        "model_vs_market",
        "market_favorite_baseline",
        "spread_report",
    ]:
        path = RESULTS_DIR / f"{name}.csv"
        if path.exists():
            dfs[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {len(dfs[name])} rows")
        else:
            print(f"  WARNING: {path} not found")
    return dfs


# ============================================================================
# 1. Model accuracy heatmaps
# ============================================================================


def plot_model_accuracy_heatmaps(acc_df: pd.DataFrame) -> None:
    """Plot Brier score and winner rank heatmaps per model type.

    One heatmap per model: rows = categories, columns = snapshot dates.
    """
    model_types = sorted(acc_df["model_type"].unique())
    snapshot_dates = sorted(acc_df["snapshot_date"].unique())
    categories = sorted(acc_df["category"].unique())

    for metric, metric_label, cmap, vmin, vmax in [
        ("brier", "Brier Score", "YlOrRd", 0, 0.5),
        ("rank", "Winner Rank", "YlOrRd", 1, 5),
        ("winner_prob", "Winner Probability", "YlGn", 0, 1),
    ]:
        fig, axes = plt.subplots(
            1, len(model_types), figsize=(5 * len(model_types), 6), sharey=True
        )
        if len(model_types) == 1:
            axes = [axes]

        for ax, mt in zip(axes, model_types, strict=True):
            subset = acc_df[acc_df["model_type"] == mt]
            pivot = subset.pivot_table(
                values=metric, index="category", columns="snapshot_date", aggfunc="first"
            )
            # Reindex to ensure consistent order
            pivot = pivot.reindex(index=categories, columns=snapshot_dates)

            im = ax.imshow(
                pivot.values.astype(float),
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(MODEL_DISPLAY.get(mt, mt), fontsize=12)
            ax.set_xticks(range(len(snapshot_dates)))
            ax.set_xticklabels([d[5:] for d in snapshot_dates], rotation=45, ha="right", fontsize=8)
            if ax == axes[0]:
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels([CATEGORY_DISPLAY.get(c, c) for c in categories], fontsize=9)

            # Annotate cells
            for i in range(len(categories)):
                for j in range(len(snapshot_dates)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text = f"{val:.2f}" if metric != "rank" else f"{int(val)}"
                        ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

        fig.colorbar(im, ax=axes, shrink=0.6, label=metric_label)
        fig.suptitle(f"{metric_label} by Model × Category × Snapshot", fontsize=14)
        _save_fig(fig, f"model_{metric}_heatmap.png")


# ============================================================================
# 2. Classification vs trading performance
# ============================================================================


def plot_classification_vs_trading(acc_df: pd.DataFrame, pnl_df: pd.DataFrame) -> None:
    """Scatter plot: mean Brier (x) vs best P&L (y) per (model, category).

    Tests whether better classifiers produce better trading returns.
    """
    # Mean Brier per (model, category)
    brier_agg = acc_df.groupby(["model_type", "category"])["brier"].mean().reset_index()
    brier_agg.rename(columns={"brier": "mean_brier"}, inplace=True)

    # Best P&L per (model, category) — best trading config
    pnl_best = pnl_df.groupby(["model_type", "category"])["total_pnl"].max().reset_index()
    pnl_best.rename(columns={"total_pnl": "best_pnl"}, inplace=True)

    merged = brier_agg.merge(pnl_best, on=["model_type", "category"])

    # Save CSV
    merged.to_csv(RESULTS_DIR / "classification_vs_trading.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for mt in sorted(merged["model_type"].unique()):
        subset = merged[merged["model_type"] == mt]
        ax.scatter(
            subset["mean_brier"],
            subset["best_pnl"],
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
            s=80,
            alpha=0.8,
        )
        # Annotate each point with category
        for _, row in subset.iterrows():
            ax.annotate(
                CATEGORY_DISPLAY.get(row["category"], row["category"]),
                (row["mean_brier"], row["best_pnl"]),
                fontsize=7,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Brier Score (lower = better classifier)")
    ax.set_ylabel("Best P&L ($)")
    ax.set_title("Classification Quality vs Trading Performance")
    ax.legend()

    # Compute rank correlation
    if len(merged) >= 4:
        tau, p_val = stats.kendalltau(merged["mean_brier"], merged["best_pnl"])
        ax.text(
            0.02,
            0.02,
            f"Kendall τ = {tau:.3f} (p = {p_val:.3f})",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
        )

    _save_fig(fig, "brier_vs_pnl_scatter.png")


# ============================================================================
# 3. Model vs market divergence
# ============================================================================


def plot_model_vs_market(mvm_df: pd.DataFrame) -> None:
    """Plot model-market divergence for winners vs non-winners."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: divergence distribution by winner status with empirical percentiles
    ax = axes[0]
    winners = mvm_df[mvm_df["is_winner"]]
    non_winners = mvm_df[~mvm_df["is_winner"]]
    ax.hist(
        non_winners["divergence"],
        bins=30,
        alpha=0.5,
        label="Non-winners",
        color="gray",
    )
    ax.hist(
        winners["divergence"],
        bins=15,
        alpha=0.7,
        label="Winners",
        color="green",
    )
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    # Add empirical percentile markers for winner divergence
    if len(winners) > 0:
        import numpy as np

        p67 = np.percentile(winners["divergence"], 67)
        p90 = np.percentile(winners["divergence"], 90)
        ymax = ax.get_ylim()[1]
        ax.axvline(x=p67, color="blue", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.axvline(x=p90, color="red", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(
            p67,
            ymax * 0.92,
            f"P67={p67:.2f}",
            fontsize=7,
            color="blue",
            ha="left",
            fontweight="bold",
        )
        ax.text(
            p90,
            ymax * 0.85,
            f"P90={p90:.2f}",
            fontsize=7,
            color="red",
            ha="left",
            fontweight="bold",
        )

    ax.set_xlabel("Model - Market Probability")
    ax.set_ylabel("Count")
    ax.set_title("Model-Market Divergence: Winners vs Non-Winners")
    ax.legend()

    # Right: divergence over time for winners only
    ax = axes[1]
    for mt in sorted(winners["model_type"].unique()):
        subset = winners[winners["model_type"] == mt]
        by_date = subset.groupby("snapshot_date")["divergence"].mean()
        ax.plot(
            range(len(by_date)),
            by_date.values,
            marker="o",
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
        )
    ax.set_xticks(range(len(by_date)))
    ax.set_xticklabels([d[5:] for d in by_date.index], rotation=45, ha="right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Snapshot Date")
    ax.set_ylabel("Mean Divergence (Winners)")
    ax.set_title("Model Edge on Winners Over Time")
    ax.legend(fontsize=8)

    _save_fig(fig, "model_vs_market_divergence.png")


# ============================================================================
# 4. Per-precursor P&L breakdown
# ============================================================================


def analyze_per_precursor_pnl(
    mvm_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-snapshot P&L: if you trade once at each snapshot and hold to settlement.

    For a given recommended config, simulate single-snapshot entry at each snapshot date.
    Uses model_vs_market data to compute Kelly-sized positions and settlement P&L.

    This answers: "I'm at Feb 22, 2026 (post-BAFTA/DGA). If I enter now with the
    recommended config, what P&L should I expect based on 2025 data?"

    The simulation for each snapshot:
    1. Look at model probabilities and market prices for all nominees
    2. Compute edge = model_prob - market_prob (for YES) or (1-model_prob) - (1-market_prob) (NO)
    3. Apply Kelly sizing with the recommended config's parameters
    4. Compute settlement: winners pay $1/contract, losers pay $0
    5. Net P&L = settlement proceeds - entry cost - fees

    Returns per-snapshot P&L DataFrame with columns:
        category, model_type, snapshot_date, event_label, n_positions,
        entry_cost, settlement_value, fees, pnl
    """
    from oscar_prediction_market.trading.schema import (
        KellyMode,
        PositionDirection,
    )

    snapshot_info = get_post_nomination_snapshot_dates(CALENDAR)
    snap_labels = {str(d): " + ".join(e) for d, e in snapshot_info}

    # Use the recommended config from best_config.json if available, else defaults
    best_config_path = RESULTS_DIR / "best_config.json"
    if best_config_path.exists():
        with open(best_config_path) as f:
            best_cfg = json.load(f)
        kf = best_cfg.get("kelly_fraction", 0.25)
        bet = best_cfg.get("buy_edge_threshold", 0.08)
        km = best_cfg.get("kelly_mode", "independent")
        rec_model = best_cfg.get("model_type", "lr")
        fee_type_str = best_cfg.get("fee_type", "maker")
        trading_side_str = best_cfg.get("trading_side", "yes")
    else:
        kf, bet, km, rec_model = 0.25, 0.08, "independent", "lr"
        fee_type_str, trading_side_str = "maker", "yes"

    fee_rate = 0.0 if fee_type_str == "maker" else 0.07
    _kelly_mode = KellyMode(km) if isinstance(km, str) else km  # noqa: F841
    _SIDE_TO_DIRS: dict[str, frozenset[PositionDirection]] = {
        "yes": frozenset({PositionDirection.YES}),
        "no": frozenset({PositionDirection.NO}),
        "all": frozenset({PositionDirection.YES, PositionDirection.NO}),
    }
    allowed_dirs = _SIDE_TO_DIRS[trading_side_str]
    bankroll = 1000.0

    categories = sorted(mvm_df["category"].unique())
    models = sorted(mvm_df["model_type"].unique())
    dates = sorted(mvm_df["snapshot_date"].unique())

    rows = []
    for cat in categories:
        for mt in models:
            for snap_date in dates:
                sub = mvm_df[
                    (mvm_df["category"] == cat)
                    & (mvm_df["model_type"] == mt)
                    & (mvm_df["snapshot_date"] == snap_date)
                ]
                if sub.empty:
                    continue

                # Compute edges per nominee per direction
                edges = []
                for _, nom_row in sub.iterrows():
                    model_prob = nom_row["model_prob"]
                    market_prob = nom_row["market_prob"]
                    nominee = nom_row["nominee"]
                    is_winner = nom_row["is_winner"]

                    # YES edge
                    if PositionDirection.YES in allowed_dirs:
                        yes_price = market_prob
                        if yes_price > 0:
                            yes_edge = model_prob - market_prob
                            if yes_edge >= bet:
                                edges.append(
                                    {
                                        "outcome": nominee,
                                        "direction": PositionDirection.YES,
                                        "edge": yes_edge,
                                        "model_prob": model_prob,
                                        "price": yes_price,
                                        "is_winner": is_winner,
                                    }
                                )

                    # NO edge
                    if PositionDirection.NO in allowed_dirs:
                        no_price = 1.0 - market_prob
                        if no_price > 0:
                            no_model_prob = 1.0 - model_prob
                            no_edge = no_model_prob - (1 - market_prob)
                            if no_edge >= bet:
                                edges.append(
                                    {
                                        "outcome": nominee,
                                        "direction": PositionDirection.NO,
                                        "edge": no_edge,
                                        "model_prob": no_model_prob,
                                        "price": no_price,
                                        "is_winner": is_winner,
                                    }
                                )

                if not edges:
                    rows.append(
                        {
                            "category": cat,
                            "model_type": mt,
                            "snapshot_date": snap_date,
                            "event_label": snap_labels.get(snap_date, ""),
                            "n_positions": 0,
                            "entry_cost": 0.0,
                            "settlement_value": 0.0,
                            "fees": 0.0,
                            "pnl": 0.0,
                        }
                    )
                    continue

                # Simple Kelly sizing: contracts = kf * bankroll * edge / price
                total_cost = 0.0
                total_settlement = 0.0
                total_fees = 0.0
                n_pos = 0

                for e in edges:
                    price = e["price"]
                    # Kelly fraction of edge
                    contracts = kf * bankroll * e["edge"] / price
                    contracts = max(0, contracts)
                    if contracts < 0.5:
                        continue

                    cost = contracts * price
                    fee = cost * fee_rate
                    total_cost += cost + fee
                    total_fees += fee
                    n_pos += 1

                    # Settlement
                    if e["direction"] == PositionDirection.YES:
                        if e["is_winner"]:
                            total_settlement += contracts * 1.0  # $1 per contract
                        # else: $0
                    else:  # NO
                        if not e["is_winner"]:
                            total_settlement += contracts * 1.0  # NO wins
                        # else: $0

                pnl = total_settlement - total_cost
                rows.append(
                    {
                        "category": cat,
                        "model_type": mt,
                        "snapshot_date": snap_date,
                        "event_label": snap_labels.get(snap_date, ""),
                        "n_positions": n_pos,
                        "entry_cost": round(total_cost, 2),
                        "settlement_value": round(total_settlement, 2),
                        "fees": round(total_fees, 2),
                        "pnl": round(pnl, 2),
                    }
                )

    precursor_df = pd.DataFrame(rows)
    precursor_df.to_csv(RESULTS_DIR / "per_precursor_pnl.csv", index=False)
    print(f"  Saved per_precursor_pnl.csv ({len(precursor_df)} rows)")

    # --- Plot: per-snapshot P&L for recommended model ---
    rec_data = precursor_df[precursor_df["model_type"] == rec_model]
    if not rec_data.empty:
        cats_plot = sorted(rec_data["category"].unique())
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(dates))
        width = 0.8 / max(len(cats_plot), 1)

        for i, cat in enumerate(cats_plot):
            cat_data = rec_data[rec_data["category"] == cat].set_index("snapshot_date")
            pnls = [cat_data.loc[d, "pnl"] if d in cat_data.index else 0 for d in dates]
            ax.bar(
                x + i * width,
                pnls,
                width=width,
                label=CATEGORY_DISPLAY.get(cat, cat),
                alpha=0.8,
            )

        ax.set_xticks(x + width * len(cats_plot) / 2)
        labels = []
        for d in dates:
            event = snap_labels.get(d, d[5:])
            labels.append(f"{d[5:]}\n{event[:30]}")
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("P&L ($)")
        ax.set_title(
            f"Per-Snapshot Entry P&L — {MODEL_DISPLAY.get(rec_model, rec_model)} "
            f"(kf={kf}, edge≥{bet}, {km})",
            fontsize=12,
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
        _save_fig(fig, "per_precursor_pnl.png")

    return precursor_df


# ============================================================================
# 5. Return decomposition
# ============================================================================


def analyze_return_decomposition(pnl_df: pd.DataFrame) -> None:
    """Decompose returns into selection, timing, and sizing components.

    Selection alpha: model picks vs equal-weight baseline
    Timing alpha: late vs early buying
    Sizing alpha: Kelly sizing vs equal-weight sizing

    Saves decomposition CSV.
    """
    # Group by model_type: compare configs with different parameters
    decomp_rows = []

    for mt in sorted(pnl_df["model_type"].unique()):
        mt_df = pnl_df[pnl_df["model_type"] == mt]

        # Best config P&L
        best_idx = mt_df["total_pnl"].idxmax()
        best_pnl = mt_df.loc[best_idx, "total_pnl"]

        # Compare kelly fractions (sizing alpha)
        for cat in sorted(mt_df["category"].unique()):
            cat_df = mt_df[mt_df["category"] == cat]
            if cat_df.empty:
                continue

            # Mean P&L across all configs (sizing-neutral)
            mean_pnl = cat_df["total_pnl"].mean()

            # Compare kelly_fraction=0.10 vs 0.25 (sizing effect)
            kf_low = cat_df[cat_df["kelly_fraction"] == 0.10]["total_pnl"].mean()
            kf_high = cat_df[cat_df["kelly_fraction"] == 0.25]["total_pnl"].mean()

            # Compare buy_edge_threshold (selection timing sensitivity)
            bet_low = cat_df[cat_df["buy_edge_threshold"] == 0.05]["total_pnl"].mean()
            bet_high = cat_df[cat_df["buy_edge_threshold"] == 0.15]["total_pnl"].mean()

            decomp_rows.append(
                {
                    "model_type": mt,
                    "category": cat,
                    "best_pnl": round(best_pnl, 2),
                    "mean_pnl": round(mean_pnl, 2),
                    "sizing_alpha": round(kf_high - kf_low, 2),
                    "selectivity_alpha": round(bet_low - bet_high, 2),
                    "n_configs": len(cat_df),
                }
            )

    if decomp_rows:
        decomp_df = pd.DataFrame(decomp_rows)
        decomp_df.to_csv(RESULTS_DIR / "return_decomposition.csv", index=False)
        print(f"  Saved return_decomposition.csv ({len(decomp_rows)} rows)")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        cats = sorted(decomp_df["category"].unique())
        x = np.arange(len(cats))
        width = 0.2

        for i, mt in enumerate(sorted(decomp_df["model_type"].unique())):
            subset = decomp_df[decomp_df["model_type"] == mt].set_index("category")
            subset = subset.reindex(cats)
            ax.bar(
                x + i * width,
                subset["sizing_alpha"].fillna(0),
                width,
                label=f"{MODEL_DISPLAY.get(mt, mt)} (sizing)",
                color=get_model_color(mt),
                alpha=0.7,
            )

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c) for c in cats], rotation=45, ha="right")
        ax.set_ylabel("Sizing Alpha ($)")
        ax.set_title("Kelly Sizing Effect (kf=0.25 - kf=0.10)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        _save_fig(fig, "return_decomposition.png")


# ============================================================================
# 6. Trading parameter grid analysis
# ============================================================================


def analyze_trading_grid(pnl_df: pd.DataFrame) -> None:
    """Analyze sensitivity to each trading parameter.

    For each parameter, show mean P&L across all other parameter values.
    """
    params = ["fee_type", "kelly_fraction", "buy_edge_threshold", "min_price", "kelly_mode"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        grouped = pnl_df.groupby(param)["total_pnl"].agg(["mean", "std", "count"]).reset_index()
        ax.bar(
            range(len(grouped)),
            grouped["mean"],
            yerr=grouped["std"] / np.sqrt(grouped["count"]),
            capsize=3,
            color=["#3274a1", "#e1812c", "#3a923a", "#c03d3e", "#9372b2"][: len(grouped)],
            alpha=0.7,
        )
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped[param].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Mean P&L ($)")
        ax.set_title(param.replace("_", " ").title())
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Remove extra subplot
    axes[-1].set_visible(False)

    fig.suptitle("Trading Parameter Sensitivity", fontsize=14)
    _save_fig(fig, "config_robustness_heatmap.png")


# ============================================================================
# 7. P&L by category
# ============================================================================


def plot_pnl_by_category(pnl_df: pd.DataFrame) -> None:
    """Show best P&L per category per model type."""
    best = pnl_df.groupby(["model_type", "category"])["total_pnl"].max().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    cats = sorted(best["category"].unique())
    x = np.arange(len(cats))
    model_types = sorted(best["model_type"].unique())
    width = 0.8 / len(model_types)

    for i, mt in enumerate(model_types):
        subset = best[best["model_type"] == mt].set_index("category")
        subset = subset.reindex(cats)
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
    ax.set_ylabel("Best P&L ($)")
    ax.set_title("Best Trading P&L by Category × Model")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    _save_fig(fig, "pnl_by_category.png")


# ============================================================================
# 8. Spread & liquidity report
# ============================================================================


def plot_spread_report(spread_df: pd.DataFrame) -> None:
    """Plot spread and liquidity metrics per category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Median spread
    ax = axes[0]
    cats = spread_df["category"]
    spread_col = "median_spread" if "median_spread" in spread_df.columns else "mean_spread"
    ax.barh(range(len(cats)), spread_df[spread_col], color="#3274a1", alpha=0.7)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([CATEGORY_DISPLAY.get(c, c) for c in cats])
    ax.set_xlabel("Median One-Way Spread ($)")
    ax.set_title("Spread by Category")

    # Trade count
    ax = axes[1]
    ax.barh(range(len(cats)), spread_df["total_trades"], color="#e1812c", alpha=0.7)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([CATEGORY_DISPLAY.get(c, c) for c in cats])
    ax.set_xlabel("Total Trades")
    ax.set_title("Liquidity by Category")

    fig.suptitle("Market Microstructure: 2025 Oscar Markets", fontsize=14)
    _save_fig(fig, "spread_liquidity.png")


# ============================================================================
# 9. Best config selection
# ============================================================================


def select_best_config(pnl_df: pd.DataFrame) -> None:
    """Select best (model_type, trading_config) for 2026 pre-commitment.

    Ranks by aggregate P&L across all categories. Saves best config
    and runner-ups to JSON.
    """
    # Aggregate P&L per (model_type, config_label)
    groupby_cols = [
        "model_type",
        "config_label",
        "fee_type",
        "kelly_fraction",
        "buy_edge_threshold",
        "min_price",
        "kelly_mode",
    ]
    if "bankroll_mode" in pnl_df.columns:
        groupby_cols.append("bankroll_mode")
    if "trading_side" in pnl_df.columns:
        groupby_cols.append("trading_side")

    agg = (
        pnl_df.groupby(groupby_cols)
        .agg(
            total_pnl=("total_pnl", "sum"),
            mean_pnl=("total_pnl", "mean"),
            n_categories=("category", "nunique"),
            total_trades=("total_trades", "sum"),
            n_negative=("total_pnl", lambda x: (x < 0).sum()),
        )
        .reset_index()
    )
    agg = agg.sort_values("total_pnl", ascending=False)

    # Best config
    best = agg.iloc[0]
    best_config = {
        "model_type": best["model_type"],
        "fee_type": best["fee_type"],
        "kelly_fraction": float(best["kelly_fraction"]),
        "buy_edge_threshold": float(best["buy_edge_threshold"]),
        "min_price": float(best["min_price"]),
        "kelly_mode": best["kelly_mode"],
        "trading_side": best.get("trading_side", "yes"),
        "bankroll_mode": best.get("bankroll_mode", "fixed"),
        "aggregate_pnl_2025": round(float(best["total_pnl"]), 2),
        "mean_pnl_per_category": round(float(best["mean_pnl"]), 2),
        "n_categories_traded": int(best["n_categories"]),
        "n_categories_negative": int(best["n_negative"]),
        "total_trades": int(best["total_trades"]),
    }

    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Best config: {best['model_type']} | {best['config_label']}")
    print(f"    Aggregate P&L: ${best['total_pnl']:.2f}")

    # Runner-ups (top 10)
    runners = []
    for _, row in agg.head(10).iterrows():
        runners.append(
            {
                "model_type": row["model_type"],
                "fee_type": row["fee_type"],
                "kelly_fraction": float(row["kelly_fraction"]),
                "buy_edge_threshold": float(row["buy_edge_threshold"]),
                "min_price": float(row["min_price"]),
                "kelly_mode": row["kelly_mode"],
                "trading_side": row.get("trading_side", "yes"),
                "bankroll_mode": row.get("bankroll_mode", "fixed"),
                "aggregate_pnl_2025": round(float(row["total_pnl"]), 2),
                "mean_pnl_per_category": round(float(row["mean_pnl"]), 2),
                "n_categories_negative": int(row["n_negative"]),
            }
        )

    with open(RESULTS_DIR / "runner_up_configs.json", "w") as f:
        json.dump(runners, f, indent=2)
    print(f"  Saved runner_up_configs.json ({len(runners)} configs)")


# ============================================================================
# 9. Timing analysis — P&L by entry date
# ============================================================================


def plot_timing_analysis(
    mvm_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
) -> None:
    """Analyse how entry timing affects P&L.

    Two views:
    (b) Per-contract P&L on the winner at each day's market price
        (best config + a conservative config per model×category)
    (c) Heatmap: final P&L × entry_date × model (using model_vs_market data
        to compute winner profit per contract: $1.00 − market_price)

    The per-contract approach is clean: you buy 1 YES-contract on the winner
    at that date's market price. Payout = $1. Profit = $1 − price.
    """
    winners = mvm_df[mvm_df["is_winner"]].copy()

    if winners.empty:
        print("  No winner data for timing analysis")
        return

    # ── (c) Heatmap: profit-per-contract if buying the winner on each date ──
    categories = sorted(winners["category"].unique())
    models = sorted(winners["model_type"].unique())
    dates = sorted(winners["snapshot_date"].unique())

    # Winner market price by date × category
    # (market_prob is a fraction; price in cents = market_prob × 100)
    fig, axes = plt.subplots(len(categories), 1, figsize=(14, 4 * len(categories)), sharex=True)
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories, strict=True):
        cat_winners = winners[winners["category"] == cat]

        # One line per model: winner model_prob and market_prob over time
        for mt in models:
            sub = cat_winners[cat_winners["model_type"] == mt].sort_values("snapshot_date")
            if sub.empty:
                continue
            # Plot model-implied edge: model_prob − market_prob
            # Positive = model thinks winner is underpriced
            ax.plot(
                range(len(sub)),
                (sub["model_prob"] - sub["market_prob"]) * 100,
                marker="o",
                label=f"{MODEL_DISPLAY.get(mt, mt)} edge",
                color=get_model_color(mt),
                linewidth=2,
            )

        # Also plot per-contract profit on secondary axis
        ax2 = ax.twinx()
        # Use first model's data for market_prob (same for all)
        first_mt = cat_winners["model_type"].iloc[0]
        market_data = cat_winners[cat_winners["model_type"] == first_mt].sort_values(
            "snapshot_date"
        )
        profit = 100 - market_data["market_prob"].values * 100
        ax2.fill_between(
            range(len(market_data)),
            profit,
            alpha=0.15,
            color="green" if profit.mean() > 0 else "red",
        )
        ax2.plot(
            range(len(market_data)),
            profit,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Per-contract profit (¢)",
        )
        ax2.set_ylabel("Profit per contract (¢)", fontsize=9)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Model Edge (pp)", fontsize=10)
        ax.set_title(f"{CATEGORY_DISPLAY.get(cat, cat)}", fontsize=12)
        ax.set_xticks(range(len(market_data)))
        ax.set_xticklabels(
            [d[5:] for d in market_data["snapshot_date"]], rotation=45, ha="right", fontsize=8
        )

        # Annotate precursor events
        snap_info = get_post_nomination_snapshot_dates(CALENDAR)
        snap_map = {str(d): ", ".join(e) for d, e in snap_info}
        for i, d in enumerate(market_data["snapshot_date"]):
            if d in snap_map:
                ax.annotate(
                    snap_map[d],
                    xy=(i, ax.get_ylim()[1]),
                    fontsize=6,
                    color="purple",
                    ha="center",
                    va="bottom",
                    rotation=30,
                )

        if ax == axes[0]:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle("Timing Analysis: Model Edge & Per-Contract Profit Over Season", fontsize=14)
    _save_fig(fig, "timing_analysis.png")

    # ── Entry date heatmap: final P&L × entry_date × model ──
    # For each date: "if I bought 1 contract on the winner at market price,
    # what is my P&L at settlement?"
    fig, ax = plt.subplots(figsize=(14, 6))
    for cat in categories:
        cat_data = winners[
            (winners["category"] == cat) & (winners["model_type"] == winners["model_type"].iloc[0])
        ].sort_values("snapshot_date")
        if cat_data.empty:
            continue
        profit = (1.0 - cat_data["market_prob"].values) * 100  # cents profit
        ax.plot(
            range(len(cat_data)),
            profit,
            marker="s",
            label=CATEGORY_DISPLAY.get(cat, cat),
            linewidth=2,
        )

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right")
    ax.set_xlabel("Entry Date")
    ax.set_ylabel("Profit per Contract (¢)")
    ax.set_title("Per-Contract Profit on Winner by Entry Date")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    _save_fig(fig, "entry_date_profit.png")

    # Save timing data as CSV
    timing_rows = []
    for cat in categories:
        cat_data = winners[(winners["category"] == cat)].sort_values(
            ["model_type", "snapshot_date"]
        )
        for _, r in cat_data.iterrows():
            timing_rows.append(
                {
                    "category": r["category"],
                    "model_type": r["model_type"],
                    "snapshot_date": r["snapshot_date"],
                    "winner": r["nominee"],
                    "model_prob": round(r["model_prob"], 4),
                    "market_prob": round(r["market_prob"], 4),
                    "edge_pp": round((r["model_prob"] - r["market_prob"]) * 100, 2),
                    "profit_per_contract": round((1.0 - r["market_prob"]) * 100, 2),
                }
            )
    if timing_rows:
        pd.DataFrame(timing_rows).to_csv(RESULTS_DIR / "timing_analysis.csv", index=False)
        print(f"  Saved timing_analysis.csv ({len(timing_rows)} rows)")


# ============================================================================
# 10. Detailed model performance analysis
# ============================================================================


def plot_probability_trajectories(mvm_df: pd.DataFrame) -> None:
    """Plot model probability vs market price per nominee per category.

    One figure per category. Each subplot is a model type.
    Lines = model prob for each nominee. Dashed = market price.
    Winner nominee highlighted.
    """
    categories = sorted(mvm_df["category"].unique())
    models = sorted(mvm_df["model_type"].unique())

    for cat in categories:
        cat_data = mvm_df[mvm_df["category"] == cat]
        nominees = sorted(cat_data["nominee"].unique())
        dates = sorted(cat_data["snapshot_date"].unique())
        n_models = len([m for m in models if not cat_data[cat_data["model_type"] == m].empty])
        if n_models == 0:
            continue

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        axidx = 0
        for mt in models:
            mt_data = cat_data[cat_data["model_type"] == mt]
            if mt_data.empty:
                continue
            ax = axes[axidx]
            axidx += 1

            for nominee in nominees:
                nom_data = mt_data[mt_data["nominee"] == nominee].sort_values("snapshot_date")
                if nom_data.empty:
                    continue
                is_winner = nom_data["is_winner"].iloc[0]
                alpha = 1.0 if is_winner else 0.3
                lw = 2.5 if is_winner else 1.0

                # Model probability
                ax.plot(
                    range(len(nom_data)),
                    nom_data["model_prob"],
                    marker="o" if is_winner else None,
                    linewidth=lw,
                    alpha=alpha,
                    label=f"{nominee[:20]} (model)" if is_winner else None,
                    markersize=4,
                )
                # Market probability (dashed)
                ax.plot(
                    range(len(nom_data)),
                    nom_data["market_prob"],
                    linestyle="--",
                    linewidth=lw,
                    alpha=alpha * 0.6,
                    label=f"{nominee[:20]} (market)" if is_winner else None,
                    markersize=3,
                )

            ax.set_title(MODEL_DISPLAY.get(mt, mt), fontsize=11)
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Probability" if axidx == 1 else "")
            ax.set_ylim(-0.02, 1.02)
            if is_winner:
                ax.legend(fontsize=7)

        cat_display = CATEGORY_DISPLAY.get(cat, cat)
        fig.suptitle(f"Probability Trajectories — {cat_display}", fontsize=13)
        _save_fig(fig, f"prob_trajectories_{cat}.png")


def plot_per_outcome_trajectories(mvm_df: pd.DataFrame) -> None:
    """Plot per-outcome probability trajectories (Option B).

    One figure per category. Each subplot is a single nominee, showing all 4
    model_prob lines (one per model type, solid coloured) plus market_prob
    (black dashed). This makes individual nominees readable and reveals
    model agreement / disagreement per outcome.

    Winner subplot gets a gold border.
    """
    categories = sorted(mvm_df["category"].unique())
    models = sorted(mvm_df["model_type"].unique())

    snap_info = get_post_nomination_snapshot_dates(CALENDAR)
    _snap_labels_short = {str(d): " + ".join(e)[:20] for d, e in snap_info}  # noqa: F841

    for cat in categories:
        cat_data = mvm_df[mvm_df["category"] == cat]
        nominees = sorted(cat_data["nominee"].unique())
        dates = sorted(cat_data["snapshot_date"].unique())
        n_nom = len(nominees)
        if n_nom == 0:
            continue

        # Layout: 2 columns, enough rows
        ncols = min(3, n_nom)
        nrows = (n_nom + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), sharey=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx, nominee in enumerate(nominees):
            row_i, col_i = divmod(idx, ncols)
            ax = axes[row_i, col_i]

            nom_data = cat_data[cat_data["nominee"] == nominee]
            is_winner = nom_data["is_winner"].iloc[0] if not nom_data.empty else False

            # Plot each model's probability
            for mt in models:
                mt_nom = nom_data[nom_data["model_type"] == mt].sort_values("snapshot_date")
                if mt_nom.empty:
                    continue
                ax.plot(
                    range(len(mt_nom)),
                    mt_nom["model_prob"],
                    marker="o",
                    color=get_model_color(mt),
                    linewidth=2,
                    markersize=4,
                    label=MODEL_DISPLAY.get(mt, mt),
                )

            # Market price (use any model's row — market_prob is same across models)
            first_mt = nom_data["model_type"].iloc[0]
            mkt = nom_data[nom_data["model_type"] == first_mt].sort_values("snapshot_date")
            ax.plot(
                range(len(mkt)),
                mkt["market_prob"],
                linestyle="--",
                color="black",
                linewidth=2,
                alpha=0.6,
                label="Market",
            )

            # Formatting
            title = nominee[:25]
            if is_winner:
                title += " ★"
                for spine in ax.spines.values():
                    spine.set_edgecolor("#DAA520")
                    spine.set_linewidth(2.5)

            ax.set_title(title, fontsize=10, fontweight="bold" if is_winner else "normal")
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=7)
            ax.set_ylim(-0.02, 1.02)
            if col_i == 0:
                ax.set_ylabel("Probability")
            if idx == 0:
                ax.legend(fontsize=6, ncol=2, loc="upper left")

        # Hide unused subplots
        for idx in range(n_nom, nrows * ncols):
            row_i, col_i = divmod(idx, ncols)
            axes[row_i, col_i].set_visible(False)

        cat_display = CATEGORY_DISPLAY.get(cat, cat)
        fig.suptitle(f"Per-Outcome Trajectories — {cat_display}", fontsize=14)
        _save_fig(fig, f"per_outcome_trajectories_{cat}.png")


def plot_divergence_heatmaps(mvm_df: pd.DataFrame) -> None:
    """Heatmap of model−market divergence, one per model×category.

    Rows = nominees, columns = snapshot dates. Color = model − market prob.
    Positive (green) = model is more bullish than market.
    """
    categories = sorted(mvm_df["category"].unique())
    models = sorted(mvm_df["model_type"].unique())

    for cat in categories:
        cat_data = mvm_df[mvm_df["category"] == cat]
        active_models = [m for m in models if not cat_data[cat_data["model_type"] == m].empty]
        n = len(active_models)
        if n == 0:
            continue

        nominees = sorted(cat_data["nominee"].unique())
        dates = sorted(cat_data["snapshot_date"].unique())

        fig, axes = plt.subplots(1, n, figsize=(6 * n, max(4, 0.5 * len(nominees))), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, mt in zip(axes, active_models, strict=True):
            mt_data = cat_data[cat_data["model_type"] == mt]
            pivot = mt_data.pivot_table(
                values="divergence", index="nominee", columns="snapshot_date", aggfunc="first"
            )
            pivot = pivot.reindex(index=nominees, columns=dates)

            vmax = max(0.2, abs(pivot.values[~np.isnan(pivot.values)]).max()) if pivot.size else 0.2
            im = ax.imshow(
                pivot.values.astype(float),
                cmap="RdYlGn",
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(MODEL_DISPLAY.get(mt, mt), fontsize=11)
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
            if ax == axes[0]:
                ax.set_yticks(range(len(nominees)))
                # Mark winner with ★
                labels = []
                for nom in nominees:
                    is_w = mt_data[mt_data["nominee"] == nom]["is_winner"].any()
                    labels.append(f"★ {nom[:25]}" if is_w else nom[:25])
                ax.set_yticklabels(labels, fontsize=8)

            # Annotate
            for i in range(len(nominees)):
                for j in range(len(dates)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6, color="black"
                        )

        fig.colorbar(im, ax=axes, shrink=0.6, label="Model − Market Prob")
        cat_display = CATEGORY_DISPLAY.get(cat, cat)
        fig.suptitle(f"Model−Market Divergence — {cat_display}", fontsize=13)
        _save_fig(fig, f"divergence_heatmap_{cat}.png")


def plot_brier_evolution(acc_df: pd.DataFrame) -> None:
    """Plot Brier score evolution over snapshots per model×category.

    One subplot per category, lines per model. Annotate precursor events.
    """
    categories = sorted(acc_df["category"].unique())
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    snap_info = get_post_nomination_snapshot_dates(CALENDAR)
    snap_labels = {str(d): ", ".join(e[:3] for e in evts) for d, evts in snap_info}

    for ax, cat in zip(axes, categories, strict=True):
        cat_data = acc_df[acc_df["category"] == cat]
        models = sorted(cat_data["model_type"].unique())
        dates = sorted(cat_data["snapshot_date"].unique())

        for mt in models:
            mt_data = cat_data[cat_data["model_type"] == mt].sort_values("snapshot_date")
            if mt_data.empty:
                continue
            ax.plot(
                range(len(mt_data)),
                mt_data["brier"],
                marker="o",
                label=MODEL_DISPLAY.get(mt, mt),
                color=get_model_color(mt),
                linewidth=2,
            )

        ax.set_xticks(range(len(dates)))
        xlabels = []
        for d in dates:
            label = d[5:]
            if d in snap_labels:
                label += f"\n({snap_labels[d]})"
            xlabels.append(label)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Brier Score" if ax == axes[0] else "")
        ax.set_title(CATEGORY_DISPLAY.get(cat, cat), fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle("Brier Score Evolution Over Precursor Events", fontsize=13)
    _save_fig(fig, "brier_evolution.png")


def plot_winner_rank_evolution(acc_df: pd.DataFrame) -> None:
    """Plot winner rank evolution (1=top pick) over snapshots."""
    categories = sorted(acc_df["category"].unique())
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories, strict=True):
        cat_data = acc_df[acc_df["category"] == cat]
        models = sorted(cat_data["model_type"].unique())

        for mt in models:
            mt_data = cat_data[cat_data["model_type"] == mt].sort_values("snapshot_date")
            if mt_data.empty or mt_data["rank"].isna().all():
                continue
            ax.plot(
                range(len(mt_data)),
                mt_data["rank"],
                marker="o",
                label=MODEL_DISPLAY.get(mt, mt),
                color=get_model_color(mt),
                linewidth=2,
            )

        dates = sorted(cat_data["snapshot_date"].unique())
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Winner Rank" if ax == axes[0] else "")
        ax.set_title(CATEGORY_DISPLAY.get(cat, cat), fontsize=11)
        ax.invert_yaxis()
        ax.axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Rank 1")
        ax.legend(fontsize=7)

    fig.suptitle("Winner Rank Over Season (1 = Model's Top Pick)", fontsize=13)
    _save_fig(fig, "winner_rank_evolution.png")


def plot_probability_concentration(mvm_df: pd.DataFrame) -> None:
    """Compare probability concentration across models and market.

    For each (category, snapshot), compute entropy of model predictions
    vs entropy of market prices. Higher entropy = more diffuse/uncertain.
    """
    categories = sorted(mvm_df["category"].unique())
    models = sorted(mvm_df["model_type"].unique())
    dates = sorted(mvm_df["snapshot_date"].unique())

    def entropy(probs: np.ndarray) -> float:
        p = probs[probs > 0]
        return float(-np.sum(p * np.log2(p)))

    rows = []
    for cat in categories:
        for date in dates:
            cat_date = mvm_df[(mvm_df["category"] == cat) & (mvm_df["snapshot_date"] == date)]
            if cat_date.empty:
                continue
            # Market entropy (same across models)
            first_mt = cat_date["model_type"].iloc[0]
            market_probs = cat_date[cat_date["model_type"] == first_mt]["market_prob"].values
            market_norm = (
                market_probs / market_probs.sum() if market_probs.sum() > 0 else market_probs
            )
            h_market = entropy(market_norm)

            for mt in models:
                mt_data = cat_date[cat_date["model_type"] == mt]
                if mt_data.empty:
                    continue
                model_probs = mt_data["model_prob"].values
                model_norm = (
                    model_probs / model_probs.sum() if model_probs.sum() > 0 else model_probs
                )
                h_model = entropy(model_norm)
                rows.append(
                    {
                        "category": cat,
                        "snapshot_date": date,
                        "model_type": mt,
                        "model_entropy": round(h_model, 4),
                        "market_entropy": round(h_market, 4),
                        "entropy_diff": round(h_model - h_market, 4),
                    }
                )

    if not rows:
        return

    conc_df = pd.DataFrame(rows)
    conc_df.to_csv(RESULTS_DIR / "probability_concentration.csv", index=False)

    # Plot: entropy over time per model, one subplot per category
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories, strict=True):
        cat_data = conc_df[conc_df["category"] == cat]

        # Market entropy
        market_ent = cat_data.groupby("snapshot_date")["market_entropy"].first()
        ax.plot(
            range(len(market_ent)),
            market_ent.values,
            marker="x",
            color="black",
            linewidth=2,
            label="Market",
            linestyle="--",
        )

        for mt in models:
            mt_data = cat_data[cat_data["model_type"] == mt].sort_values("snapshot_date")
            if mt_data.empty:
                continue
            ax.plot(
                range(len(mt_data)),
                mt_data["model_entropy"],
                marker="o",
                label=MODEL_DISPLAY.get(mt, mt),
                color=get_model_color(mt),
                linewidth=2,
            )

        dates_ordered = sorted(cat_data["snapshot_date"].unique())
        ax.set_xticks(range(len(dates_ordered)))
        ax.set_xticklabels([d[5:] for d in dates_ordered], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Entropy (bits)" if ax == axes[0] else "")
        ax.set_title(CATEGORY_DISPLAY.get(cat, cat), fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle("Prediction Entropy: Model vs Market (Higher = More Uncertain)", fontsize=13)
    _save_fig(fig, "probability_concentration.png")


def plot_nominee_correlation(mvm_df: pd.DataFrame) -> None:
    """Per-nominee correlation between model and market across snapshots.

    For each (category, model, nominee), compute Pearson r and MAE
    between model_prob and market_prob over time.
    """
    rows = []
    for (cat, mt, nom), group in mvm_df.groupby(["category", "model_type", "nominee"]):
        if len(group) < 3:
            continue
        r_val = group["model_prob"].corr(group["market_prob"])
        mae = (group["model_prob"] - group["market_prob"]).abs().mean()
        is_winner = group["is_winner"].iloc[0]
        rows.append(
            {
                "category": cat,
                "model_type": mt,
                "nominee": nom,
                "pearson_r": round(r_val, 4) if not np.isnan(r_val) else None,
                "mae": round(mae, 4),
                "n_snapshots": len(group),
                "is_winner": is_winner,
            }
        )

    if not rows:
        return

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(RESULTS_DIR / "nominee_correlation.csv", index=False)
    print(f"  Saved nominee_correlation.csv ({len(rows)} rows)")

    # Plot: scatter of Pearson r vs MAE, colored by model, shaped by winner
    fig, ax = plt.subplots(figsize=(10, 7))
    for mt in sorted(corr_df["model_type"].unique()):
        subset = corr_df[corr_df["model_type"] == mt]
        winners = subset[subset["is_winner"]]
        nonwinners = subset[~subset["is_winner"]]
        ax.scatter(
            nonwinners["pearson_r"],
            nonwinners["mae"],
            color=get_model_color(mt),
            alpha=0.4,
            s=30,
        )
        ax.scatter(
            winners["pearson_r"],
            winners["mae"],
            color=get_model_color(mt),
            alpha=1.0,
            s=100,
            marker="*",
            edgecolors="black",
            label=f"{MODEL_DISPLAY.get(mt, mt)}",
        )

    ax.set_xlabel("Pearson r (model vs market)")
    ax.set_ylabel("MAE (model vs market)")
    ax.set_title("Per-Nominee Model-Market Agreement")
    ax.legend(fontsize=9)
    _save_fig(fig, "nominee_correlation.png")


def plot_model_agreement(mvm_df: pd.DataFrame) -> None:
    """Cross-model agreement: do models agree on the top pick?

    For each (category, snapshot), get each model's top pick.
    Show agreement rate and which nominees are top-picked.
    """
    categories = sorted(mvm_df["category"].unique())
    models = sorted(mvm_df["model_type"].unique())
    dates = sorted(mvm_df["snapshot_date"].unique())

    rows = []
    for cat in categories:
        for date in dates:
            picks = {}
            for mt in models:
                sub = mvm_df[
                    (mvm_df["category"] == cat)
                    & (mvm_df["snapshot_date"] == date)
                    & (mvm_df["model_type"] == mt)
                ]
                if sub.empty:
                    continue
                top = sub.loc[sub["model_prob"].idxmax()]
                picks[mt] = top["nominee"]

            if len(picks) < 2:
                continue

            top_picks = list(picks.values())
            agreement_rate = max(top_picks.count(p) for p in set(top_picks)) / len(top_picks)
            market_sub = mvm_df[(mvm_df["category"] == cat) & (mvm_df["snapshot_date"] == date)]
            if not market_sub.empty:
                market_top = market_sub.loc[market_sub["market_prob"].idxmax(), "nominee"]
            else:
                market_top = None

            rows.append(
                {
                    "category": cat,
                    "snapshot_date": date,
                    "n_models": len(picks),
                    "agreement_rate": round(agreement_rate, 2),
                    "picks": picks,
                    "market_favorite": market_top,
                    "any_correct": any(
                        mvm_df[(mvm_df["category"] == cat) & (mvm_df["nominee"] == p)][
                            "is_winner"
                        ].any()
                        for p in picks.values()
                    ),
                }
            )

    if not rows:
        return

    agree_df = pd.DataFrame(rows)

    # Plot: agreement heatmap
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories, strict=True):
        cat_data = agree_df[agree_df["category"] == cat].sort_values("snapshot_date")
        if cat_data.empty:
            continue
        colors = ["green" if c else "red" for c in cat_data["any_correct"]]
        ax.bar(
            range(len(cat_data)),
            cat_data["agreement_rate"],
            color=colors,
            alpha=0.7,
        )

        # Annotate with pick names
        for i, (_, row) in enumerate(cat_data.iterrows()):
            picks_str = ", ".join(f"{mt}:{p[:10]}" for mt, p in row["picks"].items())
            ax.text(i, 0.02, picks_str, rotation=90, fontsize=5, va="bottom", ha="center")

        ax.set_xticks(range(len(cat_data)))
        ax.set_xticklabels(
            [d[5:] for d in cat_data["snapshot_date"]], rotation=45, ha="right", fontsize=8
        )
        ax.set_ylabel("Agreement Rate" if ax == axes[0] else "")
        ax.set_title(CATEGORY_DISPLAY.get(cat, cat), fontsize=11)
        ax.set_ylim(0, 1.1)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="green", lw=6, alpha=0.7, label="Correct winner picked"),
        Line2D([0], [0], color="red", lw=6, alpha=0.7, label="Wrong picks"),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.suptitle("Cross-Model Agreement on Top Pick", fontsize=13)
    _save_fig(fig, "model_agreement.png")

    # Save
    save_rows = []
    for _, r in agree_df.iterrows():
        flat = {
            "category": r["category"],
            "snapshot_date": r["snapshot_date"],
            "agreement_rate": r["agreement_rate"],
            "market_favorite": r["market_favorite"],
            "any_correct": r["any_correct"],
        }
        for mt, pick in r["picks"].items():
            flat[f"pick_{mt}"] = pick
        save_rows.append(flat)
    pd.DataFrame(save_rows).to_csv(RESULTS_DIR / "model_agreement.csv", index=False)
    print(f"  Saved model_agreement.csv ({len(save_rows)} rows)")


def generate_model_performance_table(
    acc_df: pd.DataFrame, mvm_df: pd.DataFrame, pnl_df: pd.DataFrame
) -> None:
    """Generate a comprehensive model performance summary table.

    For each (category, model): final Brier, final rank, winner prob,
    Spearman ρ with market, best P&L, best config.
    """
    rows = []
    categories = sorted(pnl_df["category"].unique())
    models = sorted(pnl_df["model_type"].unique())
    last_snap = acc_df["snapshot_date"].max() if not acc_df.empty else None

    for cat in categories:
        for mt in models:
            row: dict = {"category": cat, "model_type": mt}

            # Accuracy metrics at last snapshot
            if last_snap:
                acc_row = acc_df[
                    (acc_df["category"] == cat)
                    & (acc_df["model_type"] == mt)
                    & (acc_df["snapshot_date"] == last_snap)
                ]
                if not acc_row.empty:
                    row["brier"] = acc_row.iloc[0]["brier"]
                    row["rank"] = acc_row.iloc[0]["rank"]
                    row["winner_prob"] = acc_row.iloc[0]["winner_prob"]
                    row["accuracy"] = acc_row.iloc[0]["accuracy"]

            # Spearman ρ with market at last snapshot
            mvm_last = mvm_df[
                (mvm_df["category"] == cat)
                & (mvm_df["model_type"] == mt)
                & (mvm_df["snapshot_date"] == (last_snap or ""))
            ]
            if len(mvm_last) >= 3:
                rho, _ = stats.spearmanr(mvm_last["model_prob"], mvm_last["market_prob"])
                row["spearman_rho"] = round(float(rho), 4)  # type: ignore[arg-type]  # scipy returns float

            # Best P&L
            pnl_sub = pnl_df[(pnl_df["category"] == cat) & (pnl_df["model_type"] == mt)]
            if not pnl_sub.empty:
                best_idx = pnl_sub["total_pnl"].idxmax()
                row["best_pnl"] = pnl_sub.loc[best_idx, "total_pnl"]
                row["best_config"] = pnl_sub.loc[best_idx, "config_label"]
                row["best_trades"] = pnl_sub.loc[best_idx, "total_trades"]
                row["mean_pnl"] = round(pnl_sub["total_pnl"].mean(), 2)
                row["pct_profitable"] = round(
                    (pnl_sub["total_pnl"] > 0).sum() / len(pnl_sub) * 100, 1
                )

            rows.append(row)

    if rows:
        perf_df = pd.DataFrame(rows)
        perf_df.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False)
        print(f"  Saved model_performance_summary.csv ({len(rows)} rows)")


# ============================================================================
# Parameter isolation analysis
# ============================================================================


def _get_category_blue_shades(categories: list[str]) -> dict[str, str]:
    """Generate blue shades for categories, from light to dark.

    Maps sorted categories to a gradient from light blue to dark navy,
    evenly spaced in the matplotlib 'Blues' colormap.
    """

    n = len(categories)
    # Sample from 0.25-0.95 to avoid near-white and near-black extremes
    cmap = plt.get_cmap("Blues")
    return {
        cat: matplotlib.colors.rgb2hex(cmap(0.25 + 0.70 * i / max(n - 1, 1)))
        for i, cat in enumerate(sorted(categories))
    }


def analyze_parameter_sensitivity(pnl_df: pd.DataFrame) -> None:
    """Analyze each trading parameter dimension in isolation.

    For each parameter (fee_type, kelly_fraction, buy_edge_threshold,
    kelly_mode, trading_side), fix all others and show how P&L varies.
    Produces one subplot per parameter with grouped bars per category
    (blue shades) plus a black aggregate bar.

    Blue shades go from light (first category alphabetically) to dark,
    making per-category trends visible while keeping visual coherence.
    """
    # Only use fixed-bankroll configs for clean comparison
    df = pnl_df[pnl_df["bankroll_mode"] == "fixed"].copy()

    params = [
        ("fee_type", "Fee Type"),
        ("kelly_fraction", "Kelly Fraction"),
        ("buy_edge_threshold", "Edge Threshold"),
        ("kelly_mode", "Kelly Mode"),
    ]

    # Add trading_side if present
    if "trading_side" in df.columns:
        params.append(("trading_side", "Trading Side"))

    categories = sorted(df["category"].unique())
    cat_colors = _get_category_blue_shades(categories)
    n_bars = len(categories) + 1  # +1 for aggregate
    n_params = len(params)

    fig, axes = plt.subplots(n_params, 1, figsize=(max(16, 2 * 7), 4 * n_params))
    if n_params == 1:
        axes = [axes]

    sensitivity_rows = []

    for ax_idx, (param, param_label) in enumerate(params):
        ax = axes[ax_idx]
        vals = sorted(df[param].unique(), key=str)

        x = np.arange(len(vals))
        width = 0.8 / n_bars

        for i, cat in enumerate(categories):
            cat_df = df[df["category"] == cat]
            means = []
            for v in vals:
                subset = cat_df[cat_df[param] == v]
                means.append(subset["total_pnl"].mean() if len(subset) > 0 else 0)
            ax.bar(
                x + i * width,
                means,
                width=width,
                label=CATEGORY_DISPLAY.get(cat, cat),
                color=cat_colors[cat],
                edgecolor="white",
                linewidth=0.3,
            )

            for v, mean_pnl in zip(vals, means, strict=True):
                v_df = cat_df[cat_df[param] == v]
                sensitivity_rows.append(
                    {
                        "parameter": param,
                        "value": str(v),
                        "category": cat,
                        "mean_pnl": round(mean_pnl, 2),
                        "std_pnl": round(v_df["total_pnl"].std(), 2) if len(v_df) > 1 else 0,
                        "n_configs": len(v_df),
                        "pct_profitable": round(
                            (v_df["total_pnl"] > 0).sum() / max(len(v_df), 1) * 100, 1
                        ),
                        "pct_losing": round(
                            (v_df["total_pnl"] < 0).sum() / max(len(v_df), 1) * 100, 1
                        ),
                    }
                )

        # Aggregate bar (black, semi-transparent)
        agg_means = []
        for v in vals:
            subset = df[df[param] == v]
            agg_means.append(subset["total_pnl"].mean() if len(subset) > 0 else 0)
        ax.bar(
            x + len(categories) * width,
            agg_means,
            width=width,
            label="Aggregate",
            color="black",
            alpha=0.5,
            edgecolor="white",
            linewidth=0.3,
        )

        ax.set_xticks(x + width * n_bars / 2)
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_ylabel("Mean P&L ($)")
        ax.set_title(param_label, fontsize=12)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=7, ncol=min(5, n_bars), loc="upper right")

    fig.suptitle("Parameter Sensitivity: Mean P&L by Parameter Value", fontsize=14)
    _save_fig(fig, "parameter_sensitivity.png")

    # Save sensitivity data
    if sensitivity_rows:
        sens_df = pd.DataFrame(sensitivity_rows)
        sens_df.to_csv(RESULTS_DIR / "parameter_sensitivity.csv", index=False)
        print(f"  Saved parameter_sensitivity.csv ({len(sens_df)} rows)")


def analyze_trading_side(pnl_df: pd.DataFrame) -> None:
    """Deep analysis of YES vs NO vs ALL trading side performance.

    Compares P&L distributions across trading sides to determine whether
    sell-side (BUY NO) adds value over YES-only.
    """
    if "trading_side" not in pnl_df.columns:
        print("  No trading_side column — skipping")
        return

    df = pnl_df[pnl_df["bankroll_mode"] == "fixed"].copy()
    sides = sorted(df["trading_side"].unique())
    categories = sorted(df["category"].unique())

    if len(sides) < 2:
        print("  Only one trading side — skipping")
        return

    # For fair comparison, only compare configs that exist for all sides
    # The NO/ALL ablation uses subset of configs. Compare within that subset.
    # Find config params common across sides
    cfg_params = ["kelly_fraction", "buy_edge_threshold", "kelly_mode", "fee_type"]
    side_configs = {}
    for side in sides:
        side_df = df[df["trading_side"] == side]
        configs_set = set()
        for _, r in side_df.iterrows():
            configs_set.add(tuple(r[p] for p in cfg_params))
        side_configs[side] = configs_set

    common = set.intersection(*side_configs.values()) if side_configs else set()

    if not common:
        print("  No common config params across sides — using all data")
        compare_df = df
    else:
        mask = df.apply(lambda r: tuple(r[p] for p in cfg_params) in common, axis=1)
        compare_df = df[mask]

    # Plot: grouped bar chart per category
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.8 / max(len(sides), 1)

    side_colors = {"yes": "#3274a1", "no": "#e1812c", "all": "#3a923a"}

    for i, side in enumerate(sides):
        side_df = compare_df[compare_df["trading_side"] == side]
        means = []
        for cat in categories:
            cat_df = side_df[side_df["category"] == cat]
            means.append(cat_df["total_pnl"].mean() if len(cat_df) > 0 else 0)
        ax.bar(
            x + i * width,
            means,
            width=width,
            label=f"Side: {side.upper()}",
            color=side_colors.get(side, "gray"),
            alpha=0.8,
        )

    ax.set_xticks(x + width * len(sides) / 2)
    ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c) for c in categories], rotation=30, ha="right")
    ax.set_ylabel("Mean P&L ($)")
    ax.set_title("Trading Side Comparison: Mean P&L by Category", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    _save_fig(fig, "trading_side_comparison.png")

    # Summary table
    rows = []
    for side in sides:
        side_df = compare_df[compare_df["trading_side"] == side]
        for cat in categories:
            cat_df = side_df[side_df["category"] == cat]
            if cat_df.empty:
                continue
            rows.append(
                {
                    "trading_side": side,
                    "category": cat,
                    "mean_pnl": round(cat_df["total_pnl"].mean(), 2),
                    "best_pnl": round(cat_df["total_pnl"].max(), 2),
                    "worst_pnl": round(cat_df["total_pnl"].min(), 2),
                    "pct_profitable": round((cat_df["total_pnl"] > 0).sum() / len(cat_df) * 100, 1),
                    "n_configs": len(cat_df),
                }
            )
        # Aggregate across categories
        agg = compare_df[compare_df["trading_side"] == side]
        per_config_total = agg.groupby(["model_type", "config_label"])["total_pnl"].sum()
        rows.append(
            {
                "trading_side": side,
                "category": "AGGREGATE",
                "mean_pnl": round(per_config_total.mean(), 2),
                "best_pnl": round(per_config_total.max(), 2),
                "worst_pnl": round(per_config_total.min(), 2),
                "pct_profitable": round(
                    (per_config_total > 0).sum() / max(len(per_config_total), 1) * 100, 1
                ),
                "n_configs": len(per_config_total),
            }
        )

    if rows:
        side_df = pd.DataFrame(rows)
        side_df.to_csv(RESULTS_DIR / "trading_side_analysis.csv", index=False)
        print(f"  Saved trading_side_analysis.csv ({len(side_df)} rows)")


# ============================================================================
# Risk analysis
# ============================================================================


def analyze_risk_profile(pnl_df: pd.DataFrame) -> None:
    """Analyze risk profile: probability of loss and worst-case scenarios.

    For each model type, compute:
    - Probability of aggregate loss (across all categories)
    - Max drawdown per category
    - Distribution of per-category P&L
    - Tail risk: worst 10% of configs

    This helps answer: "What's the probability we lose money, and by how much?"
    """
    df = pnl_df[pnl_df["bankroll_mode"] == "fixed"].copy()
    categories = sorted(df["category"].unique())
    models = sorted(df["model_type"].unique())

    # 1. Per-config aggregate P&L across categories
    agg = (
        df.groupby(["model_type", "config_label"])
        .agg(
            total_pnl=("total_pnl", "sum"),
            n_positive=("total_pnl", lambda x: (x > 0).sum()),
            n_negative=("total_pnl", lambda x: (x < 0).sum()),
            n_zero=("total_pnl", lambda x: (x == 0).sum()),
            worst_category=("total_pnl", "min"),
        )
        .reset_index()
    )

    # 2. Risk metrics per model
    risk_rows = []
    for mt in models:
        mt_df = agg[agg["model_type"] == mt]
        if mt_df.empty:
            continue

        n_configs = len(mt_df)
        n_profit = (mt_df["total_pnl"] > 0).sum()
        n_loss = (mt_df["total_pnl"] < 0).sum()
        n_zero = (mt_df["total_pnl"] == 0).sum()

        risk_rows.append(
            {
                "model_type": mt,
                "n_configs": n_configs,
                "prob_profit": round(n_profit / n_configs * 100, 1),
                "prob_loss": round(n_loss / n_configs * 100, 1),
                "prob_zero": round(n_zero / n_configs * 100, 1),
                "mean_pnl": round(mt_df["total_pnl"].mean(), 2),
                "median_pnl": round(mt_df["total_pnl"].median(), 2),
                "best_pnl": round(mt_df["total_pnl"].max(), 2),
                "worst_pnl": round(mt_df["total_pnl"].min(), 2),
                "p10_pnl": round(mt_df["total_pnl"].quantile(0.1), 2),
                "p25_pnl": round(mt_df["total_pnl"].quantile(0.25), 2),
                "p75_pnl": round(mt_df["total_pnl"].quantile(0.75), 2),
                "sharpe_like": round(mt_df["total_pnl"].mean() / mt_df["total_pnl"].std(), 3)
                if mt_df["total_pnl"].std() > 0
                else 0,
                "worst_single_category": round(mt_df["worst_category"].min(), 2),
            }
        )

    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(RESULTS_DIR / "risk_profile.csv", index=False)
    print(f"  Saved risk_profile.csv ({len(risk_df)} rows)")

    # 3. Plot: P&L distribution per model (histogram)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, mt in enumerate(models[:4]):
        ax = axes[i]
        mt_df = agg[agg["model_type"] == mt]
        if mt_df.empty:
            continue

        pnls = mt_df["total_pnl"]
        color = get_model_color(mt)

        ax.hist(pnls, bins=20, color=color, alpha=0.7, edgecolor="white")
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

        prob_loss = (pnls < 0).sum() / len(pnls) * 100
        ax.set_title(
            f"{MODEL_DISPLAY.get(mt, mt)}\n"
            f"P(loss)={prob_loss:.0f}%, P(profit)={100 - prob_loss - (pnls == 0).sum() / len(pnls) * 100:.0f}%",
            fontsize=11,
        )
        ax.set_xlabel("Aggregate P&L ($)")
        ax.set_ylabel("# Configs")
        ax.legend(fontsize=7)

    fig.suptitle("Risk Profile: Aggregate P&L Distribution by Model", fontsize=14)
    _save_fig(fig, "risk_profile.png")

    # 4. Per-category loss exposure for recommended config
    # Show: for each (model, config), which categories lose money?
    fig, ax = plt.subplots(figsize=(14, 7))

    cat_model_data = []
    for mt in models:
        mt_sub = df[df["model_type"] == mt]
        cat_means = mt_sub.groupby("category")["total_pnl"].agg(["mean", "std", "min", "max"])
        for cat in categories:
            if cat in cat_means.index:
                cat_model_data.append(
                    {
                        "model": mt,
                        "category": cat,
                        "mean": cat_means.loc[cat, "mean"],
                        "std": cat_means.loc[cat, "std"],
                        "min": cat_means.loc[cat, "min"],
                        "max": cat_means.loc[cat, "max"],
                    }
                )

    if cat_model_data:
        cm_df = pd.DataFrame(cat_model_data)
        x = np.arange(len(categories))
        width = 0.8 / max(len(models), 1)

        for i, mt in enumerate(models):
            mt_d = cm_df[cm_df["model"] == mt]
            means = [
                mt_d[mt_d["category"] == c]["mean"].iloc[0]
                if len(mt_d[mt_d["category"] == c]) > 0
                else 0
                for c in categories
            ]
            stds = [  # noqa: F841
                mt_d[mt_d["category"] == c]["std"].iloc[0]
                if len(mt_d[mt_d["category"] == c]) > 0
                else 0
                for c in categories
            ]
            mins = [
                mt_d[mt_d["category"] == c]["min"].iloc[0]
                if len(mt_d[mt_d["category"] == c]) > 0
                else 0
                for c in categories
            ]
            maxs = [
                mt_d[mt_d["category"] == c]["max"].iloc[0]
                if len(mt_d[mt_d["category"] == c]) > 0
                else 0
                for c in categories
            ]

            ax.bar(
                x + i * width,
                means,
                width=width,
                label=MODEL_DISPLAY.get(mt, mt),
                color=get_model_color(mt),
                alpha=0.7,
            )
            # Error bars: min to max range
            ax.errorbar(
                x + i * width,
                means,
                yerr=[
                    [m - mn for m, mn in zip(means, mins, strict=True)],
                    [mx - m for m, mx in zip(means, maxs, strict=True)],
                ],
                fmt="none",
                ecolor="gray",
                alpha=0.5,
                capsize=2,
            )

        ax.set_xticks(x + width * len(models) / 2)
        ax.set_xticklabels(
            [CATEGORY_DISPLAY.get(c, c) for c in categories], rotation=30, ha="right"
        )
        ax.set_ylabel("P&L ($)")
        ax.set_title(
            "Per-Category P&L Distribution by Model\n(bars=mean, whiskers=min/max)", fontsize=13
        )
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.legend(fontsize=8)

    _save_fig(fig, "category_risk_exposure.png")


# ============================================================================
# Summary table
# ============================================================================


def print_summary_table(pnl_df: pd.DataFrame, acc_df: pd.DataFrame | None) -> None:
    """Print a summary table of best results per model × category."""
    print("\n" + "=" * 90)
    print("SUMMARY: Best P&L per Model × Category")
    print("=" * 90)

    cats = sorted(pnl_df["category"].unique())
    models = sorted(pnl_df["model_type"].unique())

    # Header
    header = f"{'Category':<20}"
    for mt in models:
        header += f" {MODEL_DISPLAY.get(mt, mt):>15}"
    print(header)
    print("-" * 90)

    for cat in cats:
        row_str = f"{CATEGORY_DISPLAY.get(cat, cat):<20}"
        for mt in models:
            subset = pnl_df[(pnl_df["category"] == cat) & (pnl_df["model_type"] == mt)]
            if not subset.empty:
                best = subset["total_pnl"].max()
                row_str += f" ${best:>+13.2f}"
            else:
                row_str += f" {'N/A':>14}"
        print(row_str)

    # Totals
    print("-" * 90)
    total_str = f"{'TOTAL':<20}"
    for mt in models:
        subset = pnl_df[pnl_df["model_type"] == mt]
        if not subset.empty:
            best_per_cat = subset.groupby("category")["total_pnl"].max().sum()
            total_str += f" ${best_per_cat:>+13.2f}"
        else:
            total_str += f" {'N/A':>14}"
    print(total_str)

    # Also print accuracy summary
    if acc_df is not None and not acc_df.empty:
        print("\n" + "=" * 90)
        print("SUMMARY: Mean Brier Score per Model × Category (last snapshot)")
        print("=" * 90)

        last_snap = acc_df["snapshot_date"].max()
        last_acc = acc_df[acc_df["snapshot_date"] == last_snap]

        header = f"{'Category':<20}"
        for mt in models:
            header += f" {MODEL_DISPLAY.get(mt, mt):>15}"
        print(header)
        print("-" * 90)

        for cat in cats:
            row_str = f"{CATEGORY_DISPLAY.get(cat, cat):<20}"
            for mt in models:
                subset = last_acc[(last_acc["category"] == cat) & (last_acc["model_type"] == mt)]
                if not subset.empty:
                    brier = subset["brier"].iloc[0]
                    row_str += f" {brier:>14.4f}"
                else:
                    row_str += f" {'N/A':>14}"
            print(row_str)


# ============================================================================
# Bankroll Allocation Analysis
# ============================================================================


def analyze_bankroll_allocation(
    pnl_df: pd.DataFrame,
    mvm_df: pd.DataFrame | None,
    spread_df: pd.DataFrame | None,
) -> None:
    """Post-hoc analysis of portfolio-level bankroll allocation strategies.

    Computes what the aggregate portfolio P&L would have been under different
    allocation schemes:

    - **Equal weight** (baseline): $1,000 per category.
    - **Edge-weighted**: Allocate proportional to mean |model - market| divergence.
      Categories where the model disagrees more get more capital.
    - **Volume-weighted**: Allocate proportional to total market trading volume.
      More liquid markets get more capital (lower execution risk).

    All allocations preserve total capital (sum of weights = n_categories).
    P&L scales linearly with bankroll, so we multiply per-category P&L by
    the weight ratio.

    Outputs:
    - ``bankroll_allocation.csv``: Best portfolio P&L per (model, config) under
      each scheme.
    - ``bankroll_allocation.png``: Bar chart comparing schemes at the optimal config.
    """
    df = pnl_df[pnl_df["bankroll_mode"] == "fixed"].copy()
    categories = sorted(df["category"].unique())
    n_cats = len(categories)

    # --- Compute allocation weights ---
    weights: dict[str, dict[str, float]] = {}

    # Equal weight (baseline)
    weights["equal"] = dict.fromkeys(categories, 1.0)

    # Edge-weighted: mean absolute divergence per category
    if mvm_df is not None and not mvm_df.empty:
        edge_by_cat = {}
        for cat in categories:
            cat_mvm = mvm_df[mvm_df["category"] == cat]
            if len(cat_mvm) > 0:
                edge_by_cat[cat] = cat_mvm["divergence"].abs().mean()
            else:
                edge_by_cat[cat] = 0.0

        total_edge = sum(edge_by_cat.values())
        if total_edge > 0:
            weights["edge_weighted"] = {
                cat: (edge_by_cat[cat] / total_edge) * n_cats for cat in categories
            }

    # Volume-weighted: total trades per category
    if spread_df is not None and not spread_df.empty:
        vol_by_cat = {}
        for cat in categories:
            cat_spread = spread_df[spread_df["category"] == cat]
            if len(cat_spread) > 0 and "total_trades" in cat_spread.columns:
                vol_by_cat[cat] = cat_spread["total_trades"].sum()
            else:
                vol_by_cat[cat] = 0.0

        total_vol = sum(vol_by_cat.values())
        if total_vol > 0:
            weights["volume_weighted"] = {
                cat: (vol_by_cat[cat] / total_vol) * n_cats for cat in categories
            }

    if len(weights) < 2:
        print("  Insufficient data for bankroll allocation comparison — skipping")
        return

    # --- Compute portfolio P&L under each allocation ---
    group_cols = ["model_type", "config_label"]
    allocation_rows = []

    for (model, config), group in df.groupby(group_cols):
        cat_pnl = {}
        for _, row in group.iterrows():
            cat_pnl[row["category"]] = row["total_pnl"]

        for scheme_name, scheme_weights in weights.items():
            portfolio_pnl = sum(
                cat_pnl.get(cat, 0) * scheme_weights.get(cat, 1.0) for cat in categories
            )
            allocation_rows.append(
                {
                    "model_type": model,
                    "config_label": config,
                    "allocation": scheme_name,
                    "portfolio_pnl": round(portfolio_pnl, 2),
                }
            )

    alloc_df = pd.DataFrame(allocation_rows)
    alloc_df.to_csv(RESULTS_DIR / "bankroll_allocation.csv", index=False)
    print(f"  Saved bankroll_allocation.csv ({len(alloc_df)} rows)")

    # --- Print weights ---
    print("\n  Allocation weights (1.0 = equal share):")
    for scheme_name, scheme_weights in weights.items():
        w_str = ", ".join(
            f"{CATEGORY_DISPLAY.get(c, c)}: {w:.2f}" for c, w in sorted(scheme_weights.items())
        )
        print(f"    {scheme_name}: {w_str}")

    # --- Best portfolio under each allocation ---
    print("\n  Best portfolio per model (best config):")
    for scheme_name in weights:
        scheme_df = alloc_df[alloc_df["allocation"] == scheme_name]
        best = scheme_df.loc[scheme_df["portfolio_pnl"].idxmax()]
        print(
            f"    {scheme_name}: ${best['portfolio_pnl']:.2f} "
            f"({best['model_type']}, {best['config_label']})"
        )

    # --- Plot: bar chart comparing best portfolio P&L per scheme ---
    schemes = list(weights.keys())
    models = sorted(df["model_type"].unique())

    fig, axes = plt.subplots(1, len(schemes), figsize=(6 * len(schemes), 6), sharey=True)
    if len(schemes) == 1:
        axes = [axes]

    model_colors = ["#3274a1", "#3a923a", "#e1812c", "#9372b2"]

    for ax, scheme in zip(axes, schemes, strict=True):
        model_pnls = []
        for model in models:
            scheme_df = alloc_df[
                (alloc_df["allocation"] == scheme) & (alloc_df["model_type"] == model)
            ]
            if scheme_df.empty:
                model_pnls.append(0)
                continue
            model_pnls.append(scheme_df["portfolio_pnl"].max())

        x = np.arange(len(models))
        bars = ax.bar(
            x,
            model_pnls,
            color=model_colors[: len(models)],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, pnl in zip(bars, model_pnls, strict=True):
            va = "bottom" if pnl >= 0 else "top"
            y_off = 5 if pnl >= 0 else -5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_off,
                f"${pnl:.0f}",
                ha="center",
                va=va,
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=30, ha="right")
        ax.set_title(scheme.replace("_", " ").title(), fontsize=12)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Best Config Portfolio P&L ($)")

    fig.suptitle("Bankroll Allocation: Best Portfolio by Scheme", fontsize=14)
    _save_fig(fig, "bankroll_allocation.png")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all analyses and generate plots."""
    global RESULTS_DIR  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Results subdirectory name under 2025/ (e.g. results_inferred_6h, "
            "results_delay_0). Default: results_inferred_6h."
        ),
    )
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = EXP_DIR / "2025" / args.results_dir

    print("=" * 70)
    print("Multi-Category Backtest Analysis: 2025 Ceremony")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading results...")
    dfs = load_results()

    pnl_df = dfs.get("daily_pnl")
    acc_df = dfs.get("model_accuracy")
    mvm_df = dfs.get("model_vs_market")
    spread_df = dfs.get("spread_report")

    if pnl_df is None or pnl_df.empty:
        print("ERROR: No P&L data found. Run run_backtests.py first.")
        return

    # 1. Model accuracy heatmaps
    if acc_df is not None and not acc_df.empty:
        print("\n--- Model Accuracy Heatmaps ---")
        plot_model_accuracy_heatmaps(acc_df)

    # 2. Classification vs trading
    if acc_df is not None and not acc_df.empty and pnl_df is not None:
        print("\n--- Classification vs Trading ---")
        plot_classification_vs_trading(acc_df, pnl_df)

    # 3. Model vs market divergence
    if mvm_df is not None and not mvm_df.empty:
        print("\n--- Model vs Market Divergence ---")
        plot_model_vs_market(mvm_df)

    # 4. Per-precursor P&L (needs mvm_df for synthetic single-snapshot backtest)
    if mvm_df is not None and not mvm_df.empty:
        print("\n--- Per-Precursor P&L ---")
        analyze_per_precursor_pnl(mvm_df, pnl_df)

    # 5. Return decomposition
    print("\n--- Return Decomposition ---")
    analyze_return_decomposition(pnl_df)

    # 6. Trading parameter grid (legacy)
    print("\n--- Trading Parameter Sensitivity (legacy) ---")
    analyze_trading_grid(pnl_df)

    # 6b. Parameter isolation analysis (new — per-dimension)
    print("\n--- Parameter Sensitivity (isolated) ---")
    analyze_parameter_sensitivity(pnl_df)

    # 6c. Trading side analysis
    print("\n--- Trading Side Analysis ---")
    analyze_trading_side(pnl_df)

    # 7. P&L by category
    print("\n--- P&L by Category ---")
    plot_pnl_by_category(pnl_df)

    # 8. Spread & liquidity
    if spread_df is not None and not spread_df.empty:
        print("\n--- Spread & Liquidity ---")
        plot_spread_report(spread_df)

    # 9. Best config selection
    print("\n--- Config Selection ---")
    select_best_config(pnl_df)

    # 9b. Risk profile analysis
    print("\n--- Risk Profile ---")
    analyze_risk_profile(pnl_df)

    # 9c. Bankroll allocation analysis
    print("\n--- Bankroll Allocation ---")
    analyze_bankroll_allocation(pnl_df, mvm_df, spread_df)

    # 10. Timing analysis
    if mvm_df is not None and not mvm_df.empty:
        print("\n--- Timing Analysis ---")
        plot_timing_analysis(mvm_df, pnl_df)

    # 11. Detailed model performance
    if mvm_df is not None and not mvm_df.empty:
        print("\n--- Probability Trajectories (overview) ---")
        plot_probability_trajectories(mvm_df)

        print("\n--- Per-Outcome Trajectories (detailed) ---")
        plot_per_outcome_trajectories(mvm_df)

        print("\n--- Divergence Heatmaps ---")
        plot_divergence_heatmaps(mvm_df)

        print("\n--- Probability Concentration ---")
        plot_probability_concentration(mvm_df)

        print("\n--- Nominee Correlation ---")
        plot_nominee_correlation(mvm_df)

        print("\n--- Model Agreement ---")
        plot_model_agreement(mvm_df)

    if acc_df is not None and not acc_df.empty:
        print("\n--- Brier Score Evolution ---")
        plot_brier_evolution(acc_df)

        print("\n--- Winner Rank Evolution ---")
        plot_winner_rank_evolution(acc_df)

    # 12. Performance summary table
    if acc_df is not None and mvm_df is not None:
        print("\n--- Model Performance Summary ---")
        generate_model_performance_table(acc_df, mvm_df, pnl_df)

    # Summary table
    print_summary_table(pnl_df, acc_df)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Plots: {PLOTS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
