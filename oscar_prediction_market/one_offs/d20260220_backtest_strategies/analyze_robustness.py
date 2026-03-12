"""Config neighborhood sensitivity and robustness score analysis.

Produces:
1. Config neighborhood heatmaps around the recommended config
2. Robustness score ranking across all configs and categories
3. Risk-bounded selection with varying tolerance parameters

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260220_backtest_strategies.analyze_robustness
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.analysis_utils.style import (
    apply_style,
    get_model_color,
)

apply_style()

# ============================================================================
# Constants
# ============================================================================

EXP_DIR = Path("storage/d20260220_backtest_strategies")
RESULTS_DIR = EXP_DIR / "2025" / "results_inferred_6h"
PLOTS_DIR = EXP_DIR / "2025" / "plots"

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
    "actor_leading": "Actor",
    "actress_leading": "Actress",
    "actor_supporting": "Supp. Actor",
    "actress_supporting": "Supp. Actress",
    "original_screenplay": "Orig. Screenplay",
    "animated_feature": "Animated Feature",
    "cinematography": "Cinematography",
}

# Reference config from the README recommended config
REF_CONFIG = {
    "fee_type": "maker",
    "kelly_fraction": 0.20,
    "buy_edge_threshold": 0.10,
    "kelly_mode": "multi_outcome",
    "trading_side": "no",
    "bankroll_mode": "fixed",
}

# Neighborhood ranges for each parameter
NEIGHBORHOOD: dict[str, list[float | str]] = {
    "kelly_fraction": [0.15, 0.20, 0.25, 0.35],
    "buy_edge_threshold": [0.06, 0.08, 0.10, 0.12],
    "trading_side": ["yes", "no", "all"],
    "kelly_mode": ["independent", "multi_outcome"],
}


def _save_fig(fig: matplotlib.figure.Figure, filename: str, *, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {filename}")


def load_pnl() -> pd.DataFrame:
    """Load P&L data."""
    path = RESULTS_DIR / "daily_pnl.csv"
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} P&L rows from {path}")
    return df


# ============================================================================
# 1. Config Neighborhood Sensitivity
# ============================================================================


def analyze_config_neighborhood(pnl_df: pd.DataFrame) -> None:
    """Analyze P&L sensitivity in the neighborhood of the recommended config.

    Fixes fee=maker, bankroll=fixed. Varies:
    - kelly_fraction: [0.15, 0.20, 0.25, 0.35]
    - buy_edge_threshold: [0.06, 0.08, 0.10, 0.12]
    - trading_side: [yes, no, all]
    - kelly_mode: [independent, multi_outcome]

    Produces:
    - Heatmap: kf × edge for each (model, trading_side) combo
    - Distribution plots: P&L by variable, faceted by model
    """
    # Filter to neighborhood
    kf_vals = NEIGHBORHOOD["kelly_fraction"]
    edge_vals = NEIGHBORHOOD["buy_edge_threshold"]
    df = pnl_df[
        (pnl_df["fee_type"] == "maker")
        & (pnl_df["bankroll_mode"] == "fixed")
        & (pnl_df["kelly_fraction"].isin(kf_vals))
        & (pnl_df["buy_edge_threshold"].isin(edge_vals))
    ].copy()

    if df.empty:
        print("  WARNING: No data in neighborhood filter")
        return

    # Aggregate P&L across categories per config
    agg = (
        df.groupby(
            [
                "model_type",
                "config_label",
                "kelly_fraction",
                "buy_edge_threshold",
                "kelly_mode",
                "trading_side",
            ]
        )
        .agg(
            agg_pnl=("total_pnl", "sum"),
            n_categories=("total_pnl", "count"),
            n_profitable=("total_pnl", lambda x: (x > 0).sum()),
            worst_cat_pnl=("total_pnl", "min"),
        )
        .reset_index()
    )

    models = sorted(agg["model_type"].unique())
    sides = ["yes", "no", "all"]

    # ── Plot A: Heatmap grid — kf × edge, faceted by (model, side) ──
    fig, axes = plt.subplots(
        len(models),
        len(sides),
        figsize=(5 * len(sides), 4 * len(models)),
        squeeze=False,
    )

    kfs = sorted(NEIGHBORHOOD["kelly_fraction"])
    edges = sorted(NEIGHBORHOOD["buy_edge_threshold"])

    for i, mt in enumerate(models):
        for j, side in enumerate(sides):
            ax = axes[i, j]
            sub = agg[(agg["model_type"] == mt) & (agg["trading_side"] == side)]

            # Pivot: average across kelly_mode
            pivot_data = np.full((len(kfs), len(edges)), np.nan)
            for ki, kf in enumerate(kfs):
                for ei, edge in enumerate(edges):
                    cell = sub[(sub["kelly_fraction"] == kf) & (sub["buy_edge_threshold"] == edge)]
                    if not cell.empty:
                        pivot_data[ki, ei] = cell["agg_pnl"].mean()

            vmax = max(abs(np.nanmin(pivot_data)), abs(np.nanmax(pivot_data)))
            if vmax == 0:
                vmax = 1
            im = ax.imshow(
                pivot_data,
                cmap="RdYlGn",
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )

            ax.set_xticks(range(len(edges)))
            ax.set_xticklabels([f"{e:.2f}" for e in edges], fontsize=8)
            ax.set_yticks(range(len(kfs)))
            ax.set_yticklabels([f"{k:.2f}" for k in kfs], fontsize=8)

            if i == len(models) - 1:
                ax.set_xlabel("Edge Threshold")
            if j == 0:
                ax.set_ylabel("Kelly Fraction")
            ax.set_title(
                f"{MODEL_DISPLAY.get(mt, mt)}\nSide={side}",
                fontsize=9,
            )

            # Annotate cells
            for ki in range(len(kfs)):
                for ei in range(len(edges)):
                    val = pivot_data[ki, ei]
                    if not np.isnan(val):
                        color = "white" if abs(val) > vmax * 0.6 else "black"
                        ax.text(
                            ei,
                            ki,
                            f"${val:.0f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                        )

            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Config Neighborhood: Aggregate P&L (all 9 categories)\n"
        "Fee=maker, bankroll=fixed, avg across kelly_mode",
        fontsize=14,
    )
    _save_fig(fig, "config_neighborhood_heatmap.png")

    # ── Plot B: P&L distribution by variable, faceted by model ──
    variables: list[tuple[str, str, Sequence[float | str]]] = [
        ("trading_side", "Trading Side", sides),
        ("kelly_mode", "Kelly Mode", ["independent", "multi_outcome"]),
        ("kelly_fraction", "Kelly Fraction", kfs),
        ("buy_edge_threshold", "Edge Threshold", edges),
    ]

    fig, axes = plt.subplots(
        len(variables),
        len(models),
        figsize=(4 * len(models), 3.5 * len(variables)),
        squeeze=False,
    )

    for vi, (var, var_label, vals) in enumerate(variables):
        for mi, mt in enumerate(models):
            ax = axes[vi, mi]
            mt_data = agg[agg["model_type"] == mt]
            box_data = []
            box_labels = []
            for v in vals:
                subset = mt_data[mt_data[var] == v]["agg_pnl"]
                if not subset.empty:
                    box_data.append(subset.values)
                    box_labels.append(str(v))

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    tick_labels=box_labels,
                    patch_artist=True,
                    medianprops={"color": "red", "linewidth": 1.5},
                )
                color = get_model_color(mt)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.4)

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            if vi == 0:
                ax.set_title(MODEL_DISPLAY.get(mt, mt), fontsize=10)
            if mi == 0:
                ax.set_ylabel(f"{var_label}\nAgg P&L ($)")
            ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        "Config Neighborhood: P&L Distribution by Parameter\n"
        "Fee=maker, bankroll=fixed, all 9 categories",
        fontsize=14,
    )
    _save_fig(fig, "config_neighborhood_distributions.png")

    # ── Save neighborhood data ──
    agg.to_csv(RESULTS_DIR / "config_neighborhood.csv", index=False)
    print(f"  Saved config_neighborhood.csv ({len(agg)} rows)")


# ============================================================================
# 2. Robustness Score Ranking
# ============================================================================


def compute_robustness_scores(
    pnl_df: pd.DataFrame,
    *,
    max_loss_fraction: float = 0.20,
    min_profitable_fraction: float = 0.33,
) -> pd.DataFrame:
    """Compute robustness score for each (model, config) across all categories.

    The robustness score combines:
    1. Aggregate P&L rank (higher is better)
    2. Worst single-category P&L rank (less negative is better)
    3. Config-neighborhood stability: fraction of neighboring configs that also
       profit (robustness to parameter perturbation)
    4. P(loss bounded): whether worst-case category loss is within tolerance

    Args:
        pnl_df: Full P&L DataFrame.
        max_loss_fraction: Maximum acceptable loss as fraction of bankroll per
            category ($1000 bankroll → $200 max loss at 0.20). Configs with
            worst_category_pnl < -max_loss_fraction * 1000 are penalized.
        min_profitable_fraction: Minimum fraction of categories that must be
            profitable (> $0). Default 0.33 = at least 3 of 9 categories.

    Returns:
        DataFrame with one row per (model_type, config_label), sorted by
        robustness_score descending.
    """
    bankroll = 1000.0
    max_loss = max_loss_fraction * bankroll
    categories = sorted(pnl_df["category"].unique())
    n_categories = len(categories)
    min_profitable_cats = int(np.ceil(min_profitable_fraction * n_categories))

    df = pnl_df[pnl_df["bankroll_mode"] == "fixed"].copy()

    # Aggregate per (model, config) across ALL categories
    agg = (
        df.groupby(
            [
                "model_type",
                "config_label",
                "fee_type",
                "kelly_fraction",
                "buy_edge_threshold",
                "kelly_mode",
                "trading_side",
            ]
        )
        .agg(
            agg_pnl=("total_pnl", "sum"),
            n_categories=("total_pnl", "count"),
            n_profitable=("total_pnl", lambda x: (x > 0).sum()),
            n_losing=("total_pnl", lambda x: (x < 0).sum()),
            worst_cat_pnl=("total_pnl", "min"),
            best_cat_pnl=("total_pnl", "max"),
            mean_cat_pnl=("total_pnl", "mean"),
            std_cat_pnl=("total_pnl", "std"),
            median_cat_pnl=("total_pnl", "median"),
        )
        .reset_index()
    )

    # Fill NaN std for single-category or constant P&L
    agg["std_cat_pnl"] = agg["std_cat_pnl"].fillna(0)

    # ── Component 1: Aggregate P&L percentile rank ──
    agg["pnl_rank"] = agg["agg_pnl"].rank(pct=True)

    # ── Component 2: Worst-category P&L percentile rank ──
    # Higher = less bad worst case
    agg["worst_cat_rank"] = agg["worst_cat_pnl"].rank(pct=True)

    # ── Component 3: Profitable fraction ──
    agg["profitable_fraction"] = agg["n_profitable"] / agg["n_categories"]

    # ── Component 4: Loss-bounded flag ──
    agg["loss_bounded"] = agg["worst_cat_pnl"] >= -max_loss

    # ── Component 5: Sharpe-like ratio (mean/std across categories) ──
    agg["sharpe_like"] = np.where(
        agg["std_cat_pnl"] > 0,
        agg["mean_cat_pnl"] / agg["std_cat_pnl"],
        0,
    )
    agg["sharpe_rank"] = agg["sharpe_like"].rank(pct=True)

    # ── Component 6: Config-neighborhood stability ──
    # For each config, check if configs ±1 step in edge and kf also profit
    edge_vals = sorted(df["buy_edge_threshold"].unique())
    kf_vals = sorted(df["kelly_fraction"].unique())
    edge_idx = {v: i for i, v in enumerate(edge_vals)}
    kf_idx = {v: i for i, v in enumerate(kf_vals)}

    # Build lookup: (model, fee, kf, edge, km, side) → agg_pnl
    key_to_pnl = {}
    for _, row in agg.iterrows():
        key = (
            row["model_type"],
            row["fee_type"],
            row["kelly_fraction"],
            row["buy_edge_threshold"],
            row["kelly_mode"],
            row["trading_side"],
        )
        key_to_pnl[key] = row["agg_pnl"]

    neighbor_fracs = []
    for _, row in agg.iterrows():
        ei = edge_idx.get(row["buy_edge_threshold"])
        ki = kf_idx.get(row["kelly_fraction"])
        if ei is None or ki is None:
            neighbor_fracs.append(0.0)
            continue

        neighbors_profitable = 0
        neighbors_total = 0
        for de in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if de == 0 and dk == 0:
                    continue  # skip self
                ni = ei + de
                nk = ki + dk
                if 0 <= ni < len(edge_vals) and 0 <= nk < len(kf_vals):
                    key = (
                        row["model_type"],
                        row["fee_type"],
                        kf_vals[nk],
                        edge_vals[ni],
                        row["kelly_mode"],
                        row["trading_side"],
                    )
                    npnl = key_to_pnl.get(key)
                    if npnl is not None:
                        neighbors_total += 1
                        if npnl > 0:
                            neighbors_profitable += 1

        frac = neighbors_profitable / max(neighbors_total, 1)
        neighbor_fracs.append(frac)

    agg["neighbor_stability"] = neighbor_fracs

    # ── Composite robustness score ──
    # Weights: 35% P&L rank, 20% worst-case, 15% Sharpe, 15% neighbor stability,
    #          10% profitable fraction, 5% loss-bounded bonus
    agg["robustness_score"] = (
        0.35 * agg["pnl_rank"]
        + 0.20 * agg["worst_cat_rank"]
        + 0.15 * agg["sharpe_rank"]
        + 0.15 * agg["neighbor_stability"]
        + 0.10 * agg["profitable_fraction"]
        + 0.05 * agg["loss_bounded"].astype(float)
    )

    # ── Penalty: if too many categories lose or worst case exceeds tolerance ──
    below_min = agg["n_profitable"] < min_profitable_cats
    agg.loc[below_min, "robustness_score"] *= 0.5  # halve score

    return agg.sort_values("robustness_score", ascending=False).reset_index(drop=True)


def analyze_robustness(pnl_df: pd.DataFrame) -> None:
    """Run robustness analysis with multiple tolerance parameters and generate plots."""
    models = sorted(pnl_df["model_type"].unique())

    # ── Sweep over tolerance parameters ──
    tolerances = [
        {"max_loss_fraction": 0.10, "min_profitable_fraction": 0.33},
        {"max_loss_fraction": 0.15, "min_profitable_fraction": 0.33},
        {"max_loss_fraction": 0.20, "min_profitable_fraction": 0.33},
        {"max_loss_fraction": 0.30, "min_profitable_fraction": 0.33},
        {"max_loss_fraction": 0.20, "min_profitable_fraction": 0.22},
        {"max_loss_fraction": 0.20, "min_profitable_fraction": 0.44},
        {"max_loss_fraction": 0.20, "min_profitable_fraction": 0.56},
    ]

    sweep_rows = []
    for tol in tolerances:
        scores = compute_robustness_scores(pnl_df, **tol)
        top = scores.head(10)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            sweep_rows.append(
                {
                    "max_loss_frac": tol["max_loss_fraction"],
                    "min_profit_frac": tol["min_profitable_fraction"],
                    "rank": rank,
                    "model_type": row["model_type"],
                    "config_label": row["config_label"],
                    "agg_pnl": round(row["agg_pnl"], 2),
                    "worst_cat_pnl": round(row["worst_cat_pnl"], 2),
                    "n_profitable": int(row["n_profitable"]),
                    "n_categories": int(row["n_categories"]),
                    "sharpe_like": round(row["sharpe_like"], 3),
                    "neighbor_stability": round(row["neighbor_stability"], 3),
                    "robustness_score": round(row["robustness_score"], 4),
                    "loss_bounded": bool(row["loss_bounded"]),
                }
            )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS_DIR / "robustness_sweep.csv", index=False)
    print(f"  Saved robustness_sweep.csv ({len(sweep_df)} rows)")

    # ── Primary analysis: default tolerances ──
    scores = compute_robustness_scores(pnl_df, max_loss_fraction=0.20, min_profitable_fraction=0.33)
    scores.to_csv(RESULTS_DIR / "robustness_scores.csv", index=False)
    print(f"  Saved robustness_scores.csv ({len(scores)} rows)")

    # ── Plot 1: Top configs by robustness score ──
    top_n = 20
    top = scores.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = [get_model_color(mt) for mt in top["model_type"]]
    bars = ax.barh(  # noqa: F841
        range(top_n),
        top["robustness_score"],
        color=colors,
        alpha=0.8,
        edgecolor="white",
    )

    # Add P&L annotation
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["robustness_score"] + 0.005,
            i,
            f"P&L=${row['agg_pnl']:.0f}  worst=${row['worst_cat_pnl']:.0f}  "
            f"prof={row['n_profitable']}/{row['n_categories']}  "
            f"nb={row['neighbor_stability']:.0%}",
            va="center",
            fontsize=7,
        )

    labels = [
        f"{MODEL_DISPLAY.get(row['model_type'], row['model_type'])}\n"
        f"kf={row['kelly_fraction']} edge={row['buy_edge_threshold']} "
        f"side={row['trading_side']} km={row['kelly_mode']}"
        for _, row in top.iterrows()
    ]
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Robustness Score")
    ax.set_title(
        f"Top {top_n} Configs by Robustness Score\n"
        f"(max_loss=20%, min_profitable=3/{len(pnl_df['category'].unique())} categories, "
        f"all 9 categories, inferred+6h)",
        fontsize=12,
    )
    _save_fig(fig, "robustness_top_configs.png")

    # ── Plot 2: Robustness score distribution per model ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()  # type: ignore[union-attr]  # numpy ndarray from subplots
    for i, mt in enumerate(models[:6]):
        ax = axes[i] if i < len(axes) else None  # type: ignore[assignment]  # guarded below
        if ax is None:
            break
        mt_data = scores[scores["model_type"] == mt]
        if mt_data.empty:
            continue
        ax.hist(
            mt_data["robustness_score"],
            bins=30,
            color=get_model_color(mt),
            alpha=0.7,
            edgecolor="white",
        )
        ax.axvline(
            x=mt_data["robustness_score"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {mt_data['robustness_score'].mean():.3f}",
        )
        ax.set_title(MODEL_DISPLAY.get(mt, mt), fontsize=10)
        ax.set_xlabel("Robustness Score")
        ax.set_ylabel("# Configs")
        ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Robustness Score Distribution by Model", fontsize=14)
    _save_fig(fig, "robustness_distribution.png")

    # ── Plot 3: P&L vs Robustness Score scatter ──
    fig, ax = plt.subplots(figsize=(12, 8))
    for mt in models:
        mt_data = scores[scores["model_type"] == mt]
        ax.scatter(
            mt_data["robustness_score"],
            mt_data["agg_pnl"],
            alpha=0.3,
            s=10,
            color=get_model_color(mt),
            label=MODEL_DISPLAY.get(mt, mt),
        )

    # Highlight top 10
    for i, (_, row) in enumerate(scores.head(10).iterrows()):
        ax.scatter(
            row["robustness_score"],
            row["agg_pnl"],
            s=80,
            edgecolors="black",
            facecolors=get_model_color(row["model_type"]),
            linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            f"#{i + 1}",
            (row["robustness_score"], row["agg_pnl"]),
            fontsize=7,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Robustness Score")
    ax.set_ylabel("Aggregate P&L ($)")
    ax.set_title("P&L vs Robustness Score (all configs, all categories)", fontsize=12)
    ax.legend(fontsize=8, markerscale=3)
    _save_fig(fig, "robustness_vs_pnl.png")

    # ── Plot 4: Tolerance sensitivity — how top configs change ──
    # Show how top-1 config changes across tolerance parameters
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Vary max_loss_fraction
    loss_fracs = [0.10, 0.15, 0.20, 0.30, 0.50]
    for mt in models:
        top1_pnls = []
        top1_scores = []
        for lf in loss_fracs:
            s = compute_robustness_scores(
                pnl_df, max_loss_fraction=lf, min_profitable_fraction=0.33
            )
            mt_top = s[s["model_type"] == mt]
            if not mt_top.empty:
                top1_pnls.append(mt_top.iloc[0]["agg_pnl"])
                top1_scores.append(mt_top.iloc[0]["robustness_score"])
            else:
                top1_pnls.append(0)
                top1_scores.append(0)

        axes[0].plot(
            loss_fracs,
            top1_pnls,
            marker="o",
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
        )

    axes[0].set_xlabel("Max Loss Tolerance (fraction of bankroll)")
    axes[0].set_ylabel("Top-1 Config Aggregate P&L ($)")
    axes[0].set_title("Sensitivity to Loss Tolerance")
    axes[0].legend(fontsize=7)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel B: Vary min_profitable_fraction
    pf_fracs = [0.11, 0.22, 0.33, 0.44, 0.56, 0.67]
    for mt in models:
        top1_pnls = []
        for pf in pf_fracs:
            s = compute_robustness_scores(
                pnl_df, max_loss_fraction=0.20, min_profitable_fraction=pf
            )
            mt_top = s[s["model_type"] == mt]
            if not mt_top.empty:
                top1_pnls.append(mt_top.iloc[0]["agg_pnl"])
            else:
                top1_pnls.append(0)

        axes[1].plot(
            pf_fracs,
            top1_pnls,
            marker="o",
            label=MODEL_DISPLAY.get(mt, mt),
            color=get_model_color(mt),
        )

    axes[1].set_xlabel("Min Profitable Categories (fraction)")
    axes[1].set_ylabel("Top-1 Config Aggregate P&L ($)")
    axes[1].set_title("Sensitivity to Profitable Category Requirement")
    axes[1].legend(fontsize=7)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Tolerance Sensitivity: Best Config Per Model", fontsize=14)
    _save_fig(fig, "robustness_tolerance_sensitivity.png")

    # ── Print summary table ──
    print("\n" + "=" * 80)
    print("Top 10 Configs by Robustness Score (max_loss=20%, min_profit=3/9)")
    print("=" * 80)
    print(
        f"{'Rank':>4} {'Model':<18} {'kf':>5} {'edge':>6} {'side':>5} {'km':>12} "
        f"{'P&L':>8} {'Worst':>8} {'Prof':>5} {'Nb%':>5} {'Score':>7}"
    )
    print("-" * 100)
    for i, (_, row) in enumerate(scores.head(10).iterrows(), 1):
        print(
            f"{i:>4} {row['model_type']:<18} "
            f"{row['kelly_fraction']:>5.2f} {row['buy_edge_threshold']:>6.2f} "
            f"{row['trading_side']:>5} {row['kelly_mode']:>12} "
            f"${row['agg_pnl']:>+7.0f} ${row['worst_cat_pnl']:>+7.0f} "
            f"{row['n_profitable']:>2}/{row['n_categories']:<2} "
            f"{row['neighbor_stability']:>4.0%} "
            f"{row['robustness_score']:>7.4f}"
        )

    # ── Print tolerance sweep summary ──
    print("\n" + "=" * 80)
    print("Tolerance Sweep: Top-1 Config at Each Setting")
    print("=" * 80)
    for _, r in sweep_df[sweep_df["rank"] == 1].iterrows():
        print(
            f"  loss≤{r['max_loss_frac']:.0%}, profit≥{r['min_profit_frac']:.0%}: "
            f"{r['model_type']:<18} P&L=${r['agg_pnl']:>+7.0f} "
            f"worst=${r['worst_cat_pnl']:>+7.0f} "
            f"score={r['robustness_score']:.4f}"
        )


def main() -> None:
    print("=" * 70)
    print("Config Neighborhood & Robustness Analysis")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pnl_df = load_pnl()

    print("\n--- Config Neighborhood Sensitivity ---")
    analyze_config_neighborhood(pnl_df)

    print("\n--- Robustness Score Analysis ---")
    analyze_robustness(pnl_df)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
