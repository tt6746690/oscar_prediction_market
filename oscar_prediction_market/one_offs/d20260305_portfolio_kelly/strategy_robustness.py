"""Analysis 5 + 6: Strategy robustness checks for category allocation.

Cross-year stability, bootstrap strategy ranking, pairwise win rates,
leave-one-year-out validation, and entry-point robustness analysis.
These are the confidence-building checks borrowed from the config_selection_sweep
methodology — if the allocation ranking is fragile, we should not deploy it.

**Analysis 5a — Cross-Year Stability**:
Rank strategies by portfolio PnL independently in 2024 and 2025, then compute
Spearman ρ across years. High ρ means the best strategy in 2024 also wins in 2025.

**Analysis 5b — Bootstrap Strategy Ranking**:
Bootstrap-resample categories (with replacement) within each year, compute
weighted portfolio PnL, record which strategy wins. 2,000 iterations.
Also does entry-point bootstrap (resample which entry snapshots are included).

**Analysis 5c — Pairwise Win Rates**:
For ~10 representative strategies, compute P(A beats B) under category bootstrap.
A 60%+ pairwise win rate is a meaningful signal; 50-55% is noise.

**Analysis 5d — Leave-One-Year-Out**:
Pick the best strategy from 2024 data, evaluate on 2025 (and vice versa).
Tests whether in-sample selection transfers out-of-sample.

**Analysis 6 — Entry-Point Robustness**:
For the top strategies, compute uplift vs uniform at each entry snapshot.
Tests whether the benefit is concentrated in a few entries or spread broadly.

**Model Interaction Check**:
For each model, find the best allocation strategy. Tests whether the optimal
allocation differs across models — if it does, the allocation is model-dependent
and harder to trust.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260305_portfolio_kelly.strategy_robustness
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oscar_prediction_market.one_offs.d20260305_portfolio_kelly.shared import (
    ALL_MODELS,
    BANKROLL,
    YEARS,
    aggregate_combined,
    aggregate_to_year_level,
    df_to_md,
    ensure_output_dir,
    ensure_plot_dir,
    evaluate_all_strategies_by_year,
    evaluate_portfolio_at_entry,
    get_all_strategies,
    get_prospective_strategies,
    prepare_data,
    short_model,
)

# ─── Constants ───────────────────────────────────────────────────────────────

REC_EDGE = 0.20
REC_KF = 0.05
REC_MODEL = "avg_ensemble"

N_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = 42

# Representative subset of strategies for pairwise comparisons (~10).
# Chosen to span: baselines, signal types, aggressiveness levels, capping.
PAIRWISE_STRATEGIES = [
    "uniform",
    "equal_active",
    "ev_100",
    "ev_25",
    "edge_100",
    "edge_25",
    "capital_100",
    "maxedge_100",
    "maxabsedge_100",
    "npos_100",
    "ev_cap30",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    """Print a markdown section header."""
    print(f"\n{'=' * 80}")
    print(f"## {title}")
    print(f"{'=' * 80}\n")


def _subsection(title: str) -> None:
    """Print a markdown subsection header."""
    print(f"\n### {title}\n")


def _filter_rec_config(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to recommended config (edge=0.20, KF=0.05)."""
    return df[(df["buy_edge_threshold"] == REC_EDGE) & (df["kelly_fraction"] == REC_KF)].copy()


def _filter_model(df: pd.DataFrame, model: str = REC_MODEL) -> pd.DataFrame:
    """Filter to a single model_type."""
    return df[df["model_type"] == model].copy()


def _precompute_group_arrays(
    entry_group: pd.DataFrame,
) -> dict:
    """Pre-extract numpy arrays from an entry group for fast bootstrap.

    Avoids repeated pandas indexing in the tight bootstrap loop by converting
    all needed columns to contiguous numpy arrays once.
    """
    from oscar_prediction_market.one_offs.d20260305_portfolio_kelly.shared import (
        SIGNAL_NAMES,
    )

    return {
        "pnl": entry_group["total_pnl"].values.astype(np.float64),
        "active": entry_group["is_active"].values.astype(bool),
        "signals": {
            sig: np.maximum(entry_group[sig].values.astype(np.float64), 0)
            for sig in SIGNAL_NAMES
            if sig in entry_group.columns
        },
        "n": len(entry_group),
    }


def _compute_weights_numpy(
    n: int,
    pnl: np.ndarray,
    active: np.ndarray,
    signals: dict[str, np.ndarray],
    strategy: str,
    aggressiveness: float,
    cap: float | None,
) -> np.ndarray:
    """Pure-numpy weight computation — same logic as shared.compute_weights.

    ~10-20x faster than the pandas version for small arrays (n ≈ 8) because
    it avoids pandas Series construction, indexing, and alignment overhead.
    """
    if strategy == "uniform":
        raw = np.ones(n)
    elif strategy == "equal_active":
        n_active = active.sum()
        if n_active == 0:
            raw = np.ones(n)
        else:
            raw = np.zeros(n)
            raw[active] = n / n_active
    elif strategy == "oracle":
        pnl_clipped = np.maximum(pnl, 0)
        total = pnl_clipped.sum()
        raw = np.ones(n) if total == 0 else pnl_clipped / total * n
    else:
        # Signal-proportional, restricted to active categories
        signal_vals = signals[strategy] * active.astype(np.float64)
        total = signal_vals.sum()
        if total == 0:
            n_active = active.sum()
            if n_active == 0:
                raw = np.ones(n)
            else:
                raw = np.zeros(n)
                raw[active] = n / n_active
        else:
            raw = signal_vals / total * n

    weights = (1.0 - aggressiveness) + aggressiveness * raw

    if cap is not None:
        max_w = cap * n
        for _ in range(20):
            over = weights > max_w
            if not over.any():
                break
            excess = (weights[over] - max_w).sum()
            weights[over] = max_w
            n_under = (~over).sum()
            if n_under == 0:
                break
            weights[~over] += excess / n_under

    return weights


def _bootstrap_pnl_all_strategies(
    group_arrays: dict,
    strategies: dict[str, dict],
    strat_names: list[str],
    indices: np.ndarray,
) -> np.ndarray:
    """Compute portfolio PnL for all strategies from one bootstrap resample.

    Pure numpy — no pandas in the inner loop. Returns shape (n_strats,).
    """
    pnl = group_arrays["pnl"][indices]
    active = group_arrays["active"][indices]
    signals = {k: v[indices] for k, v in group_arrays["signals"].items()}
    n = group_arrays["n"]

    result = np.empty(len(strat_names))
    for s_idx, sname in enumerate(strat_names):
        params = strategies[sname]
        w = _compute_weights_numpy(n, pnl, active, signals, **params)
        result[s_idx] = (pnl * w).sum()

    return result


def plot_worst_rank_scatter(stability_df: pd.DataFrame) -> None:
    """Scatter plot of 2024 rank vs 2025 rank for each strategy.

    Points near the diagonal have consistent ranking across years.
    Points far off-diagonal are year-specific performers.
    """
    plot_dir = ensure_plot_dir("robustness")
    fig, ax = plt.subplots(figsize=(10, 10))

    card_strats = {"ev_100", "maxedge_100", "maxabsedge_100", "edge_100", "capital_100", "npos_100"}
    baselines = {"uniform", "equal_active"}

    for _, row in stability_df.iterrows():
        strat = row["strategy"]
        r24 = row["rank_2024"]
        r25 = row["rank_2025"]

        if strat in card_strats:
            ax.scatter(
                r24,
                r25,
                s=120,
                c="#e74c3c",
                zorder=5,
                edgecolors="black",
                linewidths=0.8,
            )
        elif strat in baselines:
            ax.scatter(
                r24,
                r25,
                s=120,
                c="#3498db",
                zorder=5,
                marker="s",
                edgecolors="black",
                linewidths=0.8,
            )
        else:
            ax.scatter(r24, r25, s=40, c="#95a5a6", zorder=3, alpha=0.6)

        ax.annotate(
            strat,
            (r24, r25),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            alpha=0.8,
        )

    # Diagonal line y=x
    max_rank = max(stability_df["rank_2024"].max(), stability_df["rank_2025"].max())
    ax.plot([0, max_rank + 1], [0, max_rank + 1], "k--", alpha=0.3, linewidth=1)

    # Annotations for leaders
    avg_leader = stability_df.loc[stability_df["avg_rank"].idxmin()]
    worst_leader = stability_df.loc[stability_df["worst_rank"].idxmin()]
    annotation_text = (
        f"Best avg rank: {avg_leader['strategy']} ({avg_leader['avg_rank']:.1f})\n"
        f"Best worst rank: {worst_leader['strategy']} ({worst_leader['worst_rank']:.0f})"
    )
    ax.text(
        0.98,
        0.02,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
    )

    ax.set_xlabel("2024 Rank (lower = better)", fontsize=11)
    ax.set_ylabel("2025 Rank (lower = better)", fontsize=11)
    ax.set_title("2024 vs 2025 Strategy Rankings", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=10,
            markeredgecolor="black",
            label="Card strategies",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#3498db",
            markersize=10,
            markeredgecolor="black",
            label="Baselines",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#95a5a6",
            markersize=7,
            alpha=0.6,
            label="Other",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_dir / "worst_rank_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'worst_rank_scatter.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 5a: Cross-Year Strategy Stability
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_cross_year_stability(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Rank strategies by portfolio PnL independently in each year, compute Spearman ρ.

    Tests whether strategy rankings are stable across years. We do this twice:
    1. At recommended config (edge=0.20, KF=0.05) for avg_ensemble only.
    2. Across all configs (mean PnL across configs) for avg_ensemble.

    High Spearman ρ (>0.7) means the best strategy is consistently good.
    Low ρ (<0.3) means ranking is year-specific and not trustworthy.
    """
    _section("Analysis 5a: Cross-Year Strategy Stability")

    strategies = get_prospective_strategies()
    results = evaluate_all_strategies_by_year(data_by_year, strategies)
    year_level = aggregate_to_year_level(results)

    # ── Part 1: Recommended config only ──
    _subsection("At Recommended Config (avg_ensemble, edge=0.20, KF=0.05)")

    rec = year_level[
        (year_level["model_type"] == REC_MODEL)
        & (year_level["buy_edge_threshold"] == REC_EDGE)
        & (year_level["kelly_fraction"] == REC_KF)
    ]

    year_pnls: dict[int, pd.Series] = {}
    for year in YEARS:
        yr = rec[rec["year"] == year].set_index("strategy")["portfolio_pnl"]
        year_pnls[year] = yr

    # Build cross-year table
    rows: list[dict] = []
    for strat in strategies:
        pnl_2024 = year_pnls[2024].get(strat, 0.0)
        pnl_2025 = year_pnls[2025].get(strat, 0.0)
        rows.append(
            {
                "strategy": strat,
                "pnl_2024": pnl_2024,
                "pnl_2025": pnl_2025,
                "pnl_combined": pnl_2024 + pnl_2025,
            }
        )

    stability_df = pd.DataFrame(rows).sort_values("pnl_combined", ascending=False)

    # Rank within each year
    stability_df["rank_2024"] = stability_df["pnl_2024"].rank(ascending=False).astype(int)
    stability_df["rank_2025"] = stability_df["pnl_2025"].rank(ascending=False).astype(int)
    stability_df["avg_rank"] = (stability_df["rank_2024"] + stability_df["rank_2025"]) / 2
    stability_df["worst_rank"] = stability_df[["rank_2024", "rank_2025"]].max(axis=1)

    # Sort by avg_rank (year-balanced metric)
    stability_df = stability_df.sort_values("avg_rank")

    # Spearman ρ
    rho_result = spearmanr(
        stability_df["rank_2024"].to_numpy(),
        stability_df["rank_2025"].to_numpy(),
    )
    rho = float(rho_result.statistic)  # type: ignore[union-attr]
    pval = float(rho_result.pvalue)  # type: ignore[union-attr]

    strength = "Strong" if abs(rho) > 0.7 else "Moderate" if abs(rho) > 0.4 else "Weak"
    print(f"**Spearman ρ = {rho:.3f}** (p = {pval:.4f}) — {strength} cross-year consistency\n")

    display = stability_df[
        [
            "strategy",
            "rank_2024",
            "rank_2025",
            "avg_rank",
            "worst_rank",
            "pnl_2024",
            "pnl_2025",
            "pnl_combined",
        ]
    ].copy()
    display["pnl_2024"] = display["pnl_2024"].map(lambda x: f"${x:,.0f}")
    display["pnl_2025"] = display["pnl_2025"].map(lambda x: f"${x:,.0f}")
    display["pnl_combined"] = display["pnl_combined"].map(lambda x: f"${x:,.0f}")
    print(df_to_md(display, float_fmt=".0f"))
    print()

    # ── Part 2: Across all configs (avg_ensemble) ──
    _subsection("Across All Configs (avg_ensemble, mean PnL)")

    ens = year_level[year_level["model_type"] == REC_MODEL]
    mean_by_strat_year = ens.groupby(["year", "strategy"])["portfolio_pnl"].mean().reset_index()

    all_config_rows: list[dict] = []
    for strat in strategies:
        pnl_2024 = mean_by_strat_year[
            (mean_by_strat_year["strategy"] == strat) & (mean_by_strat_year["year"] == 2024)
        ]["portfolio_pnl"]
        pnl_2025 = mean_by_strat_year[
            (mean_by_strat_year["strategy"] == strat) & (mean_by_strat_year["year"] == 2025)
        ]["portfolio_pnl"]
        p24 = float(pnl_2024.iloc[0]) if len(pnl_2024) > 0 else 0.0
        p25 = float(pnl_2025.iloc[0]) if len(pnl_2025) > 0 else 0.0
        all_config_rows.append(
            {
                "strategy": strat,
                "mean_pnl_2024": p24,
                "mean_pnl_2025": p25,
                "mean_combined": p24 + p25,
            }
        )

    allcfg_df = pd.DataFrame(all_config_rows).sort_values("mean_combined", ascending=False)
    allcfg_df["rank_2024"] = allcfg_df["mean_pnl_2024"].rank(ascending=False).astype(int)
    allcfg_df["rank_2025"] = allcfg_df["mean_pnl_2025"].rank(ascending=False).astype(int)

    rho2_result = spearmanr(
        allcfg_df["rank_2024"].to_numpy(),
        allcfg_df["rank_2025"].to_numpy(),
    )
    rho2 = float(rho2_result.statistic)  # type: ignore[union-attr]
    pval2 = float(rho2_result.pvalue)  # type: ignore[union-attr]

    strength2 = "Strong" if abs(rho2) > 0.7 else "Moderate" if abs(rho2) > 0.4 else "Weak"
    print(
        f"**Spearman ρ = {rho2:.3f}** (p = {pval2:.4f}) — "
        f"{strength2} cross-year consistency (all configs)\n"
    )

    display2 = (
        allcfg_df[
            [
                "strategy",
                "rank_2024",
                "rank_2025",
                "mean_pnl_2024",
                "mean_pnl_2025",
                "mean_combined",
            ]
        ]
        .head(15)
        .copy()
    )
    display2["mean_pnl_2024"] = display2["mean_pnl_2024"].map(lambda x: f"${x:,.0f}")
    display2["mean_pnl_2025"] = display2["mean_pnl_2025"].map(lambda x: f"${x:,.0f}")
    display2["mean_combined"] = display2["mean_combined"].map(lambda x: f"${x:,.0f}")
    print(df_to_md(display2, float_fmt=".0f"))
    print()

    # ── Worst-rank scatter plot ──
    plot_worst_rank_scatter(stability_df)

    return stability_df


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 5b: Bootstrap Strategy Ranking
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_bootstrap_ranking(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Bootstrap-resample categories to test ranking stability.

    For each bootstrap iteration:
    1. Resample categories with replacement (within each year/entry).
    2. Compute weighted portfolio PnL for every strategy.
    3. Rank strategies by combined PnL and record the winner.

    After N_BOOTSTRAP iterations, report:
    - Mean rank per strategy (lower = better)
    - % of iterations where each strategy was rank 1 (most dominant)
    - % top 3 and top 5

    Also performs entry-point bootstrap: resample which entry snapshots are
    included (with replacement), then sum PnL across the resampled entries.
    This tests whether dominance depends on specific entry points.
    """
    _section("Analysis 5b: Bootstrap Strategy Ranking")

    strategies = get_prospective_strategies()
    strat_names = sorted(strategies.keys())
    n_strats = len(strat_names)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Prepare data: for avg_ensemble at recommended config, get per-entry category groups
    # Each "group" = categories at one entry snapshot in one year
    groups_by_year: dict[int, list[pd.DataFrame]] = {}
    group_arrays_by_year: dict[int, list[dict]] = {}
    for year in YEARS:
        df = data_by_year[year]
        df = _filter_model(df, REC_MODEL)
        df = _filter_rec_config(df)
        year_groups = []
        year_arrays = []
        for _entry, grp in df.groupby("entry_snapshot"):
            gdf = grp.copy()
            year_groups.append(gdf)
            year_arrays.append(_precompute_group_arrays(gdf))
        groups_by_year[year] = year_groups
        group_arrays_by_year[year] = year_arrays

    # ── Category bootstrap ──
    _subsection("Category Bootstrap (resample categories within each entry)")
    print(f"Running {N_BOOTSTRAP} bootstrap iterations...")

    # Track rank per strategy per iteration
    rank_matrix = np.zeros((N_BOOTSTRAP, n_strats))

    for b in range(N_BOOTSTRAP):
        # For each strategy, compute combined PnL across years and entries
        strat_pnls = np.zeros(n_strats)

        for year in YEARS:
            for ga in group_arrays_by_year[year]:
                n_cats = ga["n"]
                indices = rng.choice(n_cats, size=n_cats, replace=True)
                strat_pnls += _bootstrap_pnl_all_strategies(ga, strategies, strat_names, indices)

        # Rank: 1 = best (highest PnL)
        # Use negative PnL so argsort gives rank-1 to highest
        order = np.argsort(-strat_pnls)
        ranks = np.empty(n_strats, dtype=np.float64)
        ranks[order] = np.arange(1, n_strats + 1, dtype=np.float64)
        rank_matrix[b, :] = ranks

    # Compute summary statistics
    cat_boot_rows: list[dict] = []
    for s_idx, strat_name in enumerate(strat_names):
        ranks = rank_matrix[:, s_idx]
        cat_boot_rows.append(
            {
                "strategy": strat_name,
                "mean_rank": ranks.mean(),
                "median_rank": np.median(ranks),
                "pct_rank1": (ranks == 1).mean(),
                "pct_top3": (ranks <= 3).mean(),
                "pct_top5": (ranks <= 5).mean(),
                "bootstrap_type": "category",
            }
        )

    cat_boot = pd.DataFrame(cat_boot_rows).sort_values("mean_rank")

    display = (
        cat_boot[["strategy", "mean_rank", "pct_rank1", "pct_top3", "pct_top5"]].head(15).copy()
    )
    display["mean_rank"] = display["mean_rank"].map(lambda x: f"{x:.1f}")
    display["pct_rank1"] = display["pct_rank1"].map(lambda x: f"{x:.1%}")
    display["pct_top3"] = display["pct_top3"].map(lambda x: f"{x:.1%}")
    display["pct_top5"] = display["pct_top5"].map(lambda x: f"{x:.1%}")
    print(df_to_md(display, float_fmt=".1f"))
    print()

    # ── Entry-point bootstrap ──
    _subsection("Entry-Point Bootstrap (resample which entry snapshots are included)")
    print(f"Running {N_BOOTSTRAP} bootstrap iterations...")

    # Pre-compute per-entry PnL for each strategy in each year as numpy arrays
    # Shape: entry_pnl_arrays[year] = np.ndarray (n_entries, n_strats)
    entry_pnl_arrays: dict[int, np.ndarray] = {}

    for year in YEARS:
        n_entries = len(group_arrays_by_year[year])
        arr = np.zeros((n_entries, n_strats))
        for e_idx, ga in enumerate(group_arrays_by_year[year]):
            n_cats = ga["n"]
            identity_idx = np.arange(n_cats)  # no resampling — use original data
            arr[e_idx, :] = _bootstrap_pnl_all_strategies(ga, strategies, strat_names, identity_idx)
        entry_pnl_arrays[year] = arr

    entry_rank_matrix = np.zeros((N_BOOTSTRAP, n_strats))

    for b in range(N_BOOTSTRAP):
        strat_pnls = np.zeros(n_strats)
        for year in YEARS:
            n_entries = entry_pnl_arrays[year].shape[0]
            # Resample entry indices with replacement
            entry_indices = rng.choice(n_entries, size=n_entries, replace=True)
            # Sum PnLs for resampled entries — fully vectorized
            strat_pnls += entry_pnl_arrays[year][entry_indices].sum(axis=0)

        order = np.argsort(-strat_pnls)
        ranks = np.empty(n_strats, dtype=np.float64)
        ranks[order] = np.arange(1, n_strats + 1, dtype=np.float64)
        entry_rank_matrix[b, :] = ranks

    entry_boot_rows: list[dict] = []
    for s_idx, strat_name in enumerate(strat_names):
        ranks = entry_rank_matrix[:, s_idx]
        entry_boot_rows.append(
            {
                "strategy": strat_name,
                "mean_rank": ranks.mean(),
                "median_rank": np.median(ranks),
                "pct_rank1": (ranks == 1).mean(),
                "pct_top3": (ranks <= 3).mean(),
                "pct_top5": (ranks <= 5).mean(),
                "bootstrap_type": "entry_point",
            }
        )

    entry_boot = pd.DataFrame(entry_boot_rows).sort_values("mean_rank")

    display = (
        entry_boot[["strategy", "mean_rank", "pct_rank1", "pct_top3", "pct_top5"]].head(15).copy()
    )
    display["mean_rank"] = display["mean_rank"].map(lambda x: f"{x:.1f}")
    display["pct_rank1"] = display["pct_rank1"].map(lambda x: f"{x:.1%}")
    display["pct_top3"] = display["pct_top3"].map(lambda x: f"{x:.1%}")
    display["pct_top5"] = display["pct_top5"].map(lambda x: f"{x:.1%}")
    print(df_to_md(display, float_fmt=".1f"))
    print()

    # ── Plot: bootstrap rank-1 probabilities ──
    plot_dir = ensure_plot_dir("robustness")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Category bootstrap
    cat_top = cat_boot.sort_values("pct_rank1", ascending=False).head(12)
    ax1.barh(range(len(cat_top)), cat_top["pct_rank1"].values, color="#3498db", alpha=0.8)
    ax1.set_yticks(range(len(cat_top)))
    ax1.set_yticklabels(cat_top["strategy"].values, fontsize=8)
    ax1.set_xlabel("P(Rank 1)")
    ax1.set_title("Category Bootstrap — Rank-1 Probability")
    ax1.invert_yaxis()
    for i, v in enumerate(cat_top["pct_rank1"].values):
        ax1.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)

    # Entry-point bootstrap
    entry_top = entry_boot.sort_values("pct_rank1", ascending=False).head(12)
    ax2.barh(range(len(entry_top)), entry_top["pct_rank1"].values, color="#e67e22", alpha=0.8)
    ax2.set_yticks(range(len(entry_top)))
    ax2.set_yticklabels(entry_top["strategy"].values, fontsize=8)
    ax2.set_xlabel("P(Rank 1)")
    ax2.set_title("Entry-Point Bootstrap — Rank-1 Probability")
    ax2.invert_yaxis()
    for i, v in enumerate(entry_top["pct_rank1"].values):
        ax2.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)

    fig.suptitle(
        f"Bootstrap Strategy Ranking — {short_model(REC_MODEL)}, "
        f"edge={REC_EDGE}, KF={REC_KF} ({N_BOOTSTRAP:,} iterations)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "bootstrap_rank1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'bootstrap_rank1.png'}")

    # Combine results for CSV
    combined_boot = pd.concat([cat_boot, entry_boot], ignore_index=True)
    return combined_boot


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 5c: Pairwise Win Rates
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_pairwise_winrates(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Compute P(A beats B) for each pair of representative strategies.

    Under category bootstrap (resample categories with replacement), for each
    pair we count how often strategy A's portfolio PnL exceeds strategy B's.

    A pairwise win rate of 60%+ is a meaningful signal. 50-55% is
    indistinguishable from coin-flip noise.

    Uses avg_ensemble at recommended config across both years.
    """
    _section("Analysis 5c: Pairwise Win Rates")

    all_strategies = get_all_strategies()

    # Filter to strategies that actually exist in the registry
    test_strategies = [s for s in PAIRWISE_STRATEGIES if s in all_strategies]
    n_strats = len(test_strategies)

    print(f"Testing {n_strats} strategies: {test_strategies}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print()

    rng = np.random.default_rng(BOOTSTRAP_SEED + 100)

    # Prepare entry groups for avg_ensemble at recommended config
    # Pre-compute numpy arrays for fast bootstrap
    group_arrays_by_year: dict[int, list[dict]] = {}
    for year in YEARS:
        df = _filter_model(_filter_rec_config(data_by_year[year]), REC_MODEL)
        group_arrays_by_year[year] = [
            _precompute_group_arrays(grp) for _, grp in df.groupby("entry_snapshot")
        ]

    # Run bootstrap and record PnL per strategy per iteration
    pnl_matrix = np.zeros((N_BOOTSTRAP, n_strats))

    for b in range(N_BOOTSTRAP):
        for year in YEARS:
            for ga in group_arrays_by_year[year]:
                n_cats = ga["n"]
                indices = rng.choice(n_cats, size=n_cats, replace=True)
                pnl_matrix[b, :] += _bootstrap_pnl_all_strategies(
                    ga, all_strategies, test_strategies, indices
                )

    # Compute pairwise win rates: P(row beats column)
    winrate_matrix = np.zeros((n_strats, n_strats))
    for i in range(n_strats):
        for j in range(n_strats):
            if i == j:
                winrate_matrix[i, j] = 0.5
            else:
                winrate_matrix[i, j] = (pnl_matrix[:, i] > pnl_matrix[:, j]).mean()

    # Build DataFrame
    winrate_df = pd.DataFrame(
        winrate_matrix,
        index=test_strategies,
        columns=test_strategies,
    )

    # Print matrix
    display = winrate_df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: f"{x:.0%}")
    print("**Pairwise Win Rates** — P(row beats column):\n")
    print(display.to_markdown())
    print()

    # Summary: mean win rate per strategy (how often it beats the rest)
    mean_winrate = winrate_df.mean(axis=1).sort_values(ascending=False)
    print("Mean win rate vs. field:")
    for strat, wr in mean_winrate.items():
        print(f"  {strat:<20} {wr:.1%}")
    print()

    # ── Plot: pairwise win rate heatmap ──
    plot_dir = ensure_plot_dir("robustness")
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(winrate_matrix, cmap="RdYlGn", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks(range(n_strats))
    ax.set_xticklabels(test_strategies, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_strats))
    ax.set_yticklabels(test_strategies, fontsize=8)
    ax.set_title(
        f"Pairwise Win Rates — P(row > col)\n"
        f"{short_model(REC_MODEL)}, edge={REC_EDGE}, KF={REC_KF}, "
        f"{N_BOOTSTRAP:,} category bootstraps"
    )

    # Annotate cells
    for i in range(n_strats):
        for j in range(n_strats):
            val = winrate_matrix[i, j]
            color = "white" if abs(val - 0.5) > 0.15 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Win Rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "pairwise_winrate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'pairwise_winrate.png'}")

    # Flatten for CSV: (strategy_a, strategy_b, winrate)
    flat_rows: list[dict] = []
    for i, sa in enumerate(test_strategies):
        for j, sb in enumerate(test_strategies):
            flat_rows.append(
                {
                    "strategy_a": sa,
                    "strategy_b": sb,
                    "winrate_a_over_b": winrate_matrix[i, j],
                }
            )

    return pd.DataFrame(flat_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 5d: Leave-One-Year-Out
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_leave_one_year_out(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Pick the best strategy in one year, evaluate on the other.

    The key question: if we select the best strategy from 2024 in-sample,
    how close is its 2025 out-of-sample PnL to the actual 2025 best?
    This is the classic train/test validation applied to strategy selection.

    "Regret" = best_test_year_PnL - transferred_strategy_PnL. Low regret
    means the in-sample choice transfers well.
    """
    _section("Analysis 5d: Leave-One-Year-Out Validation")

    strategies = get_prospective_strategies()
    results = evaluate_all_strategies_by_year(data_by_year, strategies)
    year_level = aggregate_to_year_level(results)

    # Focus on avg_ensemble at recommended config
    rec = year_level[
        (year_level["model_type"] == REC_MODEL)
        & (year_level["buy_edge_threshold"] == REC_EDGE)
        & (year_level["kelly_fraction"] == REC_KF)
    ]

    year_strat_pnl: dict[int, dict[str, float]] = {}
    for year in YEARS:
        yr = rec[rec["year"] == year].set_index("strategy")["portfolio_pnl"]
        year_strat_pnl[year] = dict(yr)  # type: ignore[arg-type]  # Series items

    rows: list[dict] = []

    for train_year in YEARS:
        test_year = [y for y in YEARS if y != train_year][0]

        # Best strategy in train year
        train_pnls = year_strat_pnl[train_year]
        best_train_strat = max(train_pnls, key=train_pnls.get)  # type: ignore[arg-type]
        best_train_pnl = train_pnls[best_train_strat]

        # Evaluate that strategy on test year
        transferred_pnl = year_strat_pnl[test_year].get(best_train_strat, 0.0)

        # Best strategy on test year (oracle)
        test_pnls = year_strat_pnl[test_year]
        best_test_strat = max(test_pnls, key=test_pnls.get)  # type: ignore[arg-type]
        best_test_pnl = test_pnls[best_test_strat]

        # Uniform baseline on test year
        uniform_test_pnl = year_strat_pnl[test_year].get("uniform", 0.0)

        regret = best_test_pnl - transferred_pnl
        uplift_vs_uniform = transferred_pnl - uniform_test_pnl

        rows.append(
            {
                "train_year": train_year,
                "test_year": test_year,
                "best_train_strategy": best_train_strat,
                "train_pnl": best_train_pnl,
                "transferred_pnl": transferred_pnl,
                "best_test_strategy": best_test_strat,
                "best_test_pnl": best_test_pnl,
                "uniform_test_pnl": uniform_test_pnl,
                "regret": regret,
                "uplift_vs_uniform": uplift_vs_uniform,
            }
        )

    loo_df = pd.DataFrame(rows)

    # Print results
    for _, row in loo_df.iterrows():
        print(f"**Train on {row['train_year']}, test on {row['test_year']}:**")
        print(
            f"  Best in-sample strategy: {row['best_train_strategy']} "
            f"(train PnL: ${row['train_pnl']:,.0f})"
        )
        print(f"  Transferred test PnL:    ${row['transferred_pnl']:,.0f}")
        print(
            f"  Best test-year strategy: {row['best_test_strategy']} "
            f"(test PnL: ${row['best_test_pnl']:,.0f})"
        )
        print(f"  Uniform test PnL:        ${row['uniform_test_pnl']:,.0f}")
        print(f"  Regret (best - transferred): ${row['regret']:,.0f}")
        print(f"  Uplift vs uniform:       ${row['uplift_vs_uniform']:,.0f}")
        print()

    # Summary assessment
    avg_regret = loo_df["regret"].mean()
    avg_uplift = loo_df["uplift_vs_uniform"].mean()
    print(f"**Average regret: ${avg_regret:,.0f}** (low = good transfer, high = year-specific)")
    print(
        f"**Average uplift vs uniform: ${avg_uplift:,.0f}** "
        f"(positive = strategy selection adds value even OOS)"
    )
    print()

    return loo_df


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 6: Entry-Point Robustness
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_entry_robustness(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """For top strategies, compute uplift vs uniform at each entry snapshot.

    Questions answered:
    - What % of entry points show positive uplift for each strategy?
    - Is the benefit concentrated in a few entries or spread across all?
    - What are the mean and std of uplift across entry points?

    A robust strategy should improve PnL at most entry points, not just a few.
    If uplift is concentrated (e.g., only works at early entries), it may be
    timing-dependent and less trustworthy.
    """
    _section("Analysis 6: Entry-Point Robustness")

    strategies = get_prospective_strategies()

    # Pick top ~10 strategies by combined PnL (pre-computed for efficiency)
    # First compute combined PnL to select top strategies
    results = evaluate_all_strategies_by_year(data_by_year, strategies)
    year_level = aggregate_to_year_level(results)
    combined = aggregate_combined(year_level)

    # Filter to avg_ensemble at recommended config
    rec_combined = combined[
        (combined["model_type"] == REC_MODEL)
        & (combined["buy_edge_threshold"] == REC_EDGE)
        & (combined["kelly_fraction"] == REC_KF)
    ]
    # Always include the 5 agg=100 uncapped strategies (strategy card candidates)
    card_strats = ["ev_100", "maxedge_100", "maxabsedge_100", "edge_100", "capital_100", "npos_100"]
    top_strats = (
        rec_combined[rec_combined["strategy"] != "uniform"]
        .sort_values("pnl_combined", ascending=False)
        .head(10)["strategy"]
        .tolist()
    )
    # Merge: card strategies first, then top performers not already included
    merged = card_strats + [s for s in top_strats if s not in card_strats]
    # Always include uniform as the baseline reference
    test_strats = ["uniform"] + merged[:12]  # cap at 12 signal strategies

    print(f"Testing {len(test_strats)} strategies: {test_strats[:5]}...")
    print()

    # Compute per-entry PnL for each strategy, year
    entry_rows: list[dict] = []

    for year in YEARS:
        df = _filter_model(_filter_rec_config(data_by_year[year]), REC_MODEL)

        for entry, grp in df.groupby("entry_snapshot"):
            for strat_name in test_strats:
                if strat_name not in strategies:
                    continue
                pnl = evaluate_portfolio_at_entry(grp, strategies[strat_name])
                entry_rows.append(
                    {
                        "year": year,
                        "entry_snapshot": entry,
                        "strategy": strat_name,
                        "portfolio_pnl": pnl,
                    }
                )

    entry_df = pd.DataFrame(entry_rows)

    # Compute uplift vs uniform at each entry point
    uniform_pnl = entry_df[entry_df["strategy"] == "uniform"][
        ["year", "entry_snapshot", "portfolio_pnl"]
    ].rename(columns={"portfolio_pnl": "uniform_pnl"})

    uplift_df = entry_df.merge(uniform_pnl, on=["year", "entry_snapshot"])
    uplift_df["uplift"] = uplift_df["portfolio_pnl"] - uplift_df["uniform_pnl"]
    uplift_df["uplift_pct"] = np.where(
        uplift_df["uniform_pnl"].abs() > 1,
        uplift_df["uplift"] / uplift_df["uniform_pnl"].abs() * 100,
        0.0,
    )

    # Summarize per strategy
    summary_rows: list[dict] = []
    for strat_name in test_strats:
        if strat_name == "uniform":
            continue
        sdf = uplift_df[uplift_df["strategy"] == strat_name]
        if len(sdf) == 0:
            continue
        n_positive = (sdf["uplift"] > 0).sum()
        n_total = len(sdf)
        summary_rows.append(
            {
                "strategy": strat_name,
                "pct_positive_uplift": n_positive / n_total,
                "n_positive": n_positive,
                "n_total": n_total,
                "mean_uplift": sdf["uplift"].mean(),
                "std_uplift": sdf["uplift"].std(),
                "min_uplift": sdf["uplift"].min(),
                "max_uplift": sdf["uplift"].max(),
                "mean_uplift_pct": sdf["uplift_pct"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean_uplift", ascending=False)

    _subsection("Per-Strategy Entry-Point Summary")
    display = summary.copy()
    display["pct_positive_uplift"] = display["pct_positive_uplift"].map(lambda x: f"{x:.0%}")
    display["mean_uplift"] = display["mean_uplift"].map(lambda x: f"${x:,.0f}")
    display["std_uplift"] = display["std_uplift"].map(lambda x: f"${x:,.0f}")
    display["min_uplift"] = display["min_uplift"].map(lambda x: f"${x:,.0f}")
    display["max_uplift"] = display["max_uplift"].map(lambda x: f"${x:,.0f}")
    display["mean_uplift_pct"] = display["mean_uplift_pct"].map(lambda x: f"{x:+.1f}%")
    print(
        df_to_md(
            display[
                [
                    "strategy",
                    "pct_positive_uplift",
                    "mean_uplift",
                    "std_uplift",
                    "min_uplift",
                    "max_uplift",
                    "mean_uplift_pct",
                ]
            ],
            float_fmt=".1f",
        )
    )
    print()

    # Interpretation: is benefit concentrated or spread?
    _subsection("Concentration Analysis")
    for _, row in summary.iterrows():
        pct = row["pct_positive_uplift"]
        strat = row["strategy"]
        if pct >= 0.75:
            verdict = "Broadly positive — benefit spread across most entry points"
        elif pct >= 0.5:
            verdict = "Mixed — benefit at roughly half of entries"
        else:
            verdict = "Concentrated — benefit at few entries, risky to deploy"
        print(f"  {strat:<20} {pct:.0%} positive → {verdict}")
    print()

    # ── Per-year breakdown ──
    _subsection("Per-Year Entry-Point Detail")
    for year in YEARS:
        year_uplift = uplift_df[(uplift_df["year"] == year) & (uplift_df["strategy"] != "uniform")]
        year_summary = (
            year_uplift.groupby("strategy")
            .agg(
                mean_uplift=("uplift", "mean"),
                pct_positive=("uplift", lambda x: (x > 0).mean()),
                n_entries=("entry_snapshot", "nunique"),
            )
            .sort_values("mean_uplift", ascending=False)
            .reset_index()
        )
        print(f"**{year}** ({year_summary['n_entries'].iloc[0]} entry points):")
        display = year_summary.head(10).copy()
        display["mean_uplift"] = display["mean_uplift"].map(lambda x: f"${x:,.0f}")
        display["pct_positive"] = display["pct_positive"].map(lambda x: f"{x:.0%}")
        print(df_to_md(display[["strategy", "mean_uplift", "pct_positive"]], float_fmt=".1f"))
        print()

    # ── Plot: entry-point uplift distributions for top strategies ──
    plot_dir = ensure_plot_dir("robustness")
    top_5_strats = summary.head(5)["strategy"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, year in zip(axes, YEARS, strict=True):
        year_data = uplift_df[(uplift_df["year"] == year)]
        x_positions = np.arange(len(top_5_strats))

        for i, strat in enumerate(top_5_strats):
            sdf = year_data[year_data["strategy"] == strat]
            uplifts = sdf["uplift"].values
            # Scatter with jitter
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(uplifts))
            colors = ["#2ecc71" if u > 0 else "#e74c3c" for u in uplifts]
            ax.scatter(
                x_positions[i] + jitter, uplifts, c=colors, alpha=0.6, s=40, edgecolors="white"
            )
            # Mean marker
            ax.scatter(x_positions[i], uplifts.mean(), marker="D", c="black", s=80, zorder=5)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(top_5_strats, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Uplift vs. Uniform ($)")
        ax.set_title(f"{year} — Entry-Point Uplift")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Entry-Point Uplift Distribution — {short_model(REC_MODEL)}, edge={REC_EDGE}, KF={REC_KF}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "entry_uplift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'entry_uplift.png'}")

    return uplift_df


# ═══════════════════════════════════════════════════════════════════════════════
# Model Interaction Check
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_model_interaction(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """For each model, find its best allocation strategy and compare.

    The question: does the optimal allocation strategy differ across models?
    If the same strategy wins for all models, our recommendation is robust to
    model choice. If different models prefer different allocations, the
    allocation choice is entangled with model selection.

    For each model, we:
    1. Find the model's best config (edge × KF with highest combined PnL under uniform).
    2. At that config, rank all allocation strategies by combined PnL.
    3. Report the best strategy and its uplift vs. uniform.
    """
    _section("Model Interaction Check")

    strategies = get_prospective_strategies()
    results = evaluate_all_strategies_by_year(data_by_year, strategies)
    year_level = aggregate_to_year_level(results)
    combined = aggregate_combined(year_level)

    rows: list[dict] = []
    for model in ALL_MODELS:
        model_data = combined[combined["model_type"] == model]
        if len(model_data) == 0:
            continue

        # Find best config under uniform allocation
        uniform_data = model_data[model_data["strategy"] == "uniform"]
        if len(uniform_data) == 0:
            continue
        best_uniform = uniform_data.loc[uniform_data["pnl_combined"].idxmax()]
        best_edge = best_uniform["buy_edge_threshold"]
        best_kf = best_uniform["kelly_fraction"]
        uniform_pnl = best_uniform["pnl_combined"]

        # At that config, find best allocation strategy
        at_config = model_data[
            (model_data["buy_edge_threshold"] == best_edge)
            & (model_data["kelly_fraction"] == best_kf)
        ]
        best_row = at_config.loc[at_config["pnl_combined"].idxmax()]
        best_strat = best_row["strategy"]
        best_pnl = best_row["pnl_combined"]
        uplift = best_pnl - uniform_pnl

        rows.append(
            {
                "model": model,
                "model_short": short_model(model),
                "best_config": f"edge={best_edge}, KF={best_kf}",
                "best_strategy": best_strat,
                "uniform_pnl": uniform_pnl,
                "best_pnl": best_pnl,
                "uplift_vs_uniform": uplift,
                "uplift_pct": uplift / abs(uniform_pnl) * 100 if abs(uniform_pnl) > 1 else 0.0,
            }
        )

    model_df = pd.DataFrame(rows)

    # Print table
    display = model_df.copy()
    display["uniform_pnl"] = display["uniform_pnl"].map(lambda x: f"${x:,.0f}")
    display["best_pnl"] = display["best_pnl"].map(lambda x: f"${x:,.0f}")
    display["uplift_vs_uniform"] = display["uplift_vs_uniform"].map(lambda x: f"${x:,.0f}")
    display["uplift_pct"] = display["uplift_pct"].map(lambda x: f"{x:+.1f}%")

    print(
        df_to_md(
            display[
                [
                    "model_short",
                    "best_config",
                    "best_strategy",
                    "uniform_pnl",
                    "best_pnl",
                    "uplift_vs_uniform",
                    "uplift_pct",
                ]
            ],
            float_fmt=".1f",
        )
    )
    print()

    # Interpretation
    unique_strats = model_df["best_strategy"].nunique()
    total_models = len(model_df)
    if unique_strats == 1:
        print(
            f"**All {total_models} models agree on the same best strategy: "
            f"{model_df['best_strategy'].iloc[0]}** — allocation is model-independent."
        )
    elif unique_strats <= 2:
        print(
            f"**{unique_strats} different strategies across {total_models} models** "
            f"— mostly consistent, minor model-dependence."
        )
        for strat in model_df["best_strategy"].unique():
            models = model_df[model_df["best_strategy"] == strat]["model_short"].tolist()
            print(f"  {strat}: {', '.join(models)}")
    else:
        print(
            f"**{unique_strats} different strategies across {total_models} models** "
            f"— allocation is model-dependent, proceed with caution."
        )
        for strat in model_df["best_strategy"].unique():
            models = model_df[model_df["best_strategy"] == strat]["model_short"].tolist()
            print(f"  {strat}: {', '.join(models)}")
    print()

    # Also check: at recommended config, do models agree?
    _subsection("At Recommended Config (edge=0.20, KF=0.05)")
    rec_rows: list[dict] = []
    for model in ALL_MODELS:
        rec_data = combined[
            (combined["model_type"] == model)
            & (combined["buy_edge_threshold"] == REC_EDGE)
            & (combined["kelly_fraction"] == REC_KF)
        ]
        if len(rec_data) == 0:
            continue
        uniform_pnl = rec_data[rec_data["strategy"] == "uniform"]["pnl_combined"]
        u_pnl = float(uniform_pnl.iloc[0]) if len(uniform_pnl) > 0 else 0.0
        best_row = rec_data.loc[rec_data["pnl_combined"].idxmax()]
        rec_rows.append(
            {
                "model_short": short_model(model),
                "best_strategy": best_row["strategy"],
                "pnl_combined": best_row["pnl_combined"],
                "uniform_pnl": u_pnl,
                "uplift": best_row["pnl_combined"] - u_pnl,
            }
        )

    rec_df = pd.DataFrame(rec_rows)
    display = rec_df.copy()
    display["pnl_combined"] = display["pnl_combined"].map(lambda x: f"${x:,.0f}")
    display["uniform_pnl"] = display["uniform_pnl"].map(lambda x: f"${x:,.0f}")
    display["uplift"] = display["uplift"].map(lambda x: f"${x:,.0f}")
    print(df_to_md(display, float_fmt=".1f"))
    print()

    return model_df


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Card
# ═══════════════════════════════════════════════════════════════════════════════


def generate_strategy_card(data_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a strategy card summarizing all robustness metrics for key strategies.

    Reads previously saved CSVs from the output directory and joins them into a
    single summary table for the 8 card strategies: uniform, equal_active,
    ev_100, maxedge_100, maxabsedge_100, edge_100, capital_100, npos_100, oracle.

    Returns the card DataFrame and saves to strategy_card.csv.
    """
    _section("Strategy Card")

    out_dir = ensure_output_dir()
    card_strategies = [
        "uniform",
        "equal_active",
        "ev_100",
        "maxedge_100",
        "maxabsedge_100",
        "edge_100",
        "capital_100",
        "npos_100",
        "oracle",
    ]

    # ── Read cross_year_stability.csv ──
    stability_path = out_dir / "cross_year_stability.csv"
    if stability_path.exists():
        stab = pd.read_csv(stability_path)
    else:
        print(f"  WARNING: {stability_path} not found, computing from data")
        stab = analyze_cross_year_stability(data_by_year)

    # ── Read bootstrap_ranking.csv ──
    boot_path = out_dir / "bootstrap_ranking.csv"
    boot: pd.DataFrame | None = None
    if boot_path.exists():
        boot_raw = pd.read_csv(boot_path)
        # Filter to category bootstrap type
        boot = boot_raw[boot_raw["bootstrap_type"] == "category"].copy()

    # ── Read pairwise_winrates.csv ──
    pw_path = out_dir / "pairwise_winrates.csv"
    mean_winrates: dict[str, float] = {}
    if pw_path.exists():
        pw_raw = pd.read_csv(pw_path)
        # Compute mean win rate per strategy_a (excluding self-comparisons)
        for strat in card_strategies:
            strat_rows = pw_raw[(pw_raw["strategy_a"] == strat) & (pw_raw["strategy_b"] != strat)]
            if len(strat_rows) > 0:
                mean_winrates[strat] = strat_rows["winrate_a_over_b"].mean()

    # ── Read entry_point_robustness.csv ──
    entry_path = out_dir / "entry_point_robustness.csv"
    pct_positive: dict[str, float] = {}
    if entry_path.exists():
        entry_raw = pd.read_csv(entry_path)
        # Compute % positive uplift per strategy
        for strat in card_strategies:
            if strat == "uniform":
                continue
            sdf = entry_raw[entry_raw["strategy"] == strat]
            if len(sdf) > 0:
                pct_positive[strat] = (sdf["uplift"] > 0).mean()

    # ── Read noise_sensitivity.csv ──
    # Schema: strategy, noise_sigma, total_pnl, avg_std
    noise_path = out_dir / "noise_sensitivity.csv"
    noise_pnl: dict[str, float] = {}
    if noise_path.exists():
        noise_raw = pd.read_csv(noise_path)
        if "noise_sigma" in noise_raw.columns and "total_pnl" in noise_raw.columns:
            noise_1 = noise_raw[noise_raw["noise_sigma"] == 1.0]
            for strat in card_strategies:
                row = noise_1[noise_1["strategy"] == strat]
                if len(row) > 0:
                    noise_pnl[strat] = float(row["total_pnl"].iloc[0])

    # ── Compute oracle P&L directly (excluded from most prospective-only analyses) ──
    oracle_pnl_by_year: dict[int, float] = {}
    all_strategies = get_all_strategies()
    if "oracle" in all_strategies:
        oracle_params = all_strategies["oracle"]
        for year, df in data_by_year.items():
            rec_df = df[
                (df["model_type"] == REC_MODEL)
                & (df["buy_edge_threshold"] == REC_EDGE)
                & (df["kelly_fraction"] == REC_KF)
            ]
            year_pnl = 0.0
            for _entry, grp in rec_df.groupby("entry_snapshot"):
                year_pnl += evaluate_portfolio_at_entry(grp, oracle_params)
            oracle_pnl_by_year[year] = year_pnl

    # ── Assemble card ──
    card_rows: list[dict] = []
    for strat in card_strategies:
        stab_row = stab[stab["strategy"] == strat]
        pnl_24 = float(stab_row["pnl_2024"].iloc[0]) if len(stab_row) > 0 else float("nan")
        pnl_25 = float(stab_row["pnl_2025"].iloc[0]) if len(stab_row) > 0 else float("nan")

        # Oracle fallback: compute from data if not in stability CSV
        if strat == "oracle" and np.isnan(pnl_24) and oracle_pnl_by_year:
            pnl_24 = oracle_pnl_by_year.get(2024, float("nan"))
            pnl_25 = oracle_pnl_by_year.get(2025, float("nan"))
        combined = pnl_24 + pnl_25 if not (np.isnan(pnl_24) or np.isnan(pnl_25)) else float("nan")
        r24 = int(stab_row["rank_2024"].iloc[0]) if len(stab_row) > 0 else None
        r25 = int(stab_row["rank_2025"].iloc[0]) if len(stab_row) > 0 else None
        avg_r = (
            float(stab_row["avg_rank"].iloc[0])
            if len(stab_row) > 0 and "avg_rank" in stab_row.columns
            else float("nan")
        )
        worst_r = (
            float(stab_row["worst_rank"].iloc[0])
            if len(stab_row) > 0 and "worst_rank" in stab_row.columns
            else float("nan")
        )

        boot_rank1 = float("nan")
        boot_top3 = float("nan")
        if boot is not None:
            brow = boot[boot["strategy"] == strat]
            if len(brow) > 0:
                boot_rank1 = float(brow["pct_rank1"].iloc[0])
                boot_top3 = float(brow["pct_top3"].iloc[0])

        card_rows.append(
            {
                "Strategy": strat,
                "P&L '24": pnl_24,
                "P&L '25": pnl_25,
                "Combined": combined,
                "Rank '24": r24,
                "Rank '25": r25,
                "Avg Rank": avg_r,
                "Worst Rank": worst_r,
                "Boot Rank-1%": boot_rank1,
                "Boot Top-3%": boot_top3,
                "Mean Win Rate": mean_winrates.get(strat, float("nan")),
                "% Entries Positive": pct_positive.get(strat, float("nan")),
                "Noise σ=1": noise_pnl.get(strat, float("nan")),
            }
        )

    card_df = pd.DataFrame(card_rows)

    # ── Format and print ──
    display = card_df.copy()
    display["P&L '24"] = display["P&L '24"].map(lambda x: f"${x:,.0f}" if not np.isnan(x) else "—")
    display["P&L '25"] = display["P&L '25"].map(lambda x: f"${x:,.0f}" if not np.isnan(x) else "—")
    display["Combined"] = display["Combined"].map(
        lambda x: f"${x:,.0f}" if not np.isnan(x) else "—"
    )
    display["Rank '24"] = display["Rank '24"].map(lambda x: str(int(x)) if pd.notna(x) else "—")
    display["Rank '25"] = display["Rank '25"].map(lambda x: str(int(x)) if pd.notna(x) else "—")
    display["Avg Rank"] = display["Avg Rank"].map(
        lambda x: f"**{x:.1f}**" if not np.isnan(x) else "—"
    )
    display["Worst Rank"] = display["Worst Rank"].map(
        lambda x: f"{x:.0f}" if not np.isnan(x) else "—"
    )
    display["Boot Rank-1%"] = display["Boot Rank-1%"].map(
        lambda x: f"{x:.1%}" if not np.isnan(x) else "—"
    )
    display["Boot Top-3%"] = display["Boot Top-3%"].map(
        lambda x: f"{x:.1%}" if not np.isnan(x) else "—"
    )
    display["Mean Win Rate"] = display["Mean Win Rate"].map(
        lambda x: f"{x:.1%}" if not np.isnan(x) else "—"
    )
    display["% Entries Positive"] = display["% Entries Positive"].map(
        lambda x: f"{x:.0%}" if not np.isnan(x) else "—"
    )
    display["Noise σ=1"] = display["Noise σ=1"].map(
        lambda x: f"${x:,.0f}" if not np.isnan(x) else "—"
    )

    print(df_to_md(display))
    print()

    # Save
    card_df.to_csv(out_dir / "strategy_card.csv", index=False)
    print(f"  Saved: {out_dir / 'strategy_card.csv'}")

    return card_df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all robustness analyses: 5a-5d, 6, and model interaction."""
    print("=" * 80)
    print("STRATEGY ROBUSTNESS — Cross-Year Stability & Bootstrap Validation")
    print("=" * 80)
    print()
    print(f"Primary model: {short_model(REC_MODEL)} at edge={REC_EDGE}, KF={REC_KF}")
    print(f"Bankroll: ${BANKROLL:,.0f}/category")
    print(f"Years: {YEARS}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP:,}")
    print()

    # ── Load and prepare data ──
    print("Loading data...")
    data_by_year = {year: prepare_data(year) for year in YEARS}
    for year, df in data_by_year.items():
        n_configs = df["config_label"].nunique()
        n_models = df["model_type"].nunique()
        n_entries = df["entry_snapshot"].nunique()
        n_cats = df["category"].nunique()
        print(
            f"  {year}: {n_configs} configs × {n_models} models × "
            f"{n_entries} entries × {n_cats} categories = {len(df)} rows"
        )
    print()

    out_dir = ensure_output_dir()

    # ── Analysis 5a: Cross-Year Stability ──
    stability_df = analyze_cross_year_stability(data_by_year)
    stability_df.to_csv(out_dir / "cross_year_stability.csv", index=False)
    print(f"  Saved: {out_dir / 'cross_year_stability.csv'}")

    # ── Analysis 5b: Bootstrap Ranking ──
    bootstrap_df = analyze_bootstrap_ranking(data_by_year)
    bootstrap_df.to_csv(out_dir / "bootstrap_ranking.csv", index=False)
    print(f"  Saved: {out_dir / 'bootstrap_ranking.csv'}")

    # ── Analysis 5c: Pairwise Win Rates ──
    pairwise_df = analyze_pairwise_winrates(data_by_year)
    pairwise_df.to_csv(out_dir / "pairwise_winrates.csv", index=False)
    print(f"  Saved: {out_dir / 'pairwise_winrates.csv'}")

    # ── Analysis 5d: Leave-One-Year-Out ──
    loo_df = analyze_leave_one_year_out(data_by_year)
    loo_df.to_csv(out_dir / "leave_one_year_out.csv", index=False)
    print(f"  Saved: {out_dir / 'leave_one_year_out.csv'}")

    # ── Analysis 6: Entry-Point Robustness ──
    entry_df = analyze_entry_robustness(data_by_year)
    entry_df.to_csv(out_dir / "entry_point_robustness.csv", index=False)
    print(f"  Saved: {out_dir / 'entry_point_robustness.csv'}")

    # ── Model Interaction ──
    model_df = analyze_model_interaction(data_by_year)
    model_df.to_csv(out_dir / "model_interaction.csv", index=False)
    print(f"  Saved: {out_dir / 'model_interaction.csv'}")

    # ── Strategy Card ──
    _card_df = generate_strategy_card(data_by_year)

    # ── Summary ──
    _section("Summary")
    print("All outputs:")
    for fname in [
        "cross_year_stability.csv",
        "bootstrap_ranking.csv",
        "pairwise_winrates.csv",
        "leave_one_year_out.csv",
        "entry_point_robustness.csv",
        "model_interaction.csv",
        "strategy_card.csv",
    ]:
        print(f"  - {out_dir / fname}")
    for fname in [
        "plots/robustness/bootstrap_rank1.png",
        "plots/robustness/pairwise_winrate.png",
        "plots/robustness/entry_uplift.png",
        "plots/robustness/worst_rank_scatter.png",
    ]:
        print(f"  - {out_dir / fname}")


if __name__ == "__main__":
    main()
