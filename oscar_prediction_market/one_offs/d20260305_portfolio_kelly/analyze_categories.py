"""Analysis 1: Category heterogeneity, idle bankroll, and oracle/equal-active decomposition.

Foundational analysis establishing whether there is value in reallocating bankroll
across categories. Answers three questions:

1. **Category heterogeneity**: How much does P&L vary across categories?
   If categories are homogeneous, reallocation can't help.

2. **Idle bankroll**: What fraction of total bankroll sits in categories with zero trades?
   This is the most obvious source of drag — money doing nothing.

3. **Oracle + equal-active decomposition**: How much of the oracle ceiling can be captured
   by the simple equal-active heuristic (redistribute idle bankroll equally among active
   categories)? If capture % is high, we may not need sophisticated signals.

4. **Signal correlations**: Are the candidate allocation signals (EV, capital, edge, etc.)
   redundant or do they capture different information?

Usage:
    uv run python -m oscar_prediction_market.one_offs.d20260305_portfolio_kelly.analyze_categories
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oscar_prediction_market.one_offs.d20260305_portfolio_kelly.shared import (
    ALL_MODELS,
    BANKROLL,
    GROUP_COLS,
    SIGNAL_NAMES,
    YEARS,
    df_to_md,
    ensure_output_dir,
    ensure_plot_dir,
    evaluate_all_strategies_by_year,
    prepare_data,
    short_model,
)

# ─── Constants ───────────────────────────────────────────────────────────────

# "Recommended" config from the config selection sweep
REC_EDGE = 0.20
REC_KF = 0.05


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _filter_config(
    df: pd.DataFrame,
    edge: float = REC_EDGE,
    kf: float = REC_KF,
) -> pd.DataFrame:
    """Filter to a single (buy_edge_threshold, kelly_fraction) config."""
    return df[(df["buy_edge_threshold"] == edge) & (df["kelly_fraction"] == kf)].copy()


def _fmt_pnl(x: float) -> str:
    """Format a P&L value with commas and no decimals."""
    return f"{x:,.0f}"


def _pre_format_df(
    df: pd.DataFrame,
    pnl_cols: list[str] | None = None,
    edge_col: str | None = "buy_edge_threshold",
) -> pd.DataFrame:
    """Pre-format numeric columns as strings so tabulate doesn't reparse them.

    This avoids the issue where tabulate auto-detects "0.02" as float and
    applies floatfmt=",.0f" → "0".
    """
    out = df.copy()
    if edge_col and edge_col in out.columns:
        out[edge_col] = out[edge_col].apply(lambda x: f"{float(x):.2f}")
    for col in pnl_cols or []:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{float(x):,.0f}")
    return out


def _section(title: str) -> None:
    """Print a markdown section header."""
    print(f"\n{'=' * 80}")
    print(f"## {title}")
    print(f"{'=' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Category heterogeneity
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_category_heterogeneity(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Quantify per-category P&L variation for avg_ensemble at recommended config.

    For each (year, category), sums across entry snapshots to get:
    - total_pnl, capital_deployed, n_positions, ROIC
    - ev_pnl_blend (expected value — what we'd predict ex-ante)

    Then computes coefficient of variation (CV) of P&L across categories
    to measure heterogeneity.
    """
    _section("Analysis 1: Category Heterogeneity")

    rows: list[dict] = []
    for year in YEARS:
        df = _filter_config(data_by_year[year])
        df = df[df["model_type"] == "avg_ensemble"]

        cat_agg = (
            df.groupby("category")
            .agg(
                total_pnl=("total_pnl", "sum"),
                ev_pnl_blend=("ev_pnl_blend", "sum"),
                capital_deployed=("capital_deployed", "sum"),
                n_positions=("n_positions", "sum"),
                n_entries=("entry_snapshot", "nunique"),
            )
            .reset_index()
        )
        cat_agg["year"] = year
        cat_agg["is_active"] = cat_agg["n_positions"] > 0
        # ROIC = total PnL / capital deployed (undefined if no capital)
        cat_agg["roic"] = np.where(
            cat_agg["capital_deployed"] > 0,
            cat_agg["total_pnl"] / cat_agg["capital_deployed"],
            0.0,
        )
        rows.append(cat_agg)

    result = pd.concat(rows, ignore_index=True)

    # Print per-year tables
    for year in YEARS:
        yr = result[result["year"] == year].sort_values("total_pnl", ascending=False)
        display = yr[
            [
                "category",
                "n_entries",
                "n_positions",
                "capital_deployed",
                "total_pnl",
                "ev_pnl_blend",
                "roic",
                "is_active",
            ]
        ].copy()
        display["roic"] = display["roic"].map("{:.1%}".format)
        print(f"### avg_ensemble @ edge={REC_EDGE}, KF={REC_KF} — {year}\n")
        print(df_to_md(display, float_fmt=",.1f"))
        print()

        # Coefficient of variation across categories
        pnl_vals = yr["total_pnl"].values
        if pnl_vals.std() > 0 and pnl_vals.mean() != 0:
            cv = pnl_vals.std() / abs(pnl_vals.mean())
            print(f"  P&L CV across categories: {cv:.2f}")
        active = yr["is_active"].sum()
        print(f"  Active categories: {active}/{len(yr)}")
        print()

    return result


def plot_category_heterogeneity(cat_df: pd.DataFrame) -> None:
    """Bar chart of per-category P&L for each year, showing heterogeneity visually."""
    plot_dir = ensure_plot_dir("categories")

    for year in YEARS:
        yr = cat_df[cat_df["year"] == year].sort_values("total_pnl", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        cats = yr["category"].values
        pnls = yr["total_pnl"].values
        colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]

        bars = ax.bar(range(len(cats)), pnls, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Total P&L ($)")
        ax.set_title(f"Category P&L — avg_ensemble @ edge={REC_EDGE}, KF={REC_KF} ({year})")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with dollar values
        for bar, pnl in zip(bars, pnls, strict=False):
            y_offset = 5 if pnl >= 0 else -15
            ax.annotate(
                f"${pnl:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, pnl),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        fig.tight_layout()
        fig.savefig(plot_dir / f"category_pnl_{year}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {plot_dir / f'category_pnl_{year}.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Idle bankroll quantification
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_idle_bankroll(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Quantify idle bankroll across all models and configs.

    For each (model, config, entry_snapshot, year), counts how many categories
    are active (have at least one trade). The idle fraction is:
        (total_categories - active_categories) / total_categories

    Aggregated by edge threshold to show: as edge threshold rises, fewer
    categories qualify → more bankroll sits idle.
    """
    _section("Analysis 2: Idle Bankroll Quantification")

    rows: list[dict] = []
    for year in YEARS:
        df = data_by_year[year]
        for (model, config, entry), group in df.groupby(GROUP_COLS):
            n_cats = len(group)
            n_active = group["is_active"].sum()
            row0 = group.iloc[0]
            rows.append(
                {
                    "year": year,
                    "model_type": model,
                    "config_label": config,
                    "entry_snapshot": entry,
                    "buy_edge_threshold": row0["buy_edge_threshold"],
                    "kelly_fraction": row0["kelly_fraction"],
                    "n_categories": n_cats,
                    "n_active": n_active,
                    "idle_fraction": (n_cats - n_active) / n_cats,
                    "bankroll_idle": BANKROLL * (n_cats - n_active),
                    "bankroll_active": BANKROLL * n_active,
                }
            )

    idle_df = pd.DataFrame(rows)

    # Aggregate by edge threshold: mean across models, configs, entries, years
    by_edge = (
        idle_df.groupby("buy_edge_threshold")
        .agg(
            mean_active=("n_active", "mean"),
            mean_total=("n_categories", "mean"),
            mean_idle_frac=("idle_fraction", "mean"),
            median_idle_frac=("idle_fraction", "median"),
            min_idle_frac=("idle_fraction", "min"),
            max_idle_frac=("idle_fraction", "max"),
            n_obs=("idle_fraction", "count"),
        )
        .reset_index()
    )
    by_edge["mean_active"] = by_edge["mean_active"].apply(lambda x: f"{x:.1f}")
    by_edge["mean_total"] = by_edge["mean_total"].apply(lambda x: f"{x:.1f}")
    by_edge["n_obs"] = by_edge["n_obs"].astype(int).astype(str)
    by_edge["mean_idle_frac"] = by_edge["mean_idle_frac"].map("{:.1%}".format)
    by_edge["median_idle_frac"] = by_edge["median_idle_frac"].map("{:.1%}".format)
    by_edge["min_idle_frac"] = by_edge["min_idle_frac"].map("{:.1%}".format)
    by_edge["max_idle_frac"] = by_edge["max_idle_frac"].map("{:.1%}".format)
    by_edge = _pre_format_df(by_edge)

    print("### Idle Bankroll by Edge Threshold (all models, configs, entries)\n")
    print(df_to_md(by_edge, float_fmt=""))
    print()

    # Also show per-model breakdown at recommended config
    rec = idle_df[
        (idle_df["buy_edge_threshold"] == REC_EDGE) & (idle_df["kelly_fraction"] == REC_KF)
    ]
    by_model = (
        rec.groupby(["model_type", "year"])
        .agg(
            mean_active=("n_active", "mean"),
            mean_total=("n_categories", "mean"),
            mean_idle_frac=("idle_fraction", "mean"),
        )
        .reset_index()
    )
    by_model["model_short"] = by_model["model_type"].map(short_model)
    by_model["mean_active"] = by_model["mean_active"].round(1)
    by_model["mean_idle_frac"] = by_model["mean_idle_frac"].map("{:.1%}".format)

    print(f"### Idle Bankroll by Model @ edge={REC_EDGE}, KF={REC_KF}\n")
    print(
        df_to_md(by_model[["model_short", "year", "mean_active", "mean_total", "mean_idle_frac"]])
    )
    print()

    return idle_df


def plot_idle_bankroll(idle_df: pd.DataFrame) -> None:
    """Line plot: idle fraction vs edge threshold, one line per KF."""
    plot_dir = ensure_plot_dir("categories")

    agg = (
        idle_df.groupby(["buy_edge_threshold", "kelly_fraction"])
        .agg(mean_idle=("idle_fraction", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for kf, grp in agg.groupby("kelly_fraction"):
        grp_sorted = grp.sort_values("buy_edge_threshold")
        ax.plot(
            grp_sorted["buy_edge_threshold"],
            grp_sorted["mean_idle"] * 100,
            marker="o",
            label=f"KF={kf}",
        )

    ax.set_xlabel("Buy Edge Threshold")
    ax.set_ylabel("Mean Idle Bankroll (%)")
    ax.set_title("Idle Bankroll vs Edge Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(plot_dir / "idle_bankroll_by_edge.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'idle_bankroll_by_edge.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Oracle + equal-active decomposition
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_three_strategies_per_entry(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Compute uniform, equal_active, and oracle PnL for every entry point.

    Returns one row per (year, model_type, config_label, entry_snapshot) with
    columns: pnl_uniform, pnl_equal_active, pnl_oracle.
    """
    strategies = {
        "uniform": {"strategy": "uniform", "aggressiveness": 1.0, "cap": None},
        "equal_active": {"strategy": "equal_active", "aggressiveness": 1.0, "cap": None},
        "oracle": {"strategy": "oracle", "aggressiveness": 1.0, "cap": None},
    }
    results = evaluate_all_strategies_by_year(data_by_year, strategies)

    # Pivot from long-form (strategy column) to wide-form (pnl_uniform, etc.)
    pivot = results.pivot_table(
        index=[
            "year",
            "model_type",
            "config_label",
            "entry_snapshot",
            "buy_edge_threshold",
            "kelly_fraction",
        ],
        columns="strategy",
        values="portfolio_pnl",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(
        columns={
            "uniform": "pnl_uniform",
            "equal_active": "pnl_equal_active",
            "oracle": "pnl_oracle",
        }
    )
    return pivot


def analyze_oracle_decomposition(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Oracle + equal-active decomposition: THE key analysis.

    For each (model, config, year), computes:
    - uniform PnL: baseline with $BANKROLL per category
    - equal_active PnL: redistribute idle bankroll equally to active categories
    - oracle PnL: hindsight-optimal proportional to max(actual_pnl, 0)
    - equal_active capture % = (equal_active - uniform) / (oracle - uniform) × 100

    Entry-level results are summed to (model, config, year) level.
    Then combined across 2024+2025.
    """
    _section("Analysis 3: Oracle + Equal-Active Decomposition")

    entry_level = _compute_three_strategies_per_entry(data_by_year)

    # Aggregate to (year, model, config) level by summing across entries
    year_level = (
        entry_level.groupby(
            ["year", "model_type", "config_label", "buy_edge_threshold", "kelly_fraction"]
        )
        .agg(
            pnl_uniform=("pnl_uniform", "sum"),
            pnl_equal_active=("pnl_equal_active", "sum"),
            pnl_oracle=("pnl_oracle", "sum"),
            n_entries=("entry_snapshot", "nunique"),
        )
        .reset_index()
    )

    # Compute oracle ceiling (uplift possible) and capture %
    year_level["oracle_ceiling"] = year_level["pnl_oracle"] - year_level["pnl_uniform"]
    year_level["ea_uplift"] = year_level["pnl_equal_active"] - year_level["pnl_uniform"]
    year_level["capture_pct"] = np.where(
        year_level["oracle_ceiling"] > 0,
        year_level["ea_uplift"] / year_level["oracle_ceiling"] * 100,
        np.nan,
    )

    # ── Table A: Per-model summary at recommended config ──
    rec = year_level[
        (year_level["buy_edge_threshold"] == REC_EDGE) & (year_level["kelly_fraction"] == REC_KF)
    ].copy()

    print(f"### Table A: Oracle Decomposition @ edge={REC_EDGE}, KF={REC_KF} — per year\n")
    for year in YEARS:
        yr = rec[rec["year"] == year].copy()
        yr["model_short"] = yr["model_type"].map(short_model)
        display = yr[
            [
                "model_short",
                "pnl_uniform",
                "pnl_equal_active",
                "pnl_oracle",
                "oracle_ceiling",
                "ea_uplift",
                "capture_pct",
            ]
        ].sort_values("pnl_uniform", ascending=False)
        display["capture_pct"] = display["capture_pct"].map(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        print(f"**{year}:**\n")
        print(df_to_md(display))
        print()

    # Combined 2024+2025
    combined = (
        rec.groupby(["model_type"])
        .agg(
            pnl_uniform=("pnl_uniform", "sum"),
            pnl_equal_active=("pnl_equal_active", "sum"),
            pnl_oracle=("pnl_oracle", "sum"),
        )
        .reset_index()
    )
    combined["oracle_ceiling"] = combined["pnl_oracle"] - combined["pnl_uniform"]
    combined["ea_uplift"] = combined["pnl_equal_active"] - combined["pnl_uniform"]
    combined["capture_pct"] = np.where(
        combined["oracle_ceiling"] > 0,
        combined["ea_uplift"] / combined["oracle_ceiling"] * 100,
        np.nan,
    )
    combined["model_short"] = combined["model_type"].map(short_model)

    print("**Combined 2024+2025:**\n")
    display = combined[
        [
            "model_short",
            "pnl_uniform",
            "pnl_equal_active",
            "pnl_oracle",
            "oracle_ceiling",
            "ea_uplift",
            "capture_pct",
        ]
    ].sort_values("pnl_uniform", ascending=False)
    display["capture_pct"] = display["capture_pct"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    print(df_to_md(display))
    print()

    # ── Table B: avg_ensemble across all edge thresholds ──
    print("### Table B: Oracle Decomposition — avg_ensemble across edge thresholds\n")
    avg_ens = year_level[year_level["model_type"] == "avg_ensemble"]
    by_edge_year = (
        avg_ens.groupby(["buy_edge_threshold", "kelly_fraction", "year"])
        .agg(
            pnl_uniform=("pnl_uniform", "sum"),
            pnl_equal_active=("pnl_equal_active", "sum"),
            pnl_oracle=("pnl_oracle", "sum"),
        )
        .reset_index()
    )
    # Sum across years for combined
    by_edge_combined = (
        by_edge_year.groupby(["buy_edge_threshold", "kelly_fraction"])
        .agg(
            pnl_uniform=("pnl_uniform", "sum"),
            pnl_equal_active=("pnl_equal_active", "sum"),
            pnl_oracle=("pnl_oracle", "sum"),
        )
        .reset_index()
    )
    by_edge_combined["oracle_ceiling"] = (
        by_edge_combined["pnl_oracle"] - by_edge_combined["pnl_uniform"]
    )
    by_edge_combined["ea_uplift"] = (
        by_edge_combined["pnl_equal_active"] - by_edge_combined["pnl_uniform"]
    )
    by_edge_combined["capture_pct"] = np.where(
        by_edge_combined["oracle_ceiling"] > 0,
        by_edge_combined["ea_uplift"] / by_edge_combined["oracle_ceiling"] * 100,
        np.nan,
    )

    # Show KF=0.05 slice for clarity
    kf_slice = by_edge_combined[by_edge_combined["kelly_fraction"] == REC_KF].sort_values(
        "buy_edge_threshold"
    )
    display_b = kf_slice[
        [
            "buy_edge_threshold",
            "pnl_uniform",
            "pnl_equal_active",
            "pnl_oracle",
            "oracle_ceiling",
            "capture_pct",
        ]
    ].copy()
    display_b["capture_pct"] = display_b["capture_pct"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    pnl_cols_b = ["pnl_uniform", "pnl_equal_active", "pnl_oracle", "oracle_ceiling"]
    display_b = _pre_format_df(display_b, pnl_cols=pnl_cols_b)
    print(f"KF={REC_KF}, combined 2024+2025:\n")
    print(df_to_md(display_b, float_fmt=""))
    print()

    # ── Table C: All models at recommended config ──
    print("### Table C: Which model benefits most from reallocation?\n")
    rank = combined.sort_values("ea_uplift", ascending=False)
    display_c = rank[
        ["model_short", "pnl_uniform", "ea_uplift", "oracle_ceiling", "capture_pct"]
    ].copy()
    display_c["capture_pct"] = display_c["capture_pct"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    print(f"Ranked by equal-active uplift (combined 2024+2025, edge={REC_EDGE}, KF={REC_KF}):\n")
    print(df_to_md(display_c))
    print()

    return year_level


def plot_oracle_decomposition(year_level: pd.DataFrame) -> None:
    """Stacked bar chart: uniform vs equal-active uplift vs remaining oracle ceiling."""
    plot_dir = ensure_plot_dir("categories")

    # Aggregate to combined for avg_ensemble at KF=0.05
    avg = year_level[
        (year_level["model_type"] == "avg_ensemble") & (year_level["kelly_fraction"] == REC_KF)
    ]
    combined = (
        avg.groupby("buy_edge_threshold")
        .agg(
            pnl_uniform=("pnl_uniform", "sum"),
            pnl_equal_active=("pnl_equal_active", "sum"),
            pnl_oracle=("pnl_oracle", "sum"),
        )
        .reset_index()
        .sort_values("buy_edge_threshold")
    )

    combined["ea_uplift"] = combined["pnl_equal_active"] - combined["pnl_uniform"]
    combined["remaining_ceiling"] = combined["pnl_oracle"] - combined["pnl_equal_active"]
    # Clip negative remaining ceiling to 0 for display
    combined["remaining_ceiling"] = combined["remaining_ceiling"].clip(lower=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    edges = combined["buy_edge_threshold"].values
    x = np.arange(len(edges))
    width = 0.6

    ax.bar(x, combined["pnl_uniform"], width, label="Uniform", color="#3498db")
    ax.bar(
        x,
        combined["ea_uplift"],
        width,
        bottom=combined["pnl_uniform"],
        label="Equal-Active Uplift",
        color="#2ecc71",
    )
    ax.bar(
        x,
        combined["remaining_ceiling"],
        width,
        bottom=combined["pnl_equal_active"],
        label="Remaining Oracle Ceiling",
        color="#e74c3c",
        alpha=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2f}" for e in edges], rotation=45)
    ax.set_xlabel("Buy Edge Threshold")
    ax.set_ylabel("Portfolio P&L ($)")
    ax.set_title(f"Oracle Decomposition — avg_ensemble, KF={REC_KF}, Combined 2024+2025")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "oracle_decomposition.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'oracle_decomposition.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 4: Signal pairwise correlations
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_signal_correlations(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Compute rank correlations between candidate allocation signals.

    At each (model, config, entry, year) decision point, we have a vector of
    signal values across categories. We compute Spearman rank correlation between
    all pairs of signals, then average across all decision points.

    High correlation → signals are redundant (won't get diversification from
    combining them). Low correlation → signals capture different information.
    """
    _section("Analysis 4: Signal Pairwise Correlations")

    # Collect per-entry-point correlation matrices
    corr_matrices: list[pd.DataFrame] = []

    for year in YEARS:
        df = data_by_year[year]
        for (_model, _config, _entry), group in df.groupby(GROUP_COLS):
            # Only compute correlations when there are enough active categories
            active = group[group["is_active"]]
            if len(active) < 3:
                continue

            # Extract signal columns, compute Spearman rank correlation.
            # Drop columns with zero variance (e.g. n_positions all identical)
            # since rank correlation is undefined for constant vectors.
            signals = active[SIGNAL_NAMES]
            non_const = [c for c in signals.columns if signals[c].nunique() > 1]
            if len(non_const) < 2:
                continue
            corr = signals[non_const].corr(method="spearman")
            # Re-index to full signal set (NaN for dropped signals)
            corr = corr.reindex(index=SIGNAL_NAMES, columns=SIGNAL_NAMES)
            corr_matrices.append(corr)

    if not corr_matrices:
        print("Not enough data points for correlation analysis.")
        return pd.DataFrame()

    # Average correlation across all decision points, ignoring NaN
    stacked = np.stack([c.values for c in corr_matrices])
    with np.errstate(all="ignore"):
        mean_vals = np.nanmean(stacked, axis=0)
    mean_corr = pd.DataFrame(mean_vals, index=SIGNAL_NAMES, columns=SIGNAL_NAMES)
    mean_corr = mean_corr.round(3)

    print(f"### Mean Spearman Rank Correlation (n={len(corr_matrices)} decision points)\n")
    print(df_to_md(mean_corr.reset_index().rename(columns={"index": "signal"}), float_fmt=".3f"))
    print()

    # Also show by edge threshold — correlations may differ when more/fewer
    # categories are active
    print("### Signal correlations by edge threshold (avg across models/entries)\n")
    edge_corrs: list[dict] = []
    for year in YEARS:
        df = data_by_year[year]
        for (_model, _config, _entry), group in df.groupby(GROUP_COLS):
            active = group[group["is_active"]]
            if len(active) < 3:
                continue
            edge = group.iloc[0]["buy_edge_threshold"]
            signals = active[SIGNAL_NAMES]
            corr = signals.corr(method="spearman")
            # Extract upper triangle mean (excluding diagonal)
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            mean_offdiag = corr.values[mask].mean()
            edge_corrs.append(
                {
                    "buy_edge_threshold": edge,
                    "year": year,
                    "mean_pairwise_corr": mean_offdiag,
                    "n_active": len(active),
                }
            )

    if edge_corrs:
        edge_corr_df = pd.DataFrame(edge_corrs)
        by_edge = (
            edge_corr_df.groupby("buy_edge_threshold")
            .agg(
                mean_corr=("mean_pairwise_corr", "mean"),
                mean_n_active=("n_active", "mean"),
                n_obs=("mean_pairwise_corr", "count"),
            )
            .reset_index()
        )
        by_edge["mean_corr"] = by_edge["mean_corr"].round(3)
        by_edge["mean_n_active"] = by_edge["mean_n_active"].round(1)
        print(df_to_md(by_edge, float_fmt=".3f"))
        print()

    return mean_corr


def plot_signal_correlation_heatmap(mean_corr: pd.DataFrame) -> None:
    """Heatmap of mean signal rank correlations."""
    if mean_corr.empty:
        return

    plot_dir = ensure_plot_dir("categories")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    # Labels
    labels = [s.replace("_", "\n") for s in mean_corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mean_corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color)

    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Mean Signal Rank Correlation Across Decision Points")

    fig.tight_layout()
    fig.savefig(plot_dir / "signal_correlations.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'signal_correlations.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all category heterogeneity analyses."""
    print("# Category Allocation Analysis 1: Heterogeneity & Oracle Decomposition\n")
    print(f"Bankroll per category: ${BANKROLL:,.0f}")
    print(f"Recommended config: edge={REC_EDGE}, KF={REC_KF}")
    print(f"Models: {', '.join(ALL_MODELS)}")
    print(f"Years: {YEARS}")
    print()

    # Load data
    print("Loading data...")
    data_by_year = {year: prepare_data(year) for year in YEARS}
    for year, df in data_by_year.items():
        n_cats = df["category"].nunique()
        n_entries = df["entry_snapshot"].nunique()
        n_models = df["model_type"].nunique()
        n_configs = df["config_label"].nunique()
        print(
            f"  {year}: {n_cats} categories, {n_entries} entries, "
            f"{n_models} models, {n_configs} configs, {len(df):,} rows"
        )
    print()

    out = ensure_output_dir()

    # ── Analysis 1: Category heterogeneity ──
    cat_df = analyze_category_heterogeneity(data_by_year)
    cat_df.to_csv(out / "category_heterogeneity.csv", index=False)
    print(f"Saved: {out / 'category_heterogeneity.csv'}")
    plot_category_heterogeneity(cat_df)

    # ── Analysis 2: Idle bankroll ──
    idle_df = analyze_idle_bankroll(data_by_year)
    idle_df.to_csv(out / "idle_bankroll.csv", index=False)
    print(f"Saved: {out / 'idle_bankroll.csv'}")
    plot_idle_bankroll(idle_df)

    # ── Analysis 3: Oracle decomposition ──
    oracle_df = analyze_oracle_decomposition(data_by_year)
    oracle_df.to_csv(out / "oracle_decomposition.csv", index=False)
    print(f"Saved: {out / 'oracle_decomposition.csv'}")
    plot_oracle_decomposition(oracle_df)

    # ── Analysis 4: Signal correlations ──
    corr_df = analyze_signal_correlations(data_by_year)
    if not corr_df.empty:
        corr_df.to_csv(out / "signal_correlations.csv")
        print(f"Saved: {out / 'signal_correlations.csv'}")
        plot_signal_correlation_heatmap(corr_df)

    print("\n" + "=" * 80)
    print("Done. All outputs saved to:", out)
    print("=" * 80)


if __name__ == "__main__":
    main()
