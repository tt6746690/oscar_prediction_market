"""Analysis 2 + 3: Strategy scorecard, effective sample size, and noise sensitivity.

Core signal comparison that determines which category allocation approach is best.

**Analysis 2 — Strategy Scorecard**:
Evaluates all ~33 strategies across all models and configs. For each strategy computes:
- Mean combined uplift vs. uniform
- Best combined P&L across configs
- Both-year positive % (fraction of configs beating uniform in BOTH years)
- Year balance (pnl_2024 / pnl_2025 ratio)
- Cross-year rank ρ (Spearman correlation of PnL ranking across years)

Then slices the data several ways:
1. Top 10 at recommended config (edge=0.20, KF=0.05) for avg_ensemble
2. Sensitivity to edge threshold — does the best strategy change?
3. Signal group comparison — best variant per base signal

**Effective Sample Size**:
Eigenvalue-based N_eff for distinguishing allocation strategies.
Builds a (scenario × strategy) matrix, computes eigenvalues of the correlation matrix,
and reports N_eff = (Σλ)² / Σλ² — the number of independent observations we have
for comparing strategies.

**Analysis 3 — Noise Sensitivity**:
For the top 5-6 prospective strategies, adds multiplicative lognormal noise to the
signal at increasing σ levels. Measures at what noise level signal-based strategies
degrade to equal_active — answering whether the signal's Layer 2 value is real or
just an artifact of fine-tuning on two years of data.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260305_portfolio_kelly.compare_signals
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from oscar_prediction_market.one_offs.d20260305_portfolio_kelly.shared import (
    ALL_MODELS,
    BANKROLL,
    GROUP_COLS,
    OUTPUT_DIR,
    SIGNAL_NAMES,
    YEARS,
    add_noise_to_signal,
    aggregate_combined,
    aggregate_to_year_level,
    compute_weights,
    df_to_md,
    ensure_output_dir,
    ensure_plot_dir,
    evaluate_all_strategies_by_year,
    get_all_strategies,
    get_prospective_strategies,
    prepare_data,
    short_model,
)

# ─── Constants ───────────────────────────────────────────────────────────────

REC_EDGE = 0.20
REC_KF = 0.05

# Edge thresholds to test sensitivity
EDGE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]

# Signal group prefixes → base signal column mapping
SIGNAL_GROUPS: dict[str, str] = {
    "ev": "ev_pnl_blend",
    "edge": "mean_edge",
    "capital": "capital_deployed",
    "npos": "n_positions",
    "maxedge": "max_edge",
    "maxabsedge": "max_abs_edge",
}

# Noise levels for Analysis 3
NOISE_SIGMAS = [0.0, 0.2, 0.5, 1.0, 2.0]
N_NOISE_TRIALS = 50


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    """Print a markdown section header."""
    print(f"\n{'=' * 80}")
    print(f"## {title}")
    print(f"{'=' * 80}\n")


def _subsection(title: str) -> None:
    """Print a markdown subsection header."""
    print(f"\n### {title}\n")


def _filter_config(
    df: pd.DataFrame,
    edge: float = REC_EDGE,
    kf: float = REC_KF,
) -> pd.DataFrame:
    """Filter to a single (buy_edge_threshold, kelly_fraction) config."""
    return df[(df["buy_edge_threshold"] == edge) & (df["kelly_fraction"] == kf)].copy()


def _filter_model(df: pd.DataFrame, model: str = "avg_ensemble") -> pd.DataFrame:
    """Filter to a single model_type."""
    return df[df["model_type"] == model].copy()


def _pre_format_df(
    df: pd.DataFrame,
    pnl_cols: list[str] | None = None,
    pct_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pre-format numeric columns as strings for clean markdown output."""
    out = df.copy()
    for col in pnl_cols or []:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"${float(x):,.0f}")
    for col in pct_cols or []:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{float(x):.1%}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Strategy Scorecard
# ═══════════════════════════════════════════════════════════════════════════════


def build_scorecard(combined: pd.DataFrame) -> pd.DataFrame:
    """Build the strategy scorecard from combined (2024+2025) results.

    For each strategy, computes metrics across all (model, config) combinations:
    - mean_uplift: mean(strategy_pnl - uniform_pnl) across configs
    - best_combined: best pnl_combined across all configs
    - both_positive_pct: fraction of configs where pnl > uniform in BOTH years
    - year_balance: median(pnl_2024 / pnl_2025) across configs
    - cross_year_rho: Spearman correlation of strategy's config ranking 2024 vs 2025

    Args:
        combined: output of aggregate_combined() — one row per (strategy, model, config).

    Returns:
        Scorecard DataFrame sorted by mean_uplift descending.
    """
    # Get uniform baseline per (model, config)
    uniform = combined[combined["strategy"] == "uniform"][
        ["model_type", "config_label", "pnl_2024", "pnl_2025", "pnl_combined"]
    ].rename(
        columns={
            "pnl_2024": "uniform_2024",
            "pnl_2025": "uniform_2025",
            "pnl_combined": "uniform_combined",
        }
    )

    merged = combined.merge(uniform, on=["model_type", "config_label"], how="left")
    merged["uplift_combined"] = merged["pnl_combined"] - merged["uniform_combined"]
    merged["uplift_2024"] = merged["pnl_2024"] - merged["uniform_2024"]
    merged["uplift_2025"] = merged["pnl_2025"] - merged["uniform_2025"]
    # "Both-year positive" = beats uniform in both years simultaneously
    merged["both_positive"] = (merged["uplift_2024"] > 0) & (merged["uplift_2025"] > 0)

    rows: list[dict] = []
    for strat_name, grp in merged.groupby("strategy"):
        mean_uplift = grp["uplift_combined"].mean()
        best_combined = grp["pnl_combined"].max()

        both_pct = grp["both_positive"].mean()

        # Year balance: median of pnl_2024/pnl_2025 (excluding configs where 2025 ~ 0)
        valid = grp[grp["pnl_2025"].abs() > 10]
        year_bal = valid["year_balance"].median() if len(valid) > 0 else np.nan

        # Cross-year rank ρ: rank configs by PnL in each year, compute Spearman
        if len(grp) >= 3:
            result = spearmanr(grp["pnl_2024"].to_numpy(), grp["pnl_2025"].to_numpy())
            rho = float(result.statistic)  # type: ignore[union-attr]  # SignificanceResult
        else:
            rho = float("nan")

        rows.append(
            {
                "strategy": strat_name,
                "mean_uplift": mean_uplift,
                "best_combined": best_combined,
                "both_positive_pct": both_pct,
                "year_balance": year_bal,
                "cross_year_rho": rho,
                "n_configs": len(grp),
            }
        )

    scorecard = pd.DataFrame(rows).sort_values("mean_uplift", ascending=False)
    return scorecard.reset_index(drop=True)


def build_model_scorecard(combined: pd.DataFrame, model: str) -> pd.DataFrame:
    """Build scorecard restricted to a single model.

    Same metrics as build_scorecard but filtered to one model_type,
    so "across configs" means across edge × KF configs for that model.
    """
    model_df = combined[combined["model_type"] == model].copy()
    return build_scorecard(model_df)


def print_scorecard(scorecard: pd.DataFrame, title: str) -> None:
    """Pretty-print a scorecard table."""
    display = scorecard.copy()
    display["mean_uplift"] = display["mean_uplift"].map(lambda x: f"${x:,.0f}")
    display["best_combined"] = display["best_combined"].map(lambda x: f"${x:,.0f}")
    display["both_positive_pct"] = display["both_positive_pct"].map(lambda x: f"{x:.0%}")
    display["year_balance"] = display["year_balance"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    display["cross_year_rho"] = display["cross_year_rho"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    print(f"### {title}\n")
    print(df_to_md(display, float_fmt=".2f"))
    print()


def top_strategies_at_config(
    combined: pd.DataFrame,
    model: str,
    edge: float,
    kf: float,
    n: int = 10,
) -> pd.DataFrame:
    """Return top N strategies by combined PnL for a specific model and config.

    Filters combined to (model, edge, kf) then ranks strategies.
    """
    filt = combined[
        (combined["model_type"] == model)
        & (combined["buy_edge_threshold"] == edge)
        & (combined["kelly_fraction"] == kf)
    ].copy()
    filt = filt.sort_values("pnl_combined", ascending=False).head(n)
    return filt[["strategy", "pnl_2024", "pnl_2025", "pnl_combined"]].reset_index(drop=True)


def edge_sensitivity_table(
    combined: pd.DataFrame,
    model: str,
    kf: float = REC_KF,
) -> pd.DataFrame:
    """Show best strategy at each edge threshold for a given model.

    Answers: does the best allocation strategy change when you change the
    buy_edge_threshold?
    """
    rows: list[dict] = []
    for edge in EDGE_THRESHOLDS:
        filt = combined[
            (combined["model_type"] == model)
            & (combined["buy_edge_threshold"] == edge)
            & (combined["kelly_fraction"] == kf)
        ]
        if len(filt) == 0:
            continue
        # Exclude oracle for this comparison (since it's hindsight)
        prospective = filt[filt["strategy"] != "oracle"]
        best = prospective.loc[prospective["pnl_combined"].idxmax()]
        uniform = filt[filt["strategy"] == "uniform"]
        uniform_pnl = uniform["pnl_combined"].iloc[0] if len(uniform) > 0 else 0.0
        rows.append(
            {
                "edge_threshold": edge,
                "best_strategy": best["strategy"],
                "best_pnl": best["pnl_combined"],
                "uniform_pnl": uniform_pnl,
                "uplift": best["pnl_combined"] - uniform_pnl,
            }
        )
    return pd.DataFrame(rows)


def signal_group_comparison(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Group strategies by base signal and show best variant per group.

    Groups: ev, edge, capital, npos, maxedge, equal_active, uniform, oracle.
    """
    rows: list[dict] = []

    # Special strategies (not signal-based)
    for special in ["uniform", "equal_active", "oracle"]:
        match = scorecard[scorecard["strategy"] == special]
        if len(match) > 0:
            row = match.iloc[0]
            rows.append(
                {
                    "signal_group": special,
                    "best_variant": special,
                    "mean_uplift": row["mean_uplift"],
                    "best_combined": row["best_combined"],
                    "both_positive_pct": row["both_positive_pct"],
                }
            )

    # Signal-based groups
    for group_name in SIGNAL_GROUPS:
        group_strats = scorecard[scorecard["strategy"].str.startswith(group_name + "_")]
        if len(group_strats) == 0:
            continue
        best = group_strats.loc[group_strats["mean_uplift"].idxmax()]
        rows.append(
            {
                "signal_group": group_name,
                "best_variant": best["strategy"],
                "mean_uplift": best["mean_uplift"],
                "best_combined": best["best_combined"],
                "both_positive_pct": best["both_positive_pct"],
            }
        )

    result = pd.DataFrame(rows).sort_values("mean_uplift", ascending=False)
    return result.reset_index(drop=True)


def plot_scorecard_heatmap(combined: pd.DataFrame, model: str = "avg_ensemble") -> None:
    """Heatmap of strategy PnL across edge thresholds for one model at KF=0.05.

    Rows = strategies (sorted by combined PnL at recommended config).
    Columns = edge thresholds.
    Color = pnl_combined.
    """
    plot_dir = ensure_plot_dir("signals")
    kf = REC_KF

    # Build pivot table: strategy × edge → pnl_combined
    filt = combined[(combined["model_type"] == model) & (combined["kelly_fraction"] == kf)].copy()

    # Only keep strategies that exist at all edge thresholds
    strat_counts = filt.groupby("strategy")["buy_edge_threshold"].nunique()
    valid_strats = strat_counts[strat_counts >= len(EDGE_THRESHOLDS)].index

    filt = filt[filt["strategy"].isin(valid_strats)]
    if len(filt) == 0:
        print("  (no data for heatmap — skipping)")
        return

    pivot = filt.pivot_table(
        index="strategy",
        columns="buy_edge_threshold",
        values="pnl_combined",
        aggfunc="sum",
    )

    # Sort by PnL at recommended edge threshold
    if REC_EDGE in pivot.columns:
        pivot = pivot.sort_values(by=REC_EDGE, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Buy Edge Threshold")
    ax.set_ylabel("Strategy")
    ax.set_title(f"Strategy × Edge Threshold — {short_model(model)}, KF={kf}")

    # Annotate cells with dollar values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if abs(val - pivot.values.mean()) > pivot.values.std() else "black"
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Combined P&L ($)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "scorecard_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'scorecard_heatmap.png'}")


def run_analysis_2(
    data_by_year: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Analysis 2: Strategy Scorecard.

    Returns:
        (full_scorecard, combined_results) for downstream use.
    """
    _section("Analysis 2: Strategy Scorecard")

    # Evaluate all strategies across all years
    print("Evaluating all strategies across all years and configs...")
    strategies = get_all_strategies()
    results = evaluate_all_strategies_by_year(data_by_year, strategies)
    year_level = aggregate_to_year_level(results)
    combined = aggregate_combined(year_level)
    print(f"  {len(combined)} rows: {len(strategies)} strategies × models × configs")

    # ── Full scorecard (all models pooled) ──
    _subsection("Full Scorecard (all models)")
    full_scorecard = build_scorecard(combined)
    print_scorecard(full_scorecard, "All Models — Strategy Scorecard")

    # ── avg_ensemble scorecard ──
    _subsection("avg_ensemble Scorecard")
    ens_scorecard = build_model_scorecard(combined, "avg_ensemble")
    print_scorecard(ens_scorecard, "avg_ensemble — Strategy Scorecard")

    # ── Top 10 at recommended config ──
    _subsection("Top 10 at Recommended Config (avg_ensemble, edge=0.20, KF=0.05)")
    top10 = top_strategies_at_config(combined, "avg_ensemble", REC_EDGE, REC_KF, n=10)
    display = _pre_format_df(top10, pnl_cols=["pnl_2024", "pnl_2025", "pnl_combined"])
    print(df_to_md(display, float_fmt=".2f"))
    print()

    # ── Edge threshold sensitivity ──
    _subsection("Edge Threshold Sensitivity (avg_ensemble)")
    edge_table = edge_sensitivity_table(combined, "avg_ensemble")
    display = _pre_format_df(edge_table, pnl_cols=["best_pnl", "uniform_pnl", "uplift"])
    print(df_to_md(display, float_fmt=".2f"))
    print()

    # ── Signal group comparison ──
    _subsection("Signal Group Comparison (all models)")
    group_table = signal_group_comparison(full_scorecard)
    display = group_table.copy()
    display["mean_uplift"] = display["mean_uplift"].map(lambda x: f"${x:,.0f}")
    display["best_combined"] = display["best_combined"].map(lambda x: f"${x:,.0f}")
    display["both_positive_pct"] = display["both_positive_pct"].map(lambda x: f"{x:.0%}")
    print(df_to_md(display, float_fmt=".2f"))
    print()

    # ── Heatmap ──
    plot_scorecard_heatmap(combined, "avg_ensemble")

    # ── Save CSV ──
    out_dir = ensure_output_dir()
    full_scorecard.to_csv(out_dir / "signal_scorecard.csv", index=False)
    print(f"  Saved: {out_dir / 'signal_scorecard.csv'}")

    return full_scorecard, combined


# ═══════════════════════════════════════════════════════════════════════════════
# Effective Sample Size
# ═══════════════════════════════════════════════════════════════════════════════


def compute_effective_n(data_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute eigenvalue-based effective N for strategy comparisons.

    Builds a matrix where:
    - Rows = scenarios (year × model × config × entry × category)
    - Columns = allocation strategies
    - Cell = category_pnl × weight assigned by that strategy

    Then computes the correlation matrix of columns, extracts eigenvalues,
    and reports N_eff = (Σλ)² / Σλ².

    Why this matters: if strategies assign very similar weights (high correlation),
    the effective number of independent comparisons is small, meaning scorecard
    differences may be noise. N_eff tells us how many truly independent
    "data points" (scenarios) we have for distinguishing strategies.

    Returns:
        DataFrame with eigenvalue index, eigenvalue, cumulative variance explained,
        and N_eff summary.
    """
    _section("Effective Sample Size")

    strategies = get_all_strategies()

    # Build the scenario × strategy matrix
    # Each row = one (year, model, config, entry, category) — a single allocation decision
    # Each column = the weighted P&L contribution for one strategy
    all_rows: list[dict] = []

    for year, entry_pnl in data_by_year.items():
        for (model, config, entry), group in entry_pnl.groupby(GROUP_COLS):
            # Compute all strategy weights ONCE per group (not per category)
            all_weights: dict[str, pd.Series] = {}
            for strat_name, params in strategies.items():
                all_weights[strat_name] = compute_weights(group, **params)

            for iloc_idx, (_, cat_row) in enumerate(group.iterrows()):
                scenario_row: dict[str, object] = {
                    "year": year,
                    "model": model,
                    "config": config,
                    "entry": entry,
                    "category": cat_row["category"],
                    "raw_pnl": cat_row["total_pnl"],
                }

                pnl = float(cat_row["total_pnl"])
                for strat_name, weights in all_weights.items():
                    cat_weight = float(weights.iloc[iloc_idx])
                    scenario_row[f"w_{strat_name}"] = pnl * cat_weight

                all_rows.append(scenario_row)

    matrix_df = pd.DataFrame(all_rows)
    strat_cols = [c for c in matrix_df.columns if c.startswith("w_")]
    W = matrix_df[strat_cols].values  # (n_scenarios × n_strategies)

    print(f"Scenario matrix: {W.shape[0]} scenarios × {W.shape[1]} strategies")

    # Compute correlation matrix and eigenvalues
    # Handle constant columns (e.g., uniform has identical weights across categories)
    corr = np.corrcoef(W.T)
    # Replace NaN (from constant columns) with 0
    corr = np.nan_to_num(corr, nan=0.0)

    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)  # clip numerical negatives

    # N_eff = (Σλ)² / Σλ²
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues**2).sum()
    n_eff = sum_lambda**2 / sum_lambda_sq if sum_lambda_sq > 0 else 0

    print(f"Eigenvalue sum: {sum_lambda:.1f}")
    print(f"N_eff = (Σλ)² / Σλ² = {n_eff:.1f}")
    print()

    # ── Scenario-level N_eff (config-sweep style) ──
    # Aggregate from category-level to scenario-level: for each (year, model, config),
    # average the weighted PnL across entries and categories. This gives a
    # (n_scenarios × n_strategies) matrix where each scenario is a model×config combo.
    scenario_agg = matrix_df.groupby(["year", "model", "config"])[strat_cols].mean()
    W_scenarios = scenario_agg.values  # (n_scenarios × n_strategies)
    n_scenarios = W_scenarios.shape[0]

    corr_scenarios = np.corrcoef(W_scenarios)  # (n_scenarios × n_scenarios)
    corr_scenarios = np.nan_to_num(corr_scenarios, nan=0.0)
    eig_scenarios = np.sort(np.linalg.eigvalsh(corr_scenarios))[::-1]
    eig_scenarios = np.maximum(eig_scenarios, 0)
    sum_lam_s = eig_scenarios.sum()
    sum_lam_s2 = (eig_scenarios**2).sum()
    n_eff_scenarios = sum_lam_s**2 / sum_lam_s2 if sum_lam_s2 > 0 else 0

    print(f"Scenario-level N_eff = {n_eff_scenarios:.1f} out of {n_scenarios} scenarios")
    print(
        "  This measures how many independent evaluation contexts (model × config combos)"
        " we have. N_eff ≈ 5 means five independent 'experiments' despite"
        f" {n_scenarios} combos."
    )
    print()

    # ── Agg=100-only N_eff (5 signal families) ──
    # Filter to only the 5 uncapped agg=100 columns
    agg100_cols = [
        "w_ev_100",
        "w_maxedge_100",
        "w_maxabsedge_100",
        "w_edge_100",
        "w_capital_100",
        "w_npos_100",
    ]
    agg100_cols = [c for c in agg100_cols if c in strat_cols]  # safety check
    W_agg100 = matrix_df[agg100_cols].values

    corr_agg100 = np.corrcoef(W_agg100.T)  # (5 × 5)
    corr_agg100 = np.nan_to_num(corr_agg100, nan=0.0)
    eig_agg100 = np.sort(np.linalg.eigvalsh(corr_agg100))[::-1]
    eig_agg100 = np.maximum(eig_agg100, 0)
    sum_lam_a = eig_agg100.sum()
    sum_lam_a2 = (eig_agg100**2).sum()
    n_eff_agg100 = sum_lam_a**2 / sum_lam_a2 if sum_lam_a2 > 0 else 0

    print(f"Agg=100-only N_eff = {n_eff_agg100:.1f} out of {len(agg100_cols)} strategies")
    print(
        "  Even restricting to the 5 most-different signal families (all at 100%"
        " aggressiveness), N_eff is still very low — suggesting the signals produce"
        " nearly identical allocations."
    )
    print()

    # ── PC1 loading analysis ──
    # Use eigh on the full correlation matrix to get eigenvectors
    eig_vals_full, eig_vecs_full = np.linalg.eigh(corr)
    # eigh returns ascending order; PC1 is the last column
    pc1_loadings = eig_vecs_full[:, -1]

    loadings_min = pc1_loadings.min()
    loadings_max = pc1_loadings.max()
    loadings_spread = loadings_max - loadings_min
    print(
        f"PC1 loading range: {loadings_min:.3f} to {loadings_max:.3f}"
        f" (spread: {loadings_spread:.3f})"
    )

    # Compute Spearman(PC1, aggressiveness) for signal strategies only
    non_baseline = {"uniform", "equal_active", "oracle"}
    strat_names = [c.removeprefix("w_") for c in strat_cols]
    signal_indices = [i for i, name in enumerate(strat_names) if name not in non_baseline]
    signal_loadings = pc1_loadings[signal_indices]
    signal_agg_values = np.array(
        [strategies[strat_names[i]]["aggressiveness"] for i in signal_indices]
    )
    rho_pc1, _p_pc1 = stats.spearmanr(signal_loadings, signal_agg_values)
    print(f"Spearman(PC1, aggressiveness) = {rho_pc1:.2f} among signal strategies")
    print(
        "  Narrow spread means all strategies load almost identically on PC1 — the"
        " dominant pattern is 'how well did the market do overall', not 'which"
        " allocation was better'."
    )
    print()

    # Variance explained
    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var if total_var > 0 else np.zeros_like(eigenvalues)

    eigen_df = pd.DataFrame(
        {
            "component": range(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "var_explained": eigenvalues / total_var if total_var > 0 else 0,
            "cum_var_explained": cum_var,
        }
    )

    # Print top eigenvalues
    _subsection("Top Eigenvalues (Scree)")
    n_show = min(15, len(eigen_df))
    display = eigen_df.head(n_show).copy()
    display["eigenvalue"] = display["eigenvalue"].map(lambda x: f"{x:.3f}")
    display["var_explained"] = display["var_explained"].map(lambda x: f"{x:.1%}")
    display["cum_var_explained"] = display["cum_var_explained"].map(lambda x: f"{x:.1%}")
    print(df_to_md(display, float_fmt=".3f"))
    print()

    print(f"**Effective N = {n_eff:.1f}** out of {len(eigenvalues)} strategies")
    print()

    # Interpretation
    if n_eff < 5:
        print("  ⚠ Very few independent dimensions — strategies are highly correlated.")
        print("  Scorecard differences between similarly-weighted strategies may be noise.")
    elif n_eff < 10:
        print("  Moderate independence — some groups of strategies are effectively equivalent.")
    else:
        print("  Good independence — strategies span a meaningful range of allocations.")

    # ── Scree plot ──
    plot_dir = ensure_plot_dir("signals")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: eigenvalue scree plot
    ax1.bar(range(1, len(eigenvalues) + 1), eigenvalues, color="#3498db", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Eigenvalue Scree Plot")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="λ = 1 (noise level)")
    ax1.legend()

    # Right: cumulative variance explained
    ax2.plot(
        range(1, len(cum_var) + 1),
        cum_var,
        "o-",
        color="#2ecc71",
        markersize=4,
    )
    ax2.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title(f"Cumulative Variance — N_eff = {n_eff:.1f}")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(plot_dir / "effective_n_scree.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'effective_n_scree.png'}")

    # ── Save CSV ──
    out_dir = ensure_output_dir()
    eigen_df.to_csv(out_dir / "effective_n.csv", index=False)
    print(f"  Saved: {out_dir / 'effective_n.csv'}")

    return eigen_df


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Noise Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════


def _get_signal_for_strategy(strategy_name: str) -> str | None:
    """Map strategy name → signal column that should be noised.

    Returns None for strategies without a noiseable signal (uniform, equal_active, oracle).
    """
    all_strats = get_all_strategies()
    params = all_strats.get(strategy_name)
    if params is None:
        return None
    signal = params["strategy"]
    if signal in SIGNAL_NAMES:
        return signal
    return None


def _filter_to_rec_config(
    data_by_year: dict[int, pd.DataFrame],
    model: str = "avg_ensemble",
) -> dict[int, pd.DataFrame]:
    """Filter data to recommended model and config for fast noise evaluation."""
    filtered = {}
    for year, df in data_by_year.items():
        mask = (
            (df["model_type"] == model)
            & (df["buy_edge_threshold"] == REC_EDGE)
            & (df["kelly_fraction"] == REC_KF)
        )
        filtered[year] = df[mask].copy()
    return filtered


def evaluate_noisy_strategy(
    data_by_year: dict[int, pd.DataFrame],
    strategy_name: str,
    strategy_params: dict,
    noise_sigma: float,
    n_trials: int = N_NOISE_TRIALS,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate a strategy with noise injected into its signal.

    For noise_sigma=0, evaluates once (deterministic).
    For noise_sigma>0, averages over n_trials noise realizations.

    Uses only recommended config (avg_ensemble, edge=0.20, KF=0.05) for speed.
    Returns year-level aggregated results (one row per year per trial).
    """
    signal_col = _get_signal_for_strategy(strategy_name)

    # Filter to recommended config for speed — noise evaluation only needs one
    # (model, edge, KF) combo, not all 162.
    filtered = _filter_to_rec_config(data_by_year)

    if noise_sigma == 0 or signal_col is None:
        # Deterministic: evaluate once
        results = evaluate_all_strategies_by_year(filtered, {strategy_name: strategy_params})
        year_level = aggregate_to_year_level(results)
        year_level["trial"] = 0
        return year_level

    # Stochastic: average over noise realizations
    all_trials: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)

    for trial in range(n_trials):
        # Create noisy copy of data for each year
        noisy_data = {}
        for year, df in filtered.items():
            noisy_data[year] = add_noise_to_signal(df, signal_col, noise_sigma, rng)

        results = evaluate_all_strategies_by_year(noisy_data, {strategy_name: strategy_params})
        year_level = aggregate_to_year_level(results)
        year_level["trial"] = trial
        all_trials.append(year_level)

    return pd.concat(all_trials, ignore_index=True)


def run_noise_sensitivity(
    data_by_year: dict[int, pd.DataFrame],
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """Execute Analysis 3: Noise Sensitivity for top prospective strategies.

    Picks the top 6 prospective strategies by combined PnL (across all models),
    plus uniform and equal_active as baselines. For each strategy, evaluates
    at increasing noise levels and reports mean PnL degradation.

    The key question: at what noise level does a signal-based strategy degrade
    to equal_active performance? If that happens at σ=0.2 (mild noise), the
    signal's allocation value is fragile. If it survives σ=1.0+, the signal
    is genuinely informative.
    """
    _section("Analysis 3: Noise Sensitivity")

    all_strats = get_all_strategies()
    prospective = get_prospective_strategies()

    # Pick top 6 prospective strategies by mean combined PnL across all models
    strat_pnl = combined.groupby("strategy")["pnl_combined"].mean().sort_values(ascending=False)
    # Filter to prospective only (exclude oracle) and signal-based only (exclude uniform/equal_active)
    signal_based = [
        s for s in strat_pnl.index if s in prospective and s not in ("uniform", "equal_active")
    ]
    # Always include the 5 agg=100 uncapped strategies (strategy card candidates)
    card_signals = [
        "ev_100",
        "maxedge_100",
        "maxabsedge_100",
        "edge_100",
        "capital_100",
        "npos_100",
    ]
    # Add any remaining top performers not already included
    remaining = [s for s in signal_based if s not in card_signals]
    top_signals = card_signals + remaining[:3]  # 5 card + up to 3 more = 8 max

    # Always include baselines
    test_strategies = ["uniform", "equal_active"] + top_signals
    print(f"Testing {len(test_strategies)} strategies: {test_strategies}")
    print(f"Noise levels: σ = {NOISE_SIGMAS}")
    print(f"Trials per noise level: {N_NOISE_TRIALS}")
    print()

    # Evaluate each strategy at each noise level
    noise_rows: list[dict] = []
    for strat_name in test_strategies:
        params = all_strats[strat_name]
        signal_col = _get_signal_for_strategy(strat_name)
        print(f"  Evaluating {strat_name} (signal={signal_col})...")

        for sigma in NOISE_SIGMAS:
            year_results = evaluate_noisy_strategy(data_by_year, strat_name, params, sigma)

            # Aggregate across configs, models, entries to get mean PnL per year
            for year in YEARS:
                yr = year_results[year_results["year"] == year]
                if len(yr) == 0:
                    continue
                # Mean across all trials and all (model, config) combos
                mean_pnl = yr["portfolio_pnl"].mean()
                std_pnl = yr.groupby("trial")["portfolio_pnl"].sum().std() if sigma > 0 else 0.0
                noise_rows.append(
                    {
                        "strategy": strat_name,
                        "noise_sigma": sigma,
                        "year": year,
                        "mean_pnl": mean_pnl,
                        "std_pnl": std_pnl,
                    }
                )

    noise_df = pd.DataFrame(noise_rows)

    # Pivot to combined PnL
    noise_combined = (
        noise_df.groupby(["strategy", "noise_sigma"])
        .agg(
            total_pnl=("mean_pnl", "sum"),  # sum across years
            avg_std=("std_pnl", "mean"),
        )
        .reset_index()
    )

    # ── Print noise sensitivity table ──
    _subsection("Noise Sensitivity — Combined P&L")

    # Pivot: rows = strategy, columns = noise sigma
    pivot = noise_combined.pivot_table(
        index="strategy",
        columns="noise_sigma",
        values="total_pnl",
    )

    # Sort by σ=0 PnL (clean performance)
    if 0.0 in pivot.columns:
        pivot = pivot.sort_values(by=0.0, ascending=False)

    display = pivot.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    display.columns = [f"σ={c}" for c in display.columns]
    print(display.to_markdown())
    print()

    # ── Degradation analysis: at what σ does each strategy reach equal_active level? ──
    _subsection("Degradation Analysis")

    ea_clean = noise_combined[
        (noise_combined["strategy"] == "equal_active") & (noise_combined["noise_sigma"] == 0.0)
    ]
    ea_baseline = ea_clean["total_pnl"].iloc[0] if len(ea_clean) > 0 else 0.0

    print(f"equal_active baseline (σ=0): ${ea_baseline:,.0f}\n")

    for strat_name in top_signals:
        strat_data = noise_combined[noise_combined["strategy"] == strat_name].sort_values(
            "noise_sigma"
        )
        clean_pnl = strat_data[strat_data["noise_sigma"] == 0.0]["total_pnl"].iloc[0]
        degradation_sigma = None
        for _, row in strat_data.iterrows():
            if row["total_pnl"] <= ea_baseline and row["noise_sigma"] > 0:
                degradation_sigma = row["noise_sigma"]
                break

        if degradation_sigma is not None:
            print(
                f"  {strat_name}: clean=${clean_pnl:,.0f}, "
                f"degrades to equal_active at σ={degradation_sigma}"
            )
        else:
            print(
                f"  {strat_name}: clean=${clean_pnl:,.0f}, "
                f"stays above equal_active even at σ={NOISE_SIGMAS[-1]}"
            )

    print()

    # ── Plot: noise sensitivity curves ──
    plot_dir = ensure_plot_dir("signals")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for strategies
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i / max(len(test_strategies) - 1, 1)) for i in range(len(test_strategies))]

    for i, strat_name in enumerate(test_strategies):
        strat_data = noise_combined[noise_combined["strategy"] == strat_name].sort_values(
            "noise_sigma"
        )
        style = "--" if strat_name in ("uniform", "equal_active") else "-"
        linewidth = 2.5 if strat_name in ("uniform", "equal_active") else 1.5
        marker = "s" if strat_name in ("uniform", "equal_active") else "o"
        ax.plot(
            strat_data["noise_sigma"],
            strat_data["total_pnl"],
            style,
            color=colors[i],
            label=strat_name,
            linewidth=linewidth,
            marker=marker,
            markersize=5,
        )

    ax.set_xlabel("Noise σ (multiplicative lognormal)")
    ax.set_ylabel("Mean Combined P&L ($)")
    ax.set_title("Noise Sensitivity — Combined P&L vs. Signal Noise Level")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(ea_baseline, color="gray", linestyle=":", alpha=0.5, label="_ea_ref")

    fig.tight_layout()
    fig.savefig(plot_dir / "noise_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'noise_sensitivity.png'}")

    # ── Save CSV ──
    out_dir = ensure_output_dir()
    noise_combined.to_csv(out_dir / "noise_sensitivity.csv", index=False)
    print(f"  Saved: {out_dir / 'noise_sensitivity.csv'}")

    return noise_df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all analyses: strategy scorecard, effective N, noise sensitivity."""
    print("=" * 80)
    print("COMPARE SIGNALS — Category Allocation Strategy Comparison")
    print("=" * 80)
    print()
    print(f"Bankroll: ${BANKROLL:,.0f}/category")
    print(f"Models: {len(ALL_MODELS)} ({', '.join(short_model(m) for m in ALL_MODELS)})")
    print(f"Years: {YEARS}")
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

    # ── Analysis 2: Strategy Scorecard ──
    scorecard, combined = run_analysis_2(data_by_year)

    # ── Effective Sample Size ──
    _eigen_df = compute_effective_n(data_by_year)

    # ── Analysis 3: Noise Sensitivity ──
    _noise_df = run_noise_sensitivity(data_by_year, combined)

    # ── Summary ──
    _section("Summary")
    print("Key outputs:")
    print(f"  - {OUTPUT_DIR / 'signal_scorecard.csv'}")
    print(f"  - {OUTPUT_DIR / 'effective_n.csv'} (category-level, scenario-level, agg=100 N_eff)")
    print(f"  - {OUTPUT_DIR / 'noise_sensitivity.csv'}")
    print(f"  - {OUTPUT_DIR / 'plots' / 'signals' / 'scorecard_heatmap.png'}")
    print(f"  - {OUTPUT_DIR / 'plots' / 'signals' / 'effective_n_scree.png'}")
    print(f"  - {OUTPUT_DIR / 'plots' / 'signals' / 'noise_sensitivity.png'}")


if __name__ == "__main__":
    main()
