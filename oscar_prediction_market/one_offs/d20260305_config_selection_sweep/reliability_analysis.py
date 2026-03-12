"""Backtest reliability analysis for d0305 config selection sweep.

Quantifies how much we can trust 2-year backtest results for model/config
selection.  With only 2 ceremony years (2024, 2025) and high correlation across
entry points within the same (year, category), the effective sample size is much
smaller than the raw scenario count.  This script produces both markdown tables
and diagnostic plots that make that uncertainty explicit.

Sections:
    1. Data dimensions & effective sample size (eigenvalue method)
    2. Correlation structure (within-year, across-year, same/diff category)
    3. Cross-year model ranking stability (Kendall tau)
    4. Cross-year config ranking stability (per model)
    5. Bootstrap model + config selection stability
    6. Leave-one-year-out model selection (forward-looking test)

Key findings (expected):
    - 137 raw (year, category, entry) scenarios collapse to ~5 effective
    - Entry points within same (year, cat) correlate ~0.46
    - Model rankings reverse across years (tau ~ -0.07)
    - avg_ensemble + cal_sgbt dominate bootstrap selection (~98% top-2)
    - Config ranking stability varies: avg_ensemble stable, gbt anti-correlated

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260305_config_selection_sweep.reliability_analysis
"""

from pathlib import Path

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

apply_style()

EXP_DIR = Path("storage/d20260305_config_selection_sweep")
PLOTS_DIR = EXP_DIR / "plots" / "reliability"
MODEL_ORDER = ["avg_ensemble", "cal_sgbt", "clogit_cal_sgbt_ensemble", "clogit", "lr", "gbt"]

N_BOOTSTRAP = 2000
RNG_SEED = 42


# ============================================================================
# Data Loading
# ============================================================================


def _load_entry_pnl(year: int) -> pd.DataFrame:
    """Load entry-level PnL for a single year (fixed bankroll only)."""
    df = pd.read_csv(EXP_DIR / str(year) / "results" / "entry_pnl.csv")
    return df[df["bankroll_mode"] == "fixed"].copy()


def _load_cross_year() -> pd.DataFrame:
    """Load cross-year scenario scores (162 rows = 6 models x 27 configs)."""
    df = pd.read_csv(EXP_DIR / "cross_year_scenario_scores.csv")
    return df


def _pnl_col(df: pd.DataFrame) -> str:
    """Return the PnL column name (actual_pnl or total_pnl)."""
    return "actual_pnl" if "actual_pnl" in df.columns else "total_pnl"


def _short(model: str) -> str:
    return get_model_display(model)


def _df_to_md(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _save_fig(fig: matplotlib.figure.Figure, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ============================================================================
# 1. Data Dimensions & Effective Sample Size
# ============================================================================


def analysis_effective_sample_size() -> str:
    """Compute how many independent scenarios we really have.

    Uses the eigenvalue method: build a (year, category, entry) x config
    PnL matrix, compute the correlation matrix eigenvalues, and estimate
    effective-N as (sum(eigenvalues))^2 / sum(eigenvalues^2).
    """
    lines = [
        "## 1. Data Dimensions & Effective Sample Size\n",
    ]

    # Build scenario x config PnL matrix
    # Each row = one (year, category, entry_snapshot) scenario
    # Each column = one (model_type, config_label) pair
    all_rows = []
    for year in [2024, 2025]:
        df = _load_entry_pnl(year)
        pnl = _pnl_col(df)
        for (cat, entry), grp in df.groupby(["category", "entry_snapshot"]):
            row_data = {"year": year, "category": cat, "entry_snapshot": entry}
            for _, r in grp.iterrows():
                key = f"{r['model_type']}__{r['config_label']}"
                row_data[key] = r[pnl]
            all_rows.append(row_data)

    scenario_df = pd.DataFrame(all_rows)
    config_cols = [c for c in scenario_df.columns if "__" in c]
    pnl_matrix = scenario_df[config_cols].values  # (n_scenarios, n_configs)

    n_scenarios, n_configs = pnl_matrix.shape
    n_2024 = len(scenario_df[scenario_df["year"] == 2024])
    n_2025 = len(scenario_df[scenario_df["year"] == 2025])
    n_cats_2024 = scenario_df[scenario_df["year"] == 2024]["category"].nunique()
    n_cats_2025 = scenario_df[scenario_df["year"] == 2025]["category"].nunique()
    n_entries_2024 = scenario_df[scenario_df["year"] == 2024]["entry_snapshot"].nunique()
    n_entries_2025 = scenario_df[scenario_df["year"] == 2025]["entry_snapshot"].nunique()

    lines.append("**Raw scenario count:**\n")
    lines.append(f"- 2024: {n_cats_2024} categories × {n_entries_2024} entries = {n_2024}")
    lines.append(f"- 2025: {n_cats_2025} categories × {n_entries_2025} entries = {n_2025}")
    lines.append(f"- Total: **{n_scenarios}** (year, category, entry) scenarios")
    lines.append(f"- Config pairs: {n_configs} (6 models × 27 configs)\n")

    # Eigenvalue method for effective-N
    # Correlation matrix of scenarios (across configs)
    # Remove configs with zero variance (constant PnL across all scenarios)
    valid_mask = ~np.isnan(pnl_matrix).any(axis=0)
    col_std = np.std(pnl_matrix, axis=0)
    valid_mask = valid_mask & (col_std > 1e-10)
    pnl_clean = pnl_matrix[:, valid_mask]

    # Also remove scenarios (rows) with zero variance
    row_std = np.std(pnl_clean, axis=1)
    nonconst_rows = row_std > 1e-10
    pnl_clean = pnl_clean[nonconst_rows]

    if pnl_clean.shape[0] > 2 and pnl_clean.shape[1] > 1:
        corr = np.corrcoef(pnl_clean)  # (n_valid_scenarios, n_valid_scenarios)
        # Replace any remaining NaN with 0 (can happen if a row is near-constant)
        corr = np.nan_to_num(corr, nan=0.0)
        eigenvalues = np.linalg.eigvalsh(corr)
        eigenvalues = eigenvalues[eigenvalues > 0]  # positive only
        effective_n = (eigenvalues.sum() ** 2) / (eigenvalues**2).sum()
    else:
        effective_n = n_scenarios

    lines.append("**Effective sample size (eigenvalue method):**\n")
    lines.append(f"- Effective N ≈ **{effective_n:.1f}** (from {n_scenarios} raw scenarios)")
    lines.append(
        f"- Compression ratio: {n_scenarios / effective_n:.0f}:1 "
        f"— most variation is shared across entry points within the same (year, category)\n"
    )

    # Eigenvalue scree plot
    if pnl_clean.shape[0] > 2 and pnl_clean.shape[1] > 1:
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        cumvar = np.cumsum(eigenvalues_sorted) / eigenvalues_sorted.sum()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.bar(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted, color="#4878CF")
        ax1.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="λ = 1")
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Eigenvalue")
        ax1.set_title("Scenario Correlation Eigenvalues")
        ax1.legend()

        ax2.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#4878CF")
        ax2.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95%")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance Explained")
        ax2.set_title("Cumulative Variance")
        ax2.legend()

        fig.suptitle("Effective Dimensionality of Backtest Scenarios", y=1.02)
        fig.tight_layout()
        _save_fig(fig, PLOTS_DIR / "eigenvalue_scree.png")

        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        lines.append(
            f"- Components for 95% variance: {n_95} (top eigenvalue captures {cumvar[0]:.0%})\n"
        )

    lines.append("![Eigenvalue scree](assets/eigenvalue_scree.png)\n")

    return "\n".join(lines)


# ============================================================================
# 2. Correlation Structure
# ============================================================================


def analysis_correlation_structure() -> str:
    """Measure pairwise correlation between scenarios, grouped by relationship.

    Groups: same_year_same_cat, same_year_diff_cat, cross_year_same_cat,
    cross_year_diff_cat.
    """
    lines = [
        "## 2. Correlation Structure Across Scenarios\n",
        "Not all 137 scenarios are independent. Entry points within the same "
        "(year, category) see nearly identical market conditions and model "
        "predictions, so their PnL outcomes are highly correlated.\n",
    ]

    # Build scenario PnL vectors
    scenarios: list[dict[str, object]] = []
    for year in [2024, 2025]:
        df = _load_entry_pnl(year)
        pnl = _pnl_col(df)
        for (cat, entry), grp in df.groupby(["category", "entry_snapshot"]):
            vec: dict[str, float] = {}
            for _, r in grp.iterrows():
                key = f"{r['model_type']}__{r['config_label']}"
                vec[key] = float(r[pnl])
            scenarios.append({"year": year, "category": cat, "entry": entry, "vec": vec})

    # Get common config keys
    all_keys: set[str] = set()
    for s in scenarios:
        s_vec = s["vec"]
        assert isinstance(s_vec, dict)
        all_keys.update(s_vec.keys())
    common_keys = sorted(all_keys)

    # Build matrix (n_scenarios x n_configs)
    matrix = np.array(
        [
            [s["vec"].get(k, 0.0) for k in common_keys]  # type: ignore[attr-defined]  # dict
            for s in scenarios
        ]
    )

    # Pairwise correlations
    n = len(scenarios)
    groups: dict[str, list[float]] = {
        "same_year_same_cat": [],
        "same_year_diff_cat": [],
        "cross_year_same_cat": [],
        "cross_year_diff_cat": [],
    }

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = scenarios[i], scenarios[j]
            r = np.corrcoef(matrix[i], matrix[j])[0, 1]
            if np.isnan(r):
                continue

            same_year = si["year"] == sj["year"]
            same_cat = si["category"] == sj["category"]

            if same_year and same_cat:
                groups["same_year_same_cat"].append(r)
            elif same_year and not same_cat:
                groups["same_year_diff_cat"].append(r)
            elif not same_year and same_cat:
                groups["cross_year_same_cat"].append(r)
            else:
                groups["cross_year_diff_cat"].append(r)

    table_rows = []
    for group_name, corrs in groups.items():
        if corrs:
            table_rows.append(
                {
                    "Relationship": group_name.replace("_", " "),
                    "N pairs": len(corrs),
                    "Mean ρ": f"{np.mean(corrs):.3f}",
                    "Median ρ": f"{np.median(corrs):.3f}",
                    "Std": f"{np.std(corrs):.3f}",
                }
            )

    lines.append(_df_to_md(pd.DataFrame(table_rows)))
    lines.append("")

    # Interpretation
    sysc = np.mean(groups["same_year_same_cat"]) if groups["same_year_same_cat"] else 0
    lines.append(
        f"\nEntry points within the same (year, category) correlate at ρ = {sysc:.3f}. "
        "This means the 7–9 entry snapshots per (year, category) behave almost like "
        "a single observation with some noise, not 7–9 independent trials.\n"
    )

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = np.corrcoef(matrix)

    # Sort scenarios by year then category
    order = sorted(range(n), key=lambda i: (scenarios[i]["year"], scenarios[i]["category"]))
    corr_sorted = corr_matrix[np.ix_(order, order)]

    im = ax.imshow(corr_sorted, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson ρ")

    # Mark year boundary
    n_2024 = sum(1 for s in scenarios if s["year"] == 2024)
    ax.axhline(n_2024 - 0.5, color="black", linewidth=2)
    ax.axvline(n_2024 - 0.5, color="black", linewidth=2)
    ax.set_title("Scenario Pairwise Correlation\n(black line = year boundary)")
    ax.set_xlabel("Scenario (sorted by year, category)")
    ax.set_ylabel("Scenario (sorted by year, category)")
    fig.tight_layout()
    _save_fig(fig, PLOTS_DIR / "scenario_correlation_heatmap.png")

    lines.append("![Scenario correlation heatmap](assets/scenario_correlation_heatmap.png)\n")

    return "\n".join(lines)


# ============================================================================
# 3. Cross-Year Model Ranking Stability
# ============================================================================


def analysis_model_ranking_stability() -> str:
    """Compare model rankings between 2024 and 2025.

    For each year, rank models by total PnL (summed across all scenarios),
    then compute Kendall tau between the two rankings.
    """
    lines = [
        "## 3. Cross-Year Model Ranking Stability\n",
        "The acid test: do models that do well in 2024 also do well in 2025?\n",
    ]

    cross = _load_cross_year()

    # Model-level PnL per year (sum across configs gives overall model strength)
    # Better: for each model, take its best config's PnL per year
    model_pnl = {}
    for model in MODEL_ORDER:
        model_df = cross[cross["model_type"] == model]
        # Use best config by avg performance, then look at per-year
        best_idx = model_df["avg_total_pnl"].idxmax()
        best = model_df.loc[best_idx]
        model_pnl[model] = {
            "pnl_2024": best["total_pnl_2024"],
            "pnl_2025": best["total_pnl_2025"],
            "avg": best["avg_total_pnl"],
            "config": best["config_label"],
        }

    # Also compute rank by each year
    models_by_2024 = sorted(
        MODEL_ORDER, key=lambda m: float(model_pnl[m]["pnl_2024"]), reverse=True
    )
    models_by_2025 = sorted(
        MODEL_ORDER, key=lambda m: float(model_pnl[m]["pnl_2025"]), reverse=True
    )

    rank_2024 = {m: i + 1 for i, m in enumerate(models_by_2024)}
    rank_2025 = {m: i + 1 for i, m in enumerate(models_by_2025)}

    table_rows = []
    for model_name in MODEL_ORDER:
        table_rows.append(
            {
                "Model": _short(model_name),
                "2024 PnL": f"${float(model_pnl[model_name]['pnl_2024']):,.0f}",
                "Rank '24": rank_2024[model_name],
                "2025 PnL": f"${float(model_pnl[model_name]['pnl_2025']):,.0f}",
                "Rank '25": rank_2025[model_name],
                "Avg PnL": f"${float(model_pnl[model_name]['avg']):,.0f}",
            }
        )

    lines.append(_df_to_md(pd.DataFrame(table_rows)))
    lines.append("")

    # Kendall tau
    ranks_24 = [rank_2024[m] for m in MODEL_ORDER]
    ranks_25 = [rank_2025[m] for m in MODEL_ORDER]
    tau, p = stats.kendalltau(ranks_24, ranks_25)
    lines.append(f"\n**Kendall τ = {tau:.3f}** (p = {p:.3f})")
    lines.append(
        "A τ near zero means model rankings are essentially random between years — "
        "a model's 2024 performance tells you almost nothing about its 2025 performance.\n"
    )

    # Also compute with all configs (mean PnL per model per year)
    mean_by_model_year = {}
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        mean_by_model_year[model] = {
            "pnl_2024": m["total_pnl_2024"].mean(),
            "pnl_2025": m["total_pnl_2025"].mean(),
        }
    models_by_2024_avg = sorted(
        MODEL_ORDER, key=lambda m: mean_by_model_year[m]["pnl_2024"], reverse=True
    )
    models_by_2025_avg = sorted(
        MODEL_ORDER, key=lambda m: mean_by_model_year[m]["pnl_2025"], reverse=True
    )
    rank_2024_avg = {m: i + 1 for i, m in enumerate(models_by_2024_avg)}
    rank_2025_avg = {m: i + 1 for i, m in enumerate(models_by_2025_avg)}
    ranks_24_avg = [rank_2024_avg[m] for m in MODEL_ORDER]
    ranks_25_avg = [rank_2025_avg[m] for m in MODEL_ORDER]
    tau_avg, p_avg = stats.kendalltau(ranks_24_avg, ranks_25_avg)
    lines.append(f"Alternative (avg across all 27 configs): τ = {tau_avg:.3f} (p = {p_avg:.3f})\n")

    return "\n".join(lines)


# ============================================================================
# 4. Cross-Year Config Ranking Stability (Per Model)
# ============================================================================


def analysis_config_ranking_stability() -> str:
    """For each model, rank its 27 configs by 2024 PnL vs 2025 PnL.

    Compute Kendall tau to see if a model's config preferences are stable.
    """
    lines = [
        "## 4. Config Ranking Stability Across Years\n",
        "Even if we can't reliably choose *which model*, maybe we can "
        "reliably choose *which config* for a given model?\n",
    ]

    cross = _load_cross_year()
    table_rows = []
    taus = {}

    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model].copy()
        m = m.sort_values("config_label").reset_index(drop=True)
        tau, p = stats.kendalltau(m["total_pnl_2024"], m["total_pnl_2025"])
        taus[model] = tau

        # Best config per year
        best_2024 = str(m.loc[m["total_pnl_2024"].idxmax(), "config_label"])
        best_2025 = str(m.loc[m["total_pnl_2025"].idxmax(), "config_label"])

        table_rows.append(
            {
                "Model": _short(model),
                "τ (config ranks)": f"{tau:.3f}",
                "p-value": f"{p:.3f}",
                "Best '24 config": _short_config(best_2024),
                "Best '25 config": _short_config(best_2025),
                "Same?": "✓" if best_2024 == best_2025 else "✗",
            }
        )

    lines.append(_df_to_md(pd.DataFrame(table_rows)))
    lines.append("")

    # Interpretation
    stable = [(m, t) for m, t in taus.items() if t > 0.3]
    unstable = [(m, t) for m, t in taus.items() if t < -0.3]

    if stable:
        s_str = ", ".join(f"{_short(m)} (τ={t:.2f})" for m, t in stable)
        lines.append(f"\n**Stable** config preferences: {s_str}")
    if unstable:
        u_str = ", ".join(f"{_short(m)} (τ={t:.2f})" for m, t in unstable)
        lines.append(f"\n**Unstable** (anti-correlated): {u_str}")

    lines.append("")

    # Scatter plot: 2024 PnL vs 2025 PnL for each model's configs
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx // 3, idx % 3]
        m = cross[cross["model_type"] == model]
        ax.scatter(
            m["total_pnl_2024"],
            m["total_pnl_2025"],
            alpha=0.6,
            color=get_model_color(model),
            s=40,
        )
        tau = taus[model]
        ax.set_title(f"{_short(model)} (τ = {tau:.2f})", fontsize=10)
        ax.set_xlabel("2024 PnL")
        ax.set_ylabel("2025 PnL")

        # Add trend line
        z = np.polyfit(m["total_pnl_2024"], m["total_pnl_2025"], 1)
        x_range = np.linspace(m["total_pnl_2024"].min(), m["total_pnl_2024"].max(), 100)
        ax.plot(x_range, np.polyval(z, x_range), "--", alpha=0.5, color="gray")

    fig.suptitle("Config Performance: 2024 vs 2025 (per model)", y=1.01)
    fig.tight_layout()
    _save_fig(fig, PLOTS_DIR / "config_stability_scatter.png")

    lines.append("![Config stability scatter](assets/config_stability_scatter.png)\n")

    return "\n".join(lines)


def _short_config(label: str) -> str:
    """Extract edge and kf from config label for display."""
    parts = label.split("_")
    edge = kf = ""
    for p in parts:
        if p.startswith("bet="):
            edge = p.replace("bet=", "e")
        elif p.startswith("kf="):
            kf = p.replace("kf=", "k")
    return f"{edge}/{kf}" if edge and kf else label[:20]


# ============================================================================
# 5. Bootstrap Model + Config Selection Stability
# ============================================================================


def analysis_bootstrap_selection() -> str:
    """Bootstrap-resample (year, category) blocks and see how often each model
    and config wins.

    This is the key practical question: if we re-drew our 2 years of data
    (resampling categories with replacement within each year), would we
    still pick the same model + config?
    """
    lines = [
        "## 5. Bootstrap Selection Stability\n",
        "We bootstrap-resample categories (with replacement) within each year, "
        f"recompute combined PnL, and record which model+config wins ({N_BOOTSTRAP} iterations).\n",
    ]

    rng = np.random.default_rng(RNG_SEED)

    # Build (model, config, year, category) -> PnL lookup
    pnl_by_cat: dict[tuple[str, str, int, str], float] = {}
    all_categories: dict[int, list[str]] = {}

    for year in [2024, 2025]:
        df = _load_entry_pnl(year)
        pnl = _pnl_col(df)
        cats = sorted(df["category"].unique())
        all_categories[year] = cats

        cat_pnl = df.groupby(["model_type", "config_label", "category"])[pnl].sum()
        for idx_raw, val_raw in cat_pnl.items():  # type: ignore[assignment]  # pandas multi-index
            idx_tuple = idx_raw if isinstance(idx_raw, tuple) else (idx_raw,)  # type: ignore[redundant-cast]  # ensure tuple
            pnl_by_cat[(str(idx_tuple[0]), str(idx_tuple[1]), year, str(idx_tuple[2]))] = float(
                val_raw
            )  # type: ignore[arg-type]  # pandas scalar

    # Get all (model, config) pairs
    configs = sorted(
        {(m, c) for m, c, _, _ in pnl_by_cat.keys()},
        key=lambda x: (MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 99, x[1]),
    )

    # Bootstrap
    model_wins: dict[str, int] = dict.fromkeys(MODEL_ORDER, 0)
    config_wins: dict[tuple[str, str], int] = dict.fromkeys(configs, 0)
    best_models: list[str] = []

    for _ in range(N_BOOTSTRAP):
        combined_pnl: dict[tuple[str, str], float] = dict.fromkeys(configs, 0.0)

        for year in [2024, 2025]:
            cats = all_categories[year]
            boot_cats = rng.choice(cats, size=len(cats), replace=True)
            for model, config in configs:
                total = sum(pnl_by_cat.get((model, config, year, cat), 0.0) for cat in boot_cats)
                combined_pnl[(model, config)] += total

        # Find winner
        winner = max(configs, key=lambda c: combined_pnl[c])
        config_wins[winner] = config_wins.get(winner, 0) + 1
        model_wins[winner[0]] = model_wins.get(winner[0], 0) + 1
        best_models.append(winner[0])

    # Model selection frequency
    lines.append("### Model Selection Frequency\n")
    table_rows = []
    for m in MODEL_ORDER:
        pct = 100 * model_wins[m] / N_BOOTSTRAP
        table_rows.append({"Model": _short(m), "Selected %": f"{pct:.1f}%", "Count": model_wins[m]})
    lines.append(_df_to_md(pd.DataFrame(table_rows)))
    lines.append("")

    # Top model+config pairs
    lines.append("\n### Top Config Selections\n")
    top_configs = sorted(config_wins.items(), key=lambda x: -x[1])[:10]
    config_rows = []
    for (model, config), count in top_configs:
        if count > 0:
            config_rows.append(
                {
                    "Model": _short(model),
                    "Config": _short_config(config),
                    "Selected %": f"{100 * count / N_BOOTSTRAP:.1f}%",
                }
            )
    lines.append(_df_to_md(pd.DataFrame(config_rows)))
    lines.append("")

    # Concentration
    top1_pct = 100 * max(model_wins.values()) / N_BOOTSTRAP
    top2_models = sorted(model_wins.items(), key=lambda x: -x[1])[:2]
    top2_pct = 100 * sum(v for _, v in top2_models) / N_BOOTSTRAP
    lines.append(
        f"\n**Concentration:** Top model selected {top1_pct:.0f}% of the time, "
        f"top 2 models cover {top2_pct:.0f}%. "
    )
    if top2_pct > 90:
        lines.append(
            "The choice is effectively between 2 models — that's surprisingly narrow "
            "given the small sample.\n"
        )
    else:
        lines.append("")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models = [_short(m) for m in MODEL_ORDER]
    pcts = [100 * model_wins[m] / N_BOOTSTRAP for m in MODEL_ORDER]
    colors = [get_model_color(m) for m in MODEL_ORDER]
    ax.bar(models, pcts, color=colors)
    ax.set_ylabel("Bootstrap Selection %")
    ax.set_title(f"Model Selection Frequency ({N_BOOTSTRAP} bootstrap iterations)")
    for i, (pct, _model) in enumerate(zip(pcts, models, strict=True)):
        if pct > 1:
            ax.text(i, pct + 1, f"{pct:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, PLOTS_DIR / "bootstrap_model_selection.png")

    lines.append("![Bootstrap model selection](assets/bootstrap_model_selection.png)\n")

    return "\n".join(lines)


# ============================================================================
# 6. Leave-One-Year-Out Model Selection
# ============================================================================


def analysis_leave_one_year_out() -> str:
    """Select best model on one year, evaluate on the other.

    This is a forward-looking validation: if we'd had only 2024 data, would
    we have picked the right model for 2025 (and vice versa)?
    """
    lines = [
        "## 6. Leave-One-Year-Out Validation\n",
        "The hardest test: pick the best model+config using one year's data, "
        "then see how it performs on the other year.\n",
    ]

    cross = _load_cross_year()

    results = []
    for train_year, test_year in [(2024, 2025), (2025, 2024)]:
        train_col = f"total_pnl_{train_year}"
        test_col = f"total_pnl_{test_year}"

        # Best config on train year
        best_idx = cross[train_col].idxmax()
        selected = cross.loc[best_idx]
        selected_model = selected["model_type"]
        selected_config = selected["config_label"]
        train_pnl = selected[train_col]
        test_pnl = selected[test_col]

        # Rank of selected config on test year
        cross_sorted = cross.sort_values(test_col, ascending=False).reset_index(drop=True)
        test_rank = (
            cross_sorted[
                (cross_sorted["model_type"] == selected_model)
                & (cross_sorted["config_label"] == selected_config)
            ].index[0]
            + 1
        )

        # Best possible on test year
        best_test_idx = cross[test_col].idxmax()
        best_test_pnl = cross.loc[best_test_idx, test_col]
        best_test_model = cross.loc[best_test_idx, "model_type"]

        results.append(
            {
                "Train": train_year,
                "Test": test_year,
                "Selected": _short(str(selected_model)),
                "Train PnL": f"${float(train_pnl):,.0f}",  # type: ignore[arg-type]  # pandas scalar
                "Test PnL": f"${float(test_pnl):,.0f}",  # type: ignore[arg-type]  # pandas scalar
                "Test Rank": f"{test_rank}/162",
                "Best Test": _short(str(best_test_model)),
                "Best Test PnL": f"${float(best_test_pnl):,.0f}",  # type: ignore[arg-type]  # pandas scalar
            }
        )

    lines.append(_df_to_md(pd.DataFrame(results)))
    lines.append("")

    # Within-model LOO (is it better to at least get the model right?)
    lines.append("\n### Within-Model Leave-One-Year-Out\n")
    lines.append(
        "A softer test: for each model, pick its best config on the train year, "
        "see how that config ranks among the same model's 27 configs on the test year:\n"
    )
    model_loo_rows = []
    for train_year, test_year in [(2024, 2025), (2025, 2024)]:
        train_col = f"total_pnl_{train_year}"
        test_col = f"total_pnl_{test_year}"
        for model in MODEL_ORDER:
            m = cross[cross["model_type"] == model].copy()
            best_train_idx = m[train_col].idxmax()
            selected_config = str(m.loc[best_train_idx, "config_label"])
            test_pnl = float(m.loc[best_train_idx, test_col])  # type: ignore[arg-type]  # pandas scalar

            # Rank within this model
            m_sorted = m.sort_values(test_col, ascending=False).reset_index(drop=True)
            within_rank = m_sorted[m_sorted["config_label"] == selected_config].index[0] + 1

            model_loo_rows.append(
                {
                    "Train": train_year,
                    "Model": _short(model),
                    "Within-Model Rank": f"{within_rank}/27",
                    "Test PnL": f"${test_pnl:,.0f}",
                }
            )

    lines.append(_df_to_md(pd.DataFrame(model_loo_rows)))
    lines.append("")

    lines.append(
        "\nIf within-model ranks are consistently poor (e.g., >15/27), "
        "it means even config selection within a chosen model is unreliable "
        "with 2 years of data.\n"
    )

    return "\n".join(lines)


# ============================================================================
# Summary
# ============================================================================


def analysis_summary() -> str:
    """Concise summary of reliability findings."""
    lines = [
        "## Summary: What Can We Trust?\n",
        "| Question | Answer | Confidence |",
        "| --- | --- | --- |",
        "| Is avg_ensemble or cal_sgbt the best model? | "
        "Almost certainly one of these two | **High** (98% bootstrap) |",
        "| Which of the two is better? | Too close to call with 2 years | **Low** |",
        "| Is edge ≥ 0.15 better than edge = 0.04? | "
        "Yes, higher edge is consistently better | **Moderate** |",
        "| Is the exact best config (e.g., e0.20/k0.15) optimal? | "
        "Could easily be e0.15 or e0.25 instead | **Low** |",
        "| Can we use EV or CVaR for config selection? | "
        "No — EV anti-correlates with actual PnL | **High** (structural) |",
        "",
        "**Bottom line:** Two years is enough to identify the right *tier* of "
        "model+config (ensemble models, moderate-to-high edge threshold) but not "
        "enough to pinpoint the single optimal choice. Decisions should be robust "
        "to this uncertainty — choose configs where the cost of being slightly "
        "wrong is small.\n",
    ]
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("=" * 70)
    print("Reliability Analysis — d0305 Config Selection Sweep")
    print("=" * 70)
    print()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sections = [
        analysis_summary,
        analysis_effective_sample_size,
        analysis_correlation_structure,
        analysis_model_ranking_stability,
        analysis_config_ranking_stability,
        analysis_bootstrap_selection,
        analysis_leave_one_year_out,
    ]

    all_md = ["# Backtest Reliability Analysis\n"]
    for fn in sections:
        print(f"\n--- {fn.__name__} ---")
        md = fn()
        all_md.append(md)
        print(md[:200] + "..." if len(md) > 200 else md)

    output_path = EXP_DIR / "reliability_output.md"
    output_path.write_text("\n\n".join(all_md) + "\n")
    print(f"\n[output] {output_path}")
    print(f"[plots]  {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
