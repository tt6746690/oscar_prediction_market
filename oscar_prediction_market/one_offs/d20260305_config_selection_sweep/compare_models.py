"""Model comparison analysis for d0305 config selection sweep.

Compares all 6 models across multiple dimensions:
  A. Best-config comparison (each model's best config)
  B. Fixed-config comparison (multiple reference configs)
  C. Category-level breakdown
  D. Bootstrap model ranking (category + entry-point resampling)
  E. Pairwise bootstrap win-rate matrix
  F. Temporal stability (early vs late season)
  G. EV calibration per model

Outputs:
  - Markdown tables to stdout (redirect to README)
  - Plots to storage/d20260305_config_selection_sweep/plots/model_comparison/

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260305_config_selection_sweep.compare_models
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


def _df_to_md(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


apply_style()

EXP_DIR = Path("storage/d20260305_config_selection_sweep")
PLOTS_DIR = EXP_DIR / "plots" / "model_comparison"

MODEL_ORDER = ["avg_ensemble", "cal_sgbt", "clogit_cal_sgbt_ensemble", "clogit", "lr", "gbt"]

# Reference configs for fixed-config comparison (edge, kf)
REFERENCE_CONFIGS = [
    (0.10, 0.15, "moderate"),
    (0.15, 0.15, "recommended-mid"),
    (0.20, 0.15, "recommended"),
    (0.20, 0.05, "conservative"),
    (0.25, 0.15, "aggressive-edge"),
]


def _save_fig(fig: matplotlib.figure.Figure, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def _short(model: str) -> str:
    return get_model_display(model)


def _load_cross_year() -> pd.DataFrame:
    df = pd.read_csv(EXP_DIR / "cross_year_scenario_scores.csv")
    renames = {}
    for col in list(df.columns):
        if "total_pnl" in col and col.replace("total_pnl", "actual_pnl") not in df.columns:
            renames[col] = col.replace("total_pnl", "actual_pnl")
    if renames:
        df = df.rename(columns=renames)
    return df


def _load_entry_pnl(year: int) -> pd.DataFrame:
    df = pd.read_csv(EXP_DIR / str(year) / "results" / "entry_pnl.csv")
    return df[df["bankroll_mode"] == "fixed"]


def _load_category_pnl_matrix(
    year: int,
) -> tuple[np.ndarray, list[str], dict[tuple[str, str], int]]:
    """Build (n_configs, n_categories) PnL matrix from entry_pnl."""
    entry_df = _load_entry_pnl(year)
    pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"

    cat_pnl = entry_df.groupby(["config_label", "model_type", "category"], as_index=False)[
        pnl_col
    ].sum()
    cat_pnl = cat_pnl.rename(columns={pnl_col: "category_pnl"})  # type: ignore[call-overload]  # pandas groupby sum returns DataFrame

    categories = sorted(cat_pnl["category"].unique())
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    config_keys = (
        cat_pnl[["model_type", "config_label"]]
        .drop_duplicates()
        .sort_values(["model_type", "config_label"])
        .reset_index(drop=True)
    )
    config_to_idx: dict[tuple[str, str], int] = {
        (row["model_type"], row["config_label"]): i for i, row in config_keys.iterrows()
    }

    n_configs = len(config_to_idx)
    n_cats = len(categories)
    matrix = np.zeros((n_configs, n_cats))

    for _, row in cat_pnl.iterrows():
        key = (str(row["model_type"]), str(row["config_label"]))
        if key in config_to_idx:
            ci = config_to_idx[key]
            cat_i = cat_to_idx[str(row["category"])]
            matrix[ci, cat_i] = row["category_pnl"]

    return matrix, categories, config_to_idx


def _load_entry_level_matrix(
    year: int,
) -> tuple[np.ndarray, list[str], dict[tuple[str, str], int]]:
    """Build (n_configs, n_entries) PnL matrix by summing across categories per entry.

    Each column is one entry snapshot (summed across all categories for that entry).
    """
    entry_df = _load_entry_pnl(year)
    pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"

    # Sum across categories per (config, entry_snapshot)
    entry_pnl = entry_df.groupby(["config_label", "model_type", "entry_snapshot"], as_index=False)[
        pnl_col
    ].sum()
    entry_pnl = entry_pnl.rename(columns={pnl_col: "entry_pnl"})  # type: ignore[call-overload]  # pandas groupby sum returns DataFrame

    entries = sorted(entry_pnl["entry_snapshot"].unique())
    entry_to_idx = {e: i for i, e in enumerate(entries)}

    config_keys = (
        entry_pnl[["model_type", "config_label"]]
        .drop_duplicates()
        .sort_values(["model_type", "config_label"])
        .reset_index(drop=True)
    )
    config_to_idx: dict[tuple[str, str], int] = {
        (row["model_type"], row["config_label"]): i for i, row in config_keys.iterrows()
    }

    n_configs = len(config_to_idx)
    n_entries = len(entries)
    matrix = np.zeros((n_configs, n_entries))

    for _, row in entry_pnl.iterrows():
        key = (str(row["model_type"]), str(row["config_label"]))
        if key in config_to_idx:
            ci = config_to_idx[key]
            ei = entry_to_idx[str(row["entry_snapshot"])]
            matrix[ci, ei] = row["entry_pnl"]

    return matrix, entries, config_to_idx


# ============================================================================
# A. Best-config comparison
# ============================================================================


def analysis_best_config(cross: pd.DataFrame) -> str:
    """For each model, take its best config by combined actual P&L."""
    lines = [
        "## A. Best-Config Head-to-Head\n",
        "Each model's single best config (by combined 2024+2025 actual P&L):\n",
    ]

    rows = []
    plot_data: list[dict] = []
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if m.empty:
            continue
        best = m.loc[m["avg_actual_pnl"].idxmax()]
        combined = best["avg_actual_pnl"] * 2
        pnl_2024 = best.get("actual_pnl_2024", 0)
        pnl_2025 = best.get("actual_pnl_2025", 0)
        rows.append(
            {
                "Model": _short(model),
                "Edge": f"{best['buy_edge_threshold']:.2f}",
                "KF": f"{best['kelly_fraction']:.2f}",
                "P&L 2024": f"${pnl_2024:,.0f}",
                "P&L 2025": f"${pnl_2025:,.0f}",
                "Combined": f"${combined:,.0f}",
                "EV Combined": f"${best['avg_ev_pnl_blend'] * 2:,.0f}",
                "EV/Actual": f"{best['avg_ev_pnl_blend'] / best['avg_actual_pnl']:.2f}x"
                if best["avg_actual_pnl"] > 0
                else "N/A",
            }
        )
        plot_data.append(
            {
                "model": model,
                "pnl_2024": pnl_2024,
                "pnl_2025": pnl_2025,
            }
        )

    df = pd.DataFrame(rows)
    lines.append(_df_to_md(df))
    lines.append("")

    # --- Plot: stacked bar chart of 2024+2025 P&L by model ---
    fig, ax = plt.subplots(figsize=(9, 5))
    models_plot = [d["model"] for d in plot_data]
    pnl_24 = np.array([d["pnl_2024"] for d in plot_data])
    pnl_25 = np.array([d["pnl_2025"] for d in plot_data])
    x = np.arange(len(models_plot))
    colors = [get_model_color(m) for m in models_plot]

    ax.bar(x, pnl_24, width=0.6, label="2024", color=colors, alpha=0.6)
    ax.bar(
        x,
        pnl_25,
        width=0.6,
        bottom=pnl_24,
        label="2025",
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models_plot], fontsize=9)
    ax.set_ylabel("P&L ($)")
    ax.set_title("Best-Config P&L: 2024 + 2025 (stacked)", fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(loc="upper right")
    # Add combined total labels
    for i, (v24, v25) in enumerate(zip(pnl_24, pnl_25, strict=False)):
        total = v24 + v25
        ax.text(i, total + 50, f"${total:,.0f}", ha="center", va="bottom", fontsize=8)
    _save_fig(fig, PLOTS_DIR / "best_config_pnl.png")

    lines.append(
        "![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "best_config_pnl.png](assets/model_comparison/best_config_pnl.png)\n"
    )
    lines.append(
        "**Observation:** avg_ens's best config (edge=0.20, KF=0.05) achieves the "
        "highest combined P&L. cal_sgbt is close but relies on exceptional 2025 "
        "performance with modest 2024 returns. The top 3 ensemble models all use "
        "edge ≥ 0.20."
    )
    return "\n".join(lines)


# ============================================================================
# B. Fixed-config comparison
# ============================================================================


def analysis_fixed_config(cross: pd.DataFrame) -> str:
    """Compare all models at several reference configs."""
    lines = [
        "## B. Fixed-Config Comparison\n",
        "All 6 models evaluated at the same config — isolating model quality:\n",
    ]

    for edge, kf, label in REFERENCE_CONFIGS:
        sub = cross[(cross["buy_edge_threshold"] == edge) & (cross["kelly_fraction"] == kf)]
        if sub.empty:
            continue

        lines.append(f"\n### Config: edge={edge}, KF={kf} ({label})\n")
        rows = []
        for model in MODEL_ORDER:
            row = sub[sub["model_type"] == model]
            if row.empty:
                continue
            r = row.iloc[0]
            combined = r["avg_actual_pnl"] * 2
            rows.append(
                {
                    "Model": _short(model),
                    "P&L 2024": f"${r.get('actual_pnl_2024', 0):,.0f}",
                    "P&L 2025": f"${r.get('actual_pnl_2025', 0):,.0f}",
                    "Combined": f"${combined:,.0f}",
                    "EV": f"${r['avg_ev_pnl_blend'] * 2:,.0f}",
                }
            )

        df = pd.DataFrame(rows)
        lines.append(_df_to_md(df))
        lines.append("")

    # --- Plot: heatmap of combined P&L (models × configs) ---
    heatmap_data = []
    config_labels = []
    for edge, kf, _label in REFERENCE_CONFIGS:
        sub = cross[(cross["buy_edge_threshold"] == edge) & (cross["kelly_fraction"] == kf)]
        if sub.empty:
            continue
        config_labels.append(f"e={edge} k={kf}")
        row_vals = []
        for model in MODEL_ORDER:
            row = sub[sub["model_type"] == model]
            if row.empty:
                row_vals.append(0)
            else:
                row_vals.append(row.iloc[0]["avg_actual_pnl"] * 2)
        heatmap_data.append(row_vals)

    hm = np.array(heatmap_data).T  # (models × configs)
    fig, ax = plt.subplots(figsize=(9, 5))
    vabs = max(abs(hm.min()), abs(hm.max()))
    im = ax.imshow(hm, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            val = hm[i, j]
            brightness = (val + vabs) / (2 * vabs)
            color = "white" if brightness > 0.8 or brightness < 0.2 else "black"
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([_short(m) for m in MODEL_ORDER], fontsize=9)
    ax.set_title("Combined P&L by Model × Config", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Combined P&L ($)")
    _save_fig(fig, PLOTS_DIR / "fixed_config_heatmap.png")

    lines.append(
        "![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "fixed_config_heatmap.png](assets/model_comparison/fixed_config_heatmap.png)\n"
    )

    lines.append(
        "**Observation:** Model ranking is remarkably stable across configs. "
        "avg_ens and cal_sgbt swap top spots depending on config, but both "
        "consistently outperform the rest. The model tier (ensemble >> clogit >> lr >> gbt) "
        "holds regardless of config."
    )
    return "\n".join(lines)


# ============================================================================
# C. Category-level breakdown
# ============================================================================


def analysis_category_breakdown(cross: pd.DataFrame) -> str:
    """Per-category P&L by model (using best config per model)."""
    lines = [
        "## C. Category-Level Breakdown\n",
        "P&L by category, using each model's best config. Reveals which models excel where.\n",
    ]

    # Identify best config per model
    best_configs: dict[str, str] = {}
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if not m.empty:
            best_configs[model] = str(m.loc[m["avg_actual_pnl"].idxmax(), "config_label"])

    for year in [2024, 2025]:
        entry_df = _load_entry_pnl(year)
        pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"

        lines.append(f"\n### {year}\n")

        # Filter to best config per model, sum per category
        cat_rows = []
        for model in MODEL_ORDER:
            if model not in best_configs:
                continue
            cfg = best_configs[model]
            sub = entry_df[(entry_df["model_type"] == model) & (entry_df["config_label"] == cfg)]
            by_cat = sub.groupby("category")[pnl_col].sum()
            row = {"Model": _short(model)}
            for cat in sorted(by_cat.index):
                # Abbreviate category names
                short_cat = cat.replace("_", " ").title()[:12]
                row[short_cat] = f"${by_cat[cat]:,.0f}"
            row["Total"] = f"${by_cat.sum():,.0f}"
            cat_rows.append(row)

        df = pd.DataFrame(cat_rows)
        lines.append(_df_to_md(df))
        lines.append("")

    lines.append(
        "**Observation:** Category-level variation is large. Some models "
        "capture specific categories better (e.g., avg_ens on best_picture) "
        "while others may lose on the same category. This explains why "
        "category-bootstrap rank stability is important."
    )
    return "\n".join(lines)


# ============================================================================
# D. Bootstrap model ranking
# ============================================================================


def analysis_bootstrap_model_ranking(cross: pd.DataFrame, n_bootstrap: int = 5000) -> str:
    """Bootstrap model ranking via category resampling and entry-point resampling.

    For each bootstrap sample:
      1. Category bootstrap: resample K categories with replacement, sum PnL
      2. Entry bootstrap: resample E entries with replacement, sum PnL

    Ranks the 6 *models* (not configs) using each model's best config.
    """
    lines = [
        "## D. Bootstrap Model Ranking\n",
        f"Bootstrap ranking of models ({n_bootstrap:,} samples). "
        "Uses each model's best config. Resampling categories and entry "
        "points independently to assess stability.\n",
    ]

    # Best config per model
    best_configs: dict[str, str] = {}
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if not m.empty:
            best_configs[model] = str(m.loc[m["avg_actual_pnl"].idxmax(), "config_label"])

    models = [m for m in MODEL_ORDER if m in best_configs]
    n_models = len(models)

    all_rank1_pcts: dict[str, list[float]] = {m: [] for m in models}
    method_labels: list[str] = []

    for method_name, method_fn in [
        ("Category Bootstrap (portfolio-wide)", _bootstrap_category),
        ("Entry-Point Bootstrap (portfolio-wide)", _bootstrap_entry),
    ]:
        lines.append(f"\n### {method_name}\n")

        rankings = method_fn(models, best_configs, n_bootstrap)

        rows = []
        for i, model in enumerate(models):
            ranks = rankings[:, i]
            rank1_pct = (ranks == 1).mean() * 100
            all_rank1_pcts[model].append(rank1_pct)
            rows.append(
                {
                    "Model": _short(model),
                    "Mean Rank": f"{ranks.mean():.2f}",
                    "Median Rank": f"{np.median(ranks):.0f}",
                    "% Rank 1": f"{rank1_pct:.1f}%",
                    "% Top 2": f"{(ranks <= 2).mean() * 100:.1f}%",
                    "% Top 3": f"{(ranks <= 3).mean() * 100:.1f}%",
                }
            )

        method_labels.append(method_name.split(" (")[0])
        df = pd.DataFrame(rows)
        lines.append(_df_to_md(df))
        lines.append("")

    # --- Plot: grouped bar chart of % Rank 1 by bootstrap method ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_models)
    width = 0.35
    for j, label in enumerate(method_labels):
        pcts = [all_rank1_pcts[m][j] for m in models]
        offset = (j - 0.5) * width
        ax.bar(
            x + offset,
            pcts,
            width=width,
            label=label,
            color=[get_model_color(m) for m in models],
            alpha=0.6 + 0.3 * j,
            edgecolor="black",
            linewidth=0.5,
        )
        for i, v in enumerate(pcts):
            if v > 0:
                ax.text(x[i] + offset, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models], fontsize=9)
    ax.set_ylabel("% Rank 1")
    ax.set_title("Bootstrap: % Rank 1 by Model", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(v) for v in all_rank1_pcts.values()) * 1.2)
    _save_fig(fig, PLOTS_DIR / "bootstrap_rank1.png")

    lines.append(
        "![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "bootstrap_rank1.png](assets/model_comparison/bootstrap_rank1.png)\n"
    )

    return "\n".join(lines)


def _bootstrap_category(
    models: list[str],
    best_configs: dict[str, str],
    n_bootstrap: int,
) -> np.ndarray:
    """Category-resampling bootstrap: resample categories with replacement.

    Unions categories across years — a model gets 0 for categories not present
    in a given year (e.g., 2024 has 8 cats, 2025 has 9).
    """
    n_models = len(models)
    rng = np.random.default_rng(42)

    # First pass: collect all categories
    all_categories: set[str] = set()
    for year in [2024, 2025]:
        _, categories, _ = _load_category_pnl_matrix(year)
        all_categories.update(categories)
    all_cats_sorted = sorted(all_categories)
    cat_to_idx = {c: i for i, c in enumerate(all_cats_sorted)}
    n_cats = len(all_cats_sorted)

    # Build per-model category PnL aligned to union index
    pnl_matrix = np.zeros((n_models, n_cats))
    for year in [2024, 2025]:
        matrix, categories, config_to_idx = _load_category_pnl_matrix(year)
        year_cat_idx = [cat_to_idx[c] for c in categories]
        for mi, model in enumerate(models):
            key = (model, best_configs[model])
            if key in config_to_idx:
                ci = config_to_idx[key]
                for yi, gi in enumerate(year_cat_idx):
                    pnl_matrix[mi, gi] += matrix[ci, yi]

    # Bootstrap
    rankings = np.zeros((n_bootstrap, n_models), dtype=np.int32)
    for b in range(n_bootstrap):
        indices = rng.integers(0, n_cats, size=n_cats)
        boot_pnl = pnl_matrix[:, indices].sum(axis=1)
        order = np.argsort(-boot_pnl)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n_models + 1)
        rankings[b] = ranks

    return rankings


def _bootstrap_entry(
    models: list[str],
    best_configs: dict[str, str],
    n_bootstrap: int,
) -> np.ndarray:
    """Entry-point-resampling bootstrap: resample entry snapshots with replacement."""
    n_models = len(models)
    rng = np.random.default_rng(42)

    # Build per-model entry PnL vectors (combined across years)
    model_entry_pnl: dict[str, np.ndarray] = {}
    all_entries: list[str] = []
    for year in [2024, 2025]:
        matrix, entries, config_to_idx = _load_entry_level_matrix(year)
        for model in models:
            key = (model, best_configs[model])
            if key in config_to_idx:
                ci = config_to_idx[key]
                vec = matrix[ci]
                if model not in model_entry_pnl:
                    model_entry_pnl[model] = []  # type: ignore[assignment]
                model_entry_pnl[model] = np.concatenate(  # type: ignore[attr-defined]
                    [model_entry_pnl[model], vec]  # type: ignore[arg-type]
                )
        all_entries.extend(entries)

    # Build matrix: (n_models, n_entries_total)
    n_entries = len(all_entries)
    pnl_matrix = np.zeros((n_models, n_entries))
    for i, model in enumerate(models):
        if model in model_entry_pnl:
            pnl_matrix[i] = model_entry_pnl[model]

    # Bootstrap
    rankings = np.zeros((n_bootstrap, n_models), dtype=np.int32)
    for b in range(n_bootstrap):
        indices = rng.integers(0, n_entries, size=n_entries)
        boot_pnl = pnl_matrix[:, indices].sum(axis=1)
        order = np.argsort(-boot_pnl)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n_models + 1)
        rankings[b] = ranks

    return rankings


# ============================================================================
# E. Pairwise bootstrap win rate
# ============================================================================


def analysis_pairwise_winrate(cross: pd.DataFrame, n_bootstrap: int = 5000) -> str:
    """P(model A > model B) under category bootstrap."""
    lines = [
        "## E. Pairwise Bootstrap Win Rates\n",
        f"P(row model beats column model) under category bootstrap "
        f"({n_bootstrap:,} samples). Values >50% mean the row model wins more often.\n",
    ]

    best_configs: dict[str, str] = {}
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if not m.empty:
            best_configs[model] = str(m.loc[m["avg_actual_pnl"].idxmax(), "config_label"])

    models = [m for m in MODEL_ORDER if m in best_configs]
    n_models = len(models)
    rng = np.random.default_rng(42)

    # Build combined category PnL matrix (union categories across years)
    all_categories: set[str] = set()
    for year in [2024, 2025]:
        _, categories, _ = _load_category_pnl_matrix(year)
        all_categories.update(categories)
    all_cats_sorted = sorted(all_categories)
    cat_to_idx_union = {c: i for i, c in enumerate(all_cats_sorted)}
    n_cats = len(all_cats_sorted)

    pnl_matrix = np.zeros((n_models, n_cats))
    for year in [2024, 2025]:
        matrix, categories, config_to_idx = _load_category_pnl_matrix(year)
        year_cat_idx = [cat_to_idx_union[c] for c in categories]
        for mi, model in enumerate(models):
            key = (model, best_configs[model])
            if key in config_to_idx:
                ci = config_to_idx[key]
                for yi, gi in enumerate(year_cat_idx):
                    pnl_matrix[mi, gi] += matrix[ci, yi]

    # Pairwise win counts
    wins = np.zeros((n_models, n_models))
    for _b in range(n_bootstrap):
        indices = rng.integers(0, n_cats, size=n_cats)
        boot_pnl = pnl_matrix[:, indices].sum(axis=1)
        for i in range(n_models):
            for j in range(n_models):
                if boot_pnl[i] > boot_pnl[j]:
                    wins[i, j] += 1

    winrate = wins / n_bootstrap * 100

    # Format table
    header = "| | " + " | ".join(_short(m) for m in models) + " |"
    sep = "| --- |" + " ---: |" * n_models
    lines.append(header)
    lines.append(sep)
    for i, model in enumerate(models):
        cells = []
        for j in range(n_models):
            if i == j:
                cells.append("—")
            else:
                pct = winrate[i, j]
                cells.append(f"**{pct:.0f}%**" if pct > 50 else f"{pct:.0f}%")
        lines.append(f"| {_short(model)} | " + " | ".join(cells) + " |")
    lines.append("")

    # Also create a heatmap plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(winrate, cmap="RdYlGn", vmin=0, vmax=100)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                pct = winrate[i, j]
                color = "white" if pct > 80 or pct < 20 else "black"
                ax.text(
                    j,
                    i,
                    f"{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([_short(m) for m in models], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([_short(m) for m in models], fontsize=9)
    ax.set_title("Pairwise Win Rate: P(Row > Col) under Category Bootstrap", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Win Rate %")
    _save_fig(fig, PLOTS_DIR / "pairwise_winrate.png")

    lines.append(
        "![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "pairwise_winrate.png](assets/model_comparison/pairwise_winrate.png)\n"
    )

    return "\n".join(lines)


# ============================================================================
# F. Temporal stability
# ============================================================================


def analysis_temporal_stability(cross: pd.DataFrame) -> str:
    """How does each model's advantage evolve over the season? Early vs late."""
    lines = [
        "## F. Temporal Stability: Early vs Late Season\n",
        "Does the best model change depending on when you enter the market?\n",
    ]

    best_configs: dict[str, str] = {}
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if not m.empty:
            best_configs[model] = str(m.loc[m["avg_actual_pnl"].idxmax(), "config_label"])

    models = [m for m in MODEL_ORDER if m in best_configs]

    for year in [2024, 2025]:
        lines.append(f"\n### {year}\n")

        entry_df = _load_entry_pnl(year)
        pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"

        # Sorted list of entry snapshots
        all_entries = sorted(entry_df["entry_snapshot"].unique())
        n_entries = len(all_entries)
        mid = n_entries // 2
        early_entries = all_entries[:mid]
        late_entries = all_entries[mid:]

        lines.append(
            f"Entries split: early={len(early_entries)} "
            f"({early_entries[0]}..{early_entries[-1]}), "
            f"late={len(late_entries)} "
            f"({late_entries[0]}..{late_entries[-1]})\n"
        )

        rows = []
        for model in models:
            cfg = best_configs[model]
            sub = entry_df[(entry_df["model_type"] == model) & (entry_df["config_label"] == cfg)]
            early = sub[sub["entry_snapshot"].isin(early_entries)]
            late = sub[sub["entry_snapshot"].isin(late_entries)]

            early_pnl = early.groupby("entry_snapshot")[pnl_col].sum().sum()
            late_pnl = late.groupby("entry_snapshot")[pnl_col].sum().sum()
            total_pnl = early_pnl + late_pnl

            rows.append(
                {
                    "Model": _short(model),
                    "Early P&L": f"${early_pnl:,.0f}",
                    "Late P&L": f"${late_pnl:,.0f}",
                    "Total": f"${total_pnl:,.0f}",
                    "Late %": f"{late_pnl / total_pnl * 100:.0f}%" if total_pnl != 0 else "N/A",
                }
            )

        df = pd.DataFrame(rows)
        lines.append(_df_to_md(df))
        lines.append("")

    # Temporal plot: cumulative P&L by entry
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for yi, year in enumerate([2024, 2025]):
        ax = axes[yi]
        entry_df = _load_entry_pnl(year)
        pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"
        all_entries = sorted(entry_df["entry_snapshot"].unique())

        for model in models:
            cfg = best_configs[model]
            sub = entry_df[(entry_df["model_type"] == model) & (entry_df["config_label"] == cfg)]
            entry_totals = sub.groupby("entry_snapshot")[pnl_col].sum()
            cumulative = entry_totals.reindex(all_entries, fill_value=0).cumsum()
            ax.plot(
                range(len(all_entries)),
                cumulative.values,
                color=get_model_color(model),
                label=_short(model),
                linewidth=2,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
        ax.set_xlabel("Entry Point (chronological)")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title(f"{year}", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        # Use shortened entry names as ticks
        tick_positions = list(range(0, len(all_entries), max(1, len(all_entries) // 5)))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [all_entries[i].split("_", 1)[-1][:10] for i in tick_positions],
            fontsize=7,
            rotation=45,
            ha="right",
        )

    fig.suptitle("Cumulative P&L Over Season (Best Config Per Model)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, PLOTS_DIR / "temporal_cumulative.png")

    lines.append(
        "![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "temporal_cumulative.png](assets/model_comparison/temporal_cumulative.png)\n"
    )
    lines.append(
        "**Observation:** The cumulative P&L plot shows whether a model's "
        "advantage comes from a few big wins or steady accumulation. Models "
        "that jump late (post-BAFTA/SAG) are relying on late-season signal; "
        "models that accumulate steadily are more robust to entry timing."
    )

    return "\n".join(lines)


# ============================================================================
# G. EV calibration per model
# ============================================================================


def analysis_ev_calibration(cross: pd.DataFrame) -> str:
    """Bin by predicted edge, compare predicted vs realized win rate."""
    lines = [
        "## G. EV Calibration: Why EV Fails as a Config Selector\n",
        "### The mechanism\n",
        "EV is computed as $\\text{EV} = \\sum_i p_i^{\\text{blend}} \\cdot \\text{PnL}_i$ "
        "where $p_i$ comes from averaged model + market probabilities. "
        "Every trade the model takes has positive expected value *by construction* — "
        "otherwise edge < 0 and the trade wouldn't be taken.\n",
        "",
        "The anti-correlation arises because:\n",
        "1. **Lower edge threshold → more trades taken**. At edge=0.02, "
        "avg_ens takes 77 trades/year vs 29 at edge=0.20.",
        "2. **Marginal trades have high EV but negative realized P&L**. "
        "The model's probability estimates are optimistic for small-edge "
        "positions — it thinks it sees +2% edge, but it's really noise.",
        "3. **EV sums all those optimistic estimates**, so more trades = "
        "higher EV. But actual P&L subtracts the losses from noise trades.",
        "",
        "### Detailed breakdown (avg_ensemble, 2024, KF=0.05)\n",
    ]

    entry_df = _load_entry_pnl(2024)
    pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"

    m = entry_df[(entry_df["model_type"] == "avg_ensemble") & (entry_df["kelly_fraction"] == 0.05)]

    rows = []
    for et in sorted(m["buy_edge_threshold"].unique()):
        sub = m[m["buy_edge_threshold"] == et]
        total_trades = sub["total_trades"].sum()
        total_ev = sub["ev_pnl_blend"].sum()
        total_pnl = sub[pnl_col].sum()
        total_fees = sub["total_fees"].sum()
        zero_entries = (sub["total_trades"] == 0).sum()
        n_entries = len(sub)
        ratio = total_ev / total_pnl if total_pnl != 0 else float("inf")

        rows.append(
            {
                "Edge": f"{et:.2f}",
                "Trades": f"{total_trades:.0f}",
                "Active Entries": f"{n_entries - zero_entries}/{n_entries}",
                "EV": f"${total_ev:,.0f}",
                "Actual": f"${total_pnl:,.0f}",
                "Fees": f"${total_fees:,.0f}",
                "EV/Actual": f"{ratio:.2f}x",
            }
        )

    df = pd.DataFrame(rows)
    lines.append(_df_to_md(df))
    lines.append("")

    lines.append("### Key insight\n")
    lines.append(
        "Going from edge=0.20 to edge=0.02 adds 48 trades. "
        "These trades collectively add **+$1,029 to EV** but "
        "**subtract -$1,323 from actual P&L**. The marginal trades are "
        "EV-positive according to the model but EV-negative in reality — "
        "the model is overconfident on small-edge positions.\n"
    )
    lines.append("### Cross-model EV inflation at fixed config (edge=0.15, KF=0.15)\n")

    # Show EV inflation per model at a fixed config
    cross_rows = []
    for model in MODEL_ORDER:
        row = cross[
            (cross["model_type"] == model)
            & (cross["buy_edge_threshold"] == 0.15)
            & (cross["kelly_fraction"] == 0.15)
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        ev = r["avg_ev_pnl_blend"]
        actual = r["avg_actual_pnl"]
        ratio = ev / actual if actual > 0 else float("inf")
        cross_rows.append(
            {
                "Model": _short(model),
                "Avg EV": f"${ev:,.0f}",
                "Avg Actual": f"${actual:,.0f}",
                "Inflation": f"{ratio:.2f}x",
            }
        )

    df2 = pd.DataFrame(cross_rows)
    lines.append(_df_to_md(df2))
    lines.append("")

    lines.append("### Implications for config selection\n")
    lines.append(
        "- **Do NOT maximize EV** to select configs. Higher EV = more marginal trades = worse returns."
    )
    lines.append(
        "- **EV is useful for model-level comparison** — it correctly identifies the model tier."
    )
    lines.append(
        "- **Use realized P&L patterns** (bootstrap rank stability, cross-year correlation) for config selection."
    )
    lines.append(
        "- **avg_ens has the lowest inflation** (1.10x), making it the most "
        "trustworthy for any EV-based forward-looking estimates."
    )

    # EV inflation bar plot
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = []
    colors = []
    labels = []
    for model in MODEL_ORDER:
        row = cross[
            (cross["model_type"] == model)
            & (cross["buy_edge_threshold"] == 0.15)
            & (cross["kelly_fraction"] == 0.15)
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        actual = r["avg_actual_pnl"]
        ev = r["avg_ev_pnl_blend"]
        if actual > 0:
            vals.append(ev / actual)
        else:
            vals.append(0)
        colors.append(get_model_color(model))
        labels.append(_short(model))

    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.8, edgecolor="white")
    for bar, v in zip(bars, vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.05,
            f"{v:.2f}x",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Perfect calibration")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("EV / Actual P&L Ratio")
    ax.set_title("EV Inflation by Model (edge=0.15, KF=0.15)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, PLOTS_DIR / "ev_inflation.png")

    lines.append(
        "\n![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "ev_inflation.png](assets/model_comparison/ev_inflation.png)\n"
    )

    return "\n".join(lines)


# ============================================================================
# Bootstrap model ranking per-category
# ============================================================================


def analysis_per_category_model_ranking(cross: pd.DataFrame, n_bootstrap: int = 5000) -> str:
    """Per-category bootstrap model ranking: which model is best for each category?"""
    lines = [
        "## D2. Per-Category Model Ranking\n",
        "For each category, bootstrap over entry points to rank models. "
        "Uses each model's best overall config.\n",
    ]

    best_configs: dict[str, str] = {}
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if not m.empty:
            best_configs[model] = str(m.loc[m["avg_actual_pnl"].idxmax(), "config_label"])

    models = [m for m in MODEL_ORDER if m in best_configs]
    rng = np.random.default_rng(42)

    # Collect all categories across years
    all_categories: set[str] = set()
    for year in [2024, 2025]:
        entry_df = _load_entry_pnl(year)
        all_categories.update(entry_df["category"].unique())

    for cat in sorted(all_categories):
        # Build (n_models, n_entries) matrix for this category
        entry_pnl_vecs: dict[str, list[float]] = {m: [] for m in models}
        for year in [2024, 2025]:
            entry_df = _load_entry_pnl(year)
            pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"
            cat_df = entry_df[entry_df["category"] == cat]
            entries = sorted(cat_df["entry_snapshot"].unique())

            for model in models:
                cfg = best_configs[model]
                sub = cat_df[(cat_df["model_type"] == model) & (cat_df["config_label"] == cfg)]
                by_entry = sub.set_index("entry_snapshot")[pnl_col].reindex(entries, fill_value=0)
                entry_pnl_vecs[model].extend(by_entry.values.tolist())

        n_entries = len(entry_pnl_vecs[models[0]])
        if n_entries == 0:
            continue

        pnl_matrix = np.zeros((len(models), n_entries))
        for i, model in enumerate(models):
            pnl_matrix[i] = entry_pnl_vecs[model]

        # Bootstrap
        rank1_counts = np.zeros(len(models))
        for _b in range(n_bootstrap):
            indices = rng.integers(0, n_entries, size=n_entries)
            boot_pnl = pnl_matrix[:, indices].sum(axis=1)
            best_idx = np.argmax(boot_pnl)
            rank1_counts[best_idx] += 1

        rank1_pct = rank1_counts / n_bootstrap * 100

        # Actual total PnL
        actual_totals = pnl_matrix.sum(axis=1)

        cat_display = cat.replace("_", " ").title()
        lines.append(f"\n**{cat_display}** (entries: {n_entries})")
        best_model = models[int(np.argmax(rank1_pct))]
        lines.append(
            f" — Most frequent #1: {_short(best_model)} "
            f"({rank1_pct[models.index(best_model)]:.0f}%)"
        )

        cat_rows = []
        for i, model in enumerate(models):
            cat_rows.append(
                {
                    "Model": _short(model),
                    "Total P&L": f"${actual_totals[i]:,.0f}",
                    "% Rank 1": f"{rank1_pct[i]:.0f}%",
                }
            )
        df = pd.DataFrame(cat_rows)
        lines.append(_df_to_md(df))
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# H. Model Scorecard
# ============================================================================


def analysis_model_scorecard(cross: pd.DataFrame) -> str:
    """Unified model scorecard consolidating all key signals into one table.

    Signals (from PLAN section 15):
    - Both-year profitable %: fraction of configs profitable in both 2024+2025
    - Mean combined actual PnL: avg across all configs (annual mean × 2)
    - Best combined actual PnL: ceiling — best config (annual mean × 2)
    - EV inflation: mean(EV) / mean(actual) across all configs (lower = better)
    - Within-model Spearman ρ (EV↔actual): how well EV ranks configs within model
    - Cross-year Spearman ρ: corr(actual_2024_rank, actual_2025_rank)
    - Year balance: mean 2024 / mean 2025 actual (closer to 1 = more balanced)
    - CVaR-5% (best config): tail risk at the recommended config
    - P(loss): what fraction of MC scenarios are negative at the best config
    """
    lines = [
        "## H. Model Scorecard\n",
        "All key model-selection signals in one table. This consolidates "
        "findings from sections A–G into a single comparison framework.\n",
    ]

    rows = []
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if m.empty:
            continue

        n_both = m["profitable_both"].sum()
        n_total = len(m)
        both_pct = n_both / n_total * 100

        mean_combined = m["avg_actual_pnl"].mean() * 2
        best_combined = m["avg_actual_pnl"].max() * 2

        mean_ev = m["avg_ev_pnl_blend"].mean()
        mean_actual = m["avg_actual_pnl"].mean()
        inflation = mean_ev / mean_actual if mean_actual > 0 else float("inf")

        # Within-model Spearman: EV vs actual
        ev_actual_rho = stats.spearmanr(m["avg_ev_pnl_blend"], m["avg_actual_pnl"]).statistic  # type: ignore[union-attr]  # scipy SignificanceResult

        # Cross-year Spearman: actual 2024 rank vs actual 2025 rank
        cross_year_rho = stats.spearmanr(m["actual_pnl_2024"], m["actual_pnl_2025"]).statistic  # type: ignore[union-attr]  # scipy SignificanceResult

        # Year balance: mean 2024 / mean 2025
        mean_2024 = m["actual_pnl_2024"].mean()
        mean_2025 = m["actual_pnl_2025"].mean()
        year_ratio = mean_2024 / mean_2025 if mean_2025 != 0 else float("inf")

        # CVaR-5% at best config (by actual PnL)
        best_row = m.loc[m["avg_actual_pnl"].idxmax()]
        cvar5 = best_row["avg_cvar_5"]

        rows.append(
            {
                "Model": _short(model),
                "Both %": f"{both_pct:.0f}%",
                "Mean Comb.": f"${mean_combined:,.0f}",
                "Best Comb.": f"${best_combined:,.0f}",
                "Inflation": f"{inflation:.2f}x",
                "EV↔Act ρ": f"{ev_actual_rho:+.3f}",
                "Yr↔Yr ρ": f"{cross_year_rho:+.3f}",
                "Yr Balance": f"{year_ratio:.2f}",
                "CVaR-5%": f"${cvar5:,.0f}",
            }
        )

    df = pd.DataFrame(rows)
    lines.append(_df_to_md(df))
    lines.append("")

    lines.append("### Reading the scorecard\n")
    lines.append(
        "- **Both %**: % of 27 configs profitable in both years. "
        "100% = robust to config choice.\n"
        "- **Mean/Best Comb.**: average and ceiling combined (2024+2025) "
        "actual P&L across all configs.\n"
        "- **Inflation**: mean(EV) / mean(actual). "
        "Lower = model's EV is more honest. 1.0 = perfect.\n"
        "- **EV↔Act ρ**: Spearman correlation between EV and actual P&L "
        "across configs *within* this model. Negative means EV is "
        "anti-correlated with actual — higher-EV configs perform worse. "
        "This is a reversal from the d20260225 full grid because the "
        "targeted grid only varies edge/KF (see §I for deep explanation).\n"
        "- **Yr↔Yr ρ**: cross-year rank correlation of actual P&L. "
        "Higher = config rankings transfer better across years.\n"
        "- **Yr Balance**: mean 2024 actual / mean 2025 actual. "
        "Closer to 1.0 = less dependent on one year's results. "
        "Values << 1 mean the model relies heavily on 2025.\n"
        "- **CVaR-5%**: mean P&L in the worst 5% of MC scenarios at the "
        "model's best actual config. More negative = worse tail risk. "
        "Computed using model probabilities — see §I for caveats.\n"
    )

    lines.append("### Key observations\n")
    lines.append(
        "- **No model dominates all dimensions.** avg_ensemble is most balanced "
        "(lowest inflation, highest year balance, good cross-year ρ) but has "
        "negative EV↔actual correlation like all models in this grid.\n"
        "- **EV↔actual is negative for all models** in the targeted grid. "
        "This does NOT mean EV is broken — it means within a fixed-structure "
        "grid (only edge/KF varying), lower edge thresholds inflate EV while "
        "degrading actual returns (see §I).\n"
        "- **Cross-year ρ is the best discriminator**: avg_ens (0.862) and "
        "clog_sgbt (0.819) have positive cross-year correlation — their "
        "good configs stay good across years. gbt (-0.848) and lr (-0.515) "
        "show anti-correlation — classic overfitting.\n"
    )
    return "\n".join(lines)


# ============================================================================
# I. EV & CVaR Investigation
# ============================================================================


def analysis_ev_cvar_investigation(cross: pd.DataFrame) -> str:
    """Deep investigation: usefulness of EV and CVaR for config/model selection.

    Documents why EV is anti-correlated with actual P&L in the targeted grid,
    why CVaR is mostly redundant with edge threshold, and recommendations
    for what to use instead.
    """
    lines = [
        "## I. EV & CVaR: What They Tell Us and What They Don't\n",
        "This section investigates whether EV (expected P&L) and CVaR "
        "(conditional value-at-risk) are useful for config selection in the "
        "targeted grid, and recommends alternatives.\n",
    ]

    # --- Part 1: EV as config selector ---
    lines.append("### EV as a config selector: anti-correlated in the targeted grid\n")
    lines.append(
        "In the d20260225 full grid (sweeping fee_type, kelly_mode, directions, "
        "edge, KF), within-model EV↔actual Spearman was +0.91 to +0.98. "
        "In the d20260305 targeted grid (only edge/KF varying), it's **-0.57 to "
        "-0.90**. Why the reversal?\n"
    )
    lines.append(
        "**The structural parameters drove the positive correlation in d20260225.** "
        "Switches like `taker→maker` fees or `independent→multi_outcome` Kelly "
        "create large EV *and* actual P&L differences that move in the same "
        "direction. Once those are fixed (as in the targeted grid), the remaining "
        "variation is edge threshold — and lower edge = more marginal trades = "
        "more EV inflation = *worse* actual returns.\n"
    )
    lines.append(
        "| Grid | Structural params varied? | EV↔actual direction | What EV captures |\n"
        "| --- | --- | --- | --- |\n"
        "| d20260225 (full) | Yes (fee, kelly_mode, side) | **Positive** (+0.9) | "
        "Structural quality differences |\n"
        "| d20260305 (targeted) | No (only edge, KF) | **Negative** (-0.9) | "
        "Edge threshold = trade quantity, not quality |\n"
    )
    lines.append(
        "**Implication:** EV is useful for comparing structural choices "
        "(multi_outcome > independent, taker > maker) but harmful for "
        "selecting edge/KF within a fixed structure. Do not maximize EV "
        "in the targeted grid.\n"
    )

    # --- Part 2: CVaR investigation ---
    lines.append("### CVaR: mostly a proxy for edge threshold\n")
    lines.append(
        "CVaR-5% measures tail risk — the mean P&L in the worst 5% of MC "
        "scenarios. In principle, it should capture risk beyond what edge "
        "threshold alone shows. In practice:\n"
    )

    # Raw correlation
    corr_rows = []
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model]
        if m.empty:
            continue
        r_cvar_actual = stats.spearmanr(m["avg_cvar_5"], m["avg_actual_pnl"]).statistic  # type: ignore[union-attr]  # scipy SignificanceResult
        r_cvar_edge = stats.spearmanr(m["avg_cvar_5"], m["buy_edge_threshold"]).statistic  # type: ignore[union-attr]  # scipy SignificanceResult
        r_edge_actual = stats.spearmanr(m["buy_edge_threshold"], m["avg_actual_pnl"]).statistic  # type: ignore[union-attr]  # scipy SignificanceResult

        # Partial correlation: CVaR↔actual after controlling for edge
        edge = m["buy_edge_threshold"].to_numpy(dtype=float)
        cvar = m["avg_cvar_5"].to_numpy(dtype=float)
        actual = m["avg_actual_pnl"].to_numpy(dtype=float)
        cvar_resid = cvar - np.polyval(np.polyfit(edge, cvar, 1), edge)
        actual_resid = actual - np.polyval(np.polyfit(edge, actual, 1), edge)
        partial_rho = stats.spearmanr(cvar_resid, actual_resid).statistic  # type: ignore[union-attr]  # scipy SignificanceResult

        corr_rows.append(
            {
                "Model": _short(model),
                "CVaR↔actual": f"{r_cvar_actual:+.3f}",
                "CVaR↔edge": f"{r_cvar_edge:+.3f}",
                "edge↔actual": f"{r_edge_actual:+.3f}",
                "Partial (CVaR↔act|edge)": f"{partial_rho:+.3f}",
            }
        )

    df = pd.DataFrame(corr_rows)
    lines.append(_df_to_md(df))
    lines.append("")

    lines.append(
        "**The CVaR↔edge column (ρ = +0.54 to +0.91) shows CVaR is largely "
        "a proxy for edge threshold.** After controlling for edge (partial "
        "correlation column), CVaR's relationship to actual P&L is inconsistent "
        "— positive for some models, negative for others. It adds no reliable "
        "signal for config selection.\n"
    )

    # CVaR as a constraint: what does it filter?
    lines.append("### CVaR as a constraint: binary, not smooth\n")
    lines.append(
        "At various loss bounds (as % of $10K bankroll), how many avg_ensemble "
        "configs survive the CVaR-5% constraint?\n"
    )

    m = cross[cross["model_type"] == "avg_ensemble"]
    constraint_rows = []
    for bound_pct in [0.10, 0.15, 0.20, 0.30, 0.50]:
        bound = bound_pct * 10000
        feasible = m[m["avg_cvar_5"] >= -bound]
        best_actual = feasible["avg_actual_pnl"].max() * 2 if len(feasible) > 0 else 0
        constraint_rows.append(
            {
                "Loss Bound": f"{bound_pct:.0%}",
                "Feasible": f"{len(feasible)}/{len(m)}",
                "Best Combined P&L": f"${best_actual:,.0f}" if len(feasible) > 0 else "—",
            }
        )

    df2 = pd.DataFrame(constraint_rows)
    lines.append(_df_to_md(df2))
    lines.append("")

    lines.append(
        "CVaR acts as a binary switch: at L=15%, only 3/27 survive "
        "(all edge=0.25, KF=0.05). At L=20%, all 27 survive. "
        "There's no smooth middle ground where CVaR helps distinguish "
        "between configs.\n"
    )

    # Why CVaR is a proxy
    lines.append("### Why CVaR is redundant with edge threshold\n")
    lines.append(
        "CVaR is computed via Monte Carlo: sample winners per category using "
        "model probabilities, look up P&L, repeat 10,000 times, take the worst "
        "5%. The problem:\n\n"
        "1. **Model probabilities are overconfident** — the same bias that makes EV "
        "unreliable also makes CVaR unreliable. CVaR measures tail risk *according "
        "to a miscalibrated model*, not actual tail risk.\n"
        "2. **Edge threshold IS the risk dial.** Higher edge → fewer positions → "
        "less capital at risk → less tail exposure. CVaR simply reflects this.\n"
        "3. **Kelly fraction barely matters** with multi_outcome mode. The optimizer "
        "finds its own solution regardless of starting KF, so CVaR barely varies "
        "within an edge level.\n"
    )

    # Show avg_ensemble CVaR pivot to prove point 3
    pivot = m.pivot_table(values="avg_cvar_5", index="buy_edge_threshold", columns="kelly_fraction")
    lines.append("avg_ensemble CVaR-5% by edge × KF ($ values):\n")
    lines.append("```")
    lines.append(pivot.round(0).to_string())
    lines.append("```\n")
    lines.append(
        "CVaR varies by $450 across edge thresholds (the main risk dial) "
        "but only ~$20 across KF levels at a fixed edge (noise). "
        "You get the same risk information from `edge ≥ 0.15` as from "
        "`CVaR-5% ≥ -$1,800`.\n"
    )

    # Recommendation
    lines.append("### Recommendation: simplified config selection\n")
    lines.append(
        "Given the findings above, we recommend **dropping EV and CVaR from "
        "config selection within the targeted grid** and using a simpler approach:\n\n"
        "1. **Model selection:** Use the scorecard (§H) — avg_ensemble is the "
        "most robust choice (highest cross-year ρ, lowest inflation, 100% "
        "both-year profitable).\n"
        "2. **Edge threshold as the risk dial:** Pick directly based on risk "
        "tolerance:\n"
        "   - Conservative: edge ≥ 0.20 (fewer trades, higher selectivity)\n"
        "   - Moderate: edge = 0.15 (balanced)\n"
        "   - Aggressive: edge = 0.10 (more trades, more noise)\n"
        "3. **Kelly fraction:** Default to 0.05 or 0.15 — it barely matters "
        "with multi_outcome Kelly.\n"
        "4. **Structural params (already fixed):** taker fees, multi_outcome "
        "Kelly, all directions.\n\n"
        "EV and CVaR remain useful for:\n"
        "- **Cross-model comparison** (EV correctly identifies the model tier)\n"
        "- **Reporting** (CVaR answers 'what's the worst 5% scenario?', useful "
        "for bankroll expectations, though caveated by model overconfidence)\n"
        "- **Structural parameter selection** (EV correctly ranks fee_type, "
        "kelly_mode, directions)\n"
    )

    return "\n".join(lines)


# ============================================================================
# J. Per-Model Pareto Frontiers
# ============================================================================


def analysis_per_model_pareto(cross: pd.DataFrame) -> str:
    """Risk-return Pareto frontier for each model separately.

    X-axis: CVaR-5% (risk — more negative = riskier)
    Y-axis: actual combined P&L (return)

    Also shows which configs are on the Pareto frontier per model.
    """
    lines = [
        "## J. Per-Model Pareto Frontiers\n",
        "Risk-return tradeoff for each model independently. "
        "The Pareto frontier shows configs where no other config offers "
        "higher return at the same or lower risk.\n",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes_flat[idx]
        m = cross[cross["model_type"] == model].copy()
        if m.empty:
            continue

        risk = m["avg_cvar_5"].to_numpy(dtype=float)
        ret = m["avg_actual_pnl"].to_numpy(dtype=float) * 2  # combined
        edges = m["buy_edge_threshold"].to_numpy(dtype=float)
        kfs = m["kelly_fraction"].to_numpy(dtype=float)

        # Find Pareto frontier: for each point, is there another with
        # higher return AND lower risk (higher CVaR)?
        n = len(m)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (
                    ret[j] >= ret[i]
                    and risk[j] >= risk[i]
                    and (ret[j] > ret[i] or risk[j] > risk[i])
                ):
                    is_pareto[i] = False
                    break

        color = get_model_color(model)
        ax.scatter(risk[~is_pareto], ret[~is_pareto], c=color, alpha=0.4, s=40)
        ax.scatter(
            risk[is_pareto],
            ret[is_pareto],
            c=color,
            alpha=1.0,
            s=80,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label="Pareto",
        )

        # Label Pareto points
        for i in np.where(is_pareto)[0]:
            ax.annotate(
                f"e={edges[i]:.2f}\nk={kfs[i]:.2f}",
                (risk[i], ret[i]),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Connect Pareto frontier with line
        pareto_idx: np.ndarray = np.where(is_pareto)[0]
        if len(pareto_idx) > 1:
            order: np.ndarray = np.argsort(risk[pareto_idx])
            ax.plot(
                risk[pareto_idx[order]],
                ret[pareto_idx[order]],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )

        ax.set_xlabel("CVaR-5% ($)")
        ax.set_ylabel("Combined Actual P&L ($)")
        ax.set_title(_short(model), fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle("Per-Model Pareto Frontiers: CVaR-5% vs Actual P&L", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, PLOTS_DIR / "per_model_pareto.png")

    lines.append(
        "![storage/d20260305_config_selection_sweep/plots/model_comparison/"
        "per_model_pareto.png](assets/model_comparison/per_model_pareto.png)\n"
    )

    # Table of Pareto configs per model
    lines.append("### Pareto frontier configs per model\n")
    for model in MODEL_ORDER:
        m = cross[cross["model_type"] == model].copy()
        if m.empty:
            continue

        risk = m["avg_cvar_5"].to_numpy(dtype=float)
        ret = m["avg_actual_pnl"].to_numpy(dtype=float) * 2
        n = len(m)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (
                    ret[j] >= ret[i]
                    and risk[j] >= risk[i]
                    and (ret[j] > ret[i] or risk[j] > risk[i])
                ):
                    is_pareto[i] = False
                    break

        pareto_idx_tbl: np.ndarray = np.where(is_pareto)[0]
        pareto_df = m.iloc[pareto_idx_tbl].sort_values("avg_cvar_5")

        lines.append(f"**{_short(model)}** ({is_pareto.sum()} Pareto configs):\n")
        tbl_rows = []
        for _, r in pareto_df.iterrows():
            tbl_rows.append(
                {
                    "Edge": f"{r['buy_edge_threshold']:.2f}",
                    "KF": f"{r['kelly_fraction']:.2f}",
                    "Combined P&L": f"${r['avg_actual_pnl'] * 2:,.0f}",
                    "CVaR-5%": f"${r['avg_cvar_5']:,.0f}",
                    "EV Combined": f"${r['avg_ev_pnl_blend'] * 2:,.0f}",
                }
            )
        df = pd.DataFrame(tbl_rows)
        lines.append(_df_to_md(df))
        lines.append("")

    lines.append(
        "**Takeaway:** For avg_ensemble, all Pareto configs use high edge "
        "thresholds (0.20–0.25). The frontier is nearly flat — small risk "
        "reduction for modest return sacrifice. This confirms that edge "
        "threshold is the primary risk lever; within a model, the Pareto "
        "frontier is not a rich optimization space."
    )

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("Loading data...", flush=True)
    cross = _load_cross_year()
    print(f"  {len(cross)} configs loaded")
    print(f"  Models: {sorted(cross['model_type'].unique())}", flush=True)
    print()

    sections = []

    print("A. Best-config comparison...", flush=True)
    sections.append(analysis_best_config(cross))

    print("B. Fixed-config comparison...", flush=True)
    sections.append(analysis_fixed_config(cross))

    print("C. Category-level breakdown...", flush=True)
    sections.append(analysis_category_breakdown(cross))

    print("D. Bootstrap model ranking...", flush=True)
    sections.append(analysis_bootstrap_model_ranking(cross))

    print("D2. Per-category model ranking...", flush=True)
    sections.append(analysis_per_category_model_ranking(cross))

    print("E. Pairwise bootstrap win rates...", flush=True)
    sections.append(analysis_pairwise_winrate(cross))

    print("F. Temporal stability...", flush=True)
    sections.append(analysis_temporal_stability(cross))

    print("G. EV calibration...", flush=True)
    sections.append(analysis_ev_calibration(cross))

    print("H. Model scorecard...", flush=True)
    sections.append(analysis_model_scorecard(cross))

    print("I. EV & CVaR investigation...", flush=True)
    sections.append(analysis_ev_cvar_investigation(cross))

    print("J. Per-model Pareto frontiers...", flush=True)
    sections.append(analysis_per_model_pareto(cross))

    # Write markdown
    output = PLOTS_DIR.parent.parent / "model_comparison_output.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write("# Model Comparison Deep-Dive\n\n")
        f.write(
            "Comprehensive comparison of 6 models across the d0305 targeted sweep "
            "(27 configs × 6 models × 2 years). Analysis uses existing backtest "
            "results — no re-running needed.\n\n"
        )
        f.write("**Models:** " + ", ".join(_short(m) for m in MODEL_ORDER) + "\n\n")
        for section in sections:
            f.write(section)
            f.write("\n\n")

    print(f"\nMarkdown output: {output}")
    print(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
