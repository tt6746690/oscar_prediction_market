"""Analysis 4: Joint optimization over (config × allocation × model) space.

Does the optimal config change when reallocation is added?
Does the optimal model change?
This is the **interaction analysis** — the core question is whether config selection
and allocation strategy are separable (can be optimized independently) or entangled
(must be jointly optimized).

If separable: pick the best config under uniform, then layer the best allocation on top.
If entangled: the full (model × config × strategy) space matters and the naive two-step
approach leaves money on the table.

**Section A — Full (Config × Strategy) Sweep**:
Evaluate all ~5,346 (model × config × strategy) triples. Identify top triples,
check whether avg_ensemble's optimal config shifts under reallocation.

**Section B — Edge Threshold Interaction**:
For avg_ensemble, does the optimal edge threshold depend on the allocation strategy?
Heatmap of strategy × edge → mean PnL. Scatter of aggressiveness vs optimal edge.

**Section C — Model Interaction**:
Does model ranking change under reallocation? Compare uniform vs best-per-model
strategy. If a weaker model benefits disproportionately from reallocation, the
ranking may shift.

**Section D — Pareto Frontier (PnL vs MC CVaR)**:
For avg_ensemble, build the (combined_PnL, MC_CVaR_5%) frontier across
(config × strategy) pairs. Identify Pareto-optimal points. Compare with uniform.

**Section E — Configuration Recommendation**:
Synthesize: what's the best (model, config, allocation) triple?

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260305_portfolio_kelly.joint_optimization
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.portfolio_simulation import (
    CategoryScenario,
    compute_cvar,
    sample_portfolio_pnl,
)
from oscar_prediction_market.one_offs.d20260305_portfolio_kelly.shared import (
    ALL_MODELS,
    YEARS,
    aggregate_combined,
    aggregate_to_year_level,
    df_to_md,
    ensure_output_dir,
    ensure_plot_dir,
    evaluate_all_strategies_by_year,
    get_all_strategies,
    get_prospective_strategies,
    load_all_data,
    prepare_data,
    short_model,
)

# ─── Constants ───────────────────────────────────────────────────────────────

REC_MODEL = "avg_ensemble"
REC_EDGE = 0.20
REC_KF = 0.05

# Edge thresholds present in the targeted grid
EDGE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
KF_VALUES = [0.05, 0.10, 0.15]

N_BOOTSTRAP = 2_000
N_MC_SAMPLES = 10_000
BOOTSTRAP_SEED = 42

# Representative strategies for the Pareto analysis (keep compute tractable).
# Span: baseline, active-only, signal types, aggressiveness, caps.
PARETO_STRATEGIES = [
    "uniform",
    "equal_active",
    "ev_100",
    "ev_50",
    "ev_cap30",
    "edge_100",
    "edge_50",
    "capital_100",
    "maxedge_100",
    "npos_100",
]


# ─── Fast numpy bootstrap helpers ────────────────────────────────────────────


def _precompute_group_arrays(entry_group: pd.DataFrame) -> dict:
    """Pre-extract numpy arrays from an entry group for fast bootstrap.

    Avoids repeated pandas indexing in the tight bootstrap loop.
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
    """Pure-numpy weight computation — same logic as shared.compute_weights."""
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


# ─── Display helpers ─────────────────────────────────────────────────────────


def _section(title: str) -> None:
    """Print a markdown section header."""
    print(f"\n{'=' * 80}")
    print(f"## {title}")
    print(f"{'=' * 80}\n")


def _subsection(title: str) -> None:
    """Print a markdown subsection header."""
    print(f"\n### {title}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Full (Config × Strategy) Sweep
# ═══════════════════════════════════════════════════════════════════════════════


def run_full_sweep(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Evaluate every (model × config × strategy) triple, return combined PnL.

    This is the core computation: ~6 models × 27 configs × ~33 strategies × 2 years
    = ~10,692 evaluations. Each evaluation sums weighted PnL across entry snapshots.

    Returns:
        DataFrame with one row per (strategy, model_type, config_label) and columns
        for pnl_2024, pnl_2025, pnl_combined.
    """
    strategies = get_all_strategies()
    print(
        f"Evaluating {len(ALL_MODELS)} models × {len(strategies)} strategies "
        f"across {len(YEARS)} years..."
    )

    results = evaluate_all_strategies_by_year(data_by_year, strategies)
    year_level = aggregate_to_year_level(results)
    combined = aggregate_combined(year_level)

    print(f"Total triples: {len(combined):,}")
    return combined


def analyze_full_sweep(combined: pd.DataFrame) -> None:
    """Section A analysis: top triples, config shift under reallocation."""
    _section("A: Full (Config × Strategy) Sweep")

    # ── A1: Top 20 triples by combined PnL ──
    _subsection("A1: Top 20 Triples by Combined PnL")

    top20 = combined.nlargest(20, "pnl_combined").copy()
    top20["model_short"] = top20["model_type"].map(short_model)
    display_cols = [
        "model_short",
        "config_label",
        "strategy",
        "buy_edge_threshold",
        "kelly_fraction",
        "pnl_2024",
        "pnl_2025",
        "pnl_combined",
    ]
    print(df_to_md(top20[display_cols]))

    # ── A2: avg_ensemble config shift under reallocation ──
    _subsection("A2: avg_ensemble — Does Optimal Config Shift Under Reallocation?")

    ens = combined[combined["model_type"] == REC_MODEL].copy()

    # Best config under uniform
    uniform = ens[ens["strategy"] == "uniform"]
    best_uniform = uniform.nlargest(1, "pnl_combined").iloc[0]
    print(
        f"Best config under uniform: edge={best_uniform['buy_edge_threshold']}, "
        f"KF={best_uniform['kelly_fraction']}, "
        f"PnL={best_uniform['pnl_combined']:,.0f}"
    )

    # Best config under each strategy
    rows: list[dict] = []
    prospective = get_prospective_strategies()
    for strat_name in prospective:
        strat_data = ens[ens["strategy"] == strat_name]
        if strat_data.empty:
            continue
        best = strat_data.nlargest(1, "pnl_combined").iloc[0]
        rows.append(
            {
                "strategy": strat_name,
                "best_edge": best["buy_edge_threshold"],
                "best_kf": best["kelly_fraction"],
                "best_pnl": best["pnl_combined"],
                "uniform_edge": best_uniform["buy_edge_threshold"],
                "uniform_kf": best_uniform["kelly_fraction"],
                "config_shifted": (
                    best["buy_edge_threshold"] != best_uniform["buy_edge_threshold"]
                    or best["kelly_fraction"] != best_uniform["kelly_fraction"]
                ),
            }
        )

    config_shift = pd.DataFrame(rows).sort_values("best_pnl", ascending=False)
    n_shifted = config_shift["config_shifted"].sum()
    n_total = len(config_shift)
    print(f"\n{n_shifted}/{n_total} strategies have a different optimal config vs uniform.\n")
    print(df_to_md(config_shift.head(15), float_fmt=".2f"))

    # ── A3: Per-model best strategy ──
    _subsection("A3: Per-Model Best Strategy vs Uniform Baseline")

    rows = []
    for model in ALL_MODELS:
        model_data = combined[combined["model_type"] == model]

        # Best uniform config
        uni = model_data[model_data["strategy"] == "uniform"]
        if uni.empty:
            continue
        best_uni = uni.nlargest(1, "pnl_combined").iloc[0]

        # Best (config, strategy) triple (excluding oracle)
        prospective_data = model_data[model_data["strategy"] != "oracle"]
        best_all = prospective_data.nlargest(1, "pnl_combined").iloc[0]

        rows.append(
            {
                "model": short_model(model),
                "uniform_best_pnl": best_uni["pnl_combined"],
                "uniform_edge": best_uni["buy_edge_threshold"],
                "uniform_kf": best_uni["kelly_fraction"],
                "joint_best_pnl": best_all["pnl_combined"],
                "joint_strategy": best_all["strategy"],
                "joint_edge": best_all["buy_edge_threshold"],
                "joint_kf": best_all["kelly_fraction"],
                "uplift": best_all["pnl_combined"] - best_uni["pnl_combined"],
                "uplift_pct": (
                    (best_all["pnl_combined"] / best_uni["pnl_combined"] - 1) * 100
                    if best_uni["pnl_combined"] != 0
                    else float("nan")
                ),
            }
        )

    model_summary = pd.DataFrame(rows)
    print(df_to_md(model_summary, float_fmt=".1f"))


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Edge Threshold Interaction
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_edge_interaction(combined: pd.DataFrame) -> pd.DataFrame:
    """Section B: does the optimal edge threshold depend on the allocation strategy?

    The hypothesis: reallocation amplifies PnL from high-edge categories, so the
    optimal edge threshold might shift lower (more permissive) or higher (more
    selective) depending on the strategy.

    For each (strategy, edge_threshold) pair, compute mean PnL across KF values.
    This marginalizes over KF to isolate the edge×strategy interaction.

    Returns:
        Long-form DataFrame: strategy, buy_edge_threshold, mean_pnl
    """
    _section("B: Edge Threshold Interaction")

    ens = combined[combined["model_type"] == REC_MODEL].copy()
    prospective = get_prospective_strategies()

    # ── Compute mean PnL across KF for each (strategy, edge) ──
    edge_strat_rows: list[dict] = []
    for strat_name in prospective:
        strat_data = ens[ens["strategy"] == strat_name]
        for edge in sorted(strat_data["buy_edge_threshold"].unique()):
            edge_data = strat_data[strat_data["buy_edge_threshold"] == edge]
            edge_strat_rows.append(
                {
                    "strategy": strat_name,
                    "buy_edge_threshold": edge,
                    "mean_pnl": edge_data["pnl_combined"].mean(),
                    "max_pnl": edge_data["pnl_combined"].max(),
                }
            )

    edge_interaction = pd.DataFrame(edge_strat_rows)

    # ── Table: optimal edge per strategy ──
    _subsection("B1: Optimal Edge Threshold per Strategy")

    opt_rows: list[dict] = []
    uniform_opt = edge_interaction[edge_interaction["strategy"] == "uniform"]
    uniform_best_edge = uniform_opt.loc[uniform_opt["mean_pnl"].idxmax(), "buy_edge_threshold"]

    for strat_name in prospective:
        strat_data = edge_interaction[edge_interaction["strategy"] == strat_name]
        if strat_data.empty:
            continue
        best_idx = strat_data["mean_pnl"].idxmax()
        best_row = strat_data.loc[best_idx]
        opt_rows.append(
            {
                "strategy": strat_name,
                "optimal_edge": best_row["buy_edge_threshold"],
                "mean_pnl_at_optimal": best_row["mean_pnl"],
                "uniform_optimal_edge": uniform_best_edge,
                "edge_shifted": best_row["buy_edge_threshold"] != uniform_best_edge,
            }
        )

    opt_df = pd.DataFrame(opt_rows).sort_values("mean_pnl_at_optimal", ascending=False)
    n_shifted = opt_df["edge_shifted"].sum()
    print(
        f"{n_shifted}/{len(opt_df)} strategies have a different optimal edge vs uniform "
        f"(uniform optimal = {uniform_best_edge}).\n"
    )
    print(df_to_md(opt_df.head(20), float_fmt=".2f"))

    # ── Heatmap: strategy × edge → mean PnL ──
    _subsection("B2: Heatmap — Strategy × Edge → Mean PnL")

    # Pivot for heatmap. Select a representative subset of strategies.
    heatmap_strats = [
        "uniform",
        "equal_active",
        "ev_100",
        "ev_50",
        "ev_25",
        "ev_cap30",
        "edge_100",
        "edge_50",
        "capital_100",
        "maxedge_100",
        "npos_100",
    ]
    heatmap_data = edge_interaction[edge_interaction["strategy"].isin(heatmap_strats)]
    pivot = heatmap_data.pivot_table(
        index="strategy",
        columns="buy_edge_threshold",
        values="mean_pnl",
    )
    # Reorder rows by max PnL
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    print(df_to_md(pivot.reset_index(), float_fmt=",.0f"))

    _plot_edge_heatmap(pivot)

    # ── Scatter: strategy aggressiveness vs optimal edge ──
    _subsection("B3: Aggressiveness vs Optimal Edge")

    all_strats = get_all_strategies()
    for _, row in opt_df.iterrows():
        strat_key = row["strategy"]
        if strat_key in all_strats:
            opt_df.loc[opt_df["strategy"] == strat_key, "aggressiveness"] = all_strats[
                strat_key
            ].get("aggressiveness", 1.0)

    if "aggressiveness" in opt_df.columns:
        _plot_aggressiveness_vs_edge(opt_df)

    return edge_interaction


def _plot_edge_heatmap(pivot: pd.DataFrame) -> None:
    """Plot heatmap of strategy × edge → mean PnL."""
    plot_dir = ensure_plot_dir("joint")

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{e:.2f}" for e in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                color = "white" if abs(val - pivot.values.mean()) > pivot.values.std() else "black"
                ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xlabel("Buy Edge Threshold")
    ax.set_ylabel("Strategy")
    ax.set_title("Mean Combined PnL by Strategy × Edge Threshold (avg_ensemble)")
    fig.colorbar(im, ax=ax, label="Mean Combined PnL ($)")

    fig.tight_layout()
    fig.savefig(plot_dir / "edge_strategy_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {plot_dir / 'edge_strategy_heatmap.png'}")


def _plot_aggressiveness_vs_edge(opt_df: pd.DataFrame) -> None:
    """Scatter: strategy aggressiveness vs optimal edge threshold."""
    plot_dir = ensure_plot_dir("joint")

    df = opt_df.dropna(subset=["aggressiveness"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["aggressiveness"], df["optimal_edge"], alpha=0.6, s=50)

    # Label some points
    for _, row in df.iterrows():
        ax.annotate(
            row["strategy"],
            (row["aggressiveness"], row["optimal_edge"]),
            fontsize=6,
            alpha=0.7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Aggressiveness")
    ax.set_ylabel("Optimal Edge Threshold")
    ax.set_title("Strategy Aggressiveness vs Optimal Edge (avg_ensemble)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "aggressiveness_vs_edge.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_dir / 'aggressiveness_vs_edge.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Model Interaction
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_model_interaction(combined: pd.DataFrame) -> pd.DataFrame:
    """Section C: does model ranking change under reallocation?

    For each model, find:
    1. Best config under uniform allocation
    2. Best (config, strategy) under joint optimization (non-oracle)
    3. Reallocation uplift = joint_best - uniform_best

    If the model ranking (by best PnL) changes between uniform and joint,
    then model selection and allocation are entangled.

    Returns:
        DataFrame comparing model rankings under uniform vs reallocation.
    """
    _section("C: Model Interaction — Does Model Ranking Change?")

    rows: list[dict] = []
    for model in ALL_MODELS:
        model_data = combined[combined["model_type"] == model]

        # Best under uniform
        uni = model_data[model_data["strategy"] == "uniform"]
        if uni.empty:
            continue
        best_uni = uni.nlargest(1, "pnl_combined").iloc[0]

        # Best under joint (non-oracle)
        joint = model_data[~model_data["strategy"].isin(["oracle"])]
        best_joint = joint.nlargest(1, "pnl_combined").iloc[0]

        rows.append(
            {
                "model": model,
                "model_short": short_model(model),
                # Uniform
                "uniform_pnl": best_uni["pnl_combined"],
                "uniform_edge": best_uni["buy_edge_threshold"],
                "uniform_kf": best_uni["kelly_fraction"],
                # Joint
                "joint_pnl": best_joint["pnl_combined"],
                "joint_strategy": best_joint["strategy"],
                "joint_edge": best_joint["buy_edge_threshold"],
                "joint_kf": best_joint["kelly_fraction"],
                # Uplift
                "uplift": best_joint["pnl_combined"] - best_uni["pnl_combined"],
            }
        )

    model_df = pd.DataFrame(rows)

    # Rank under uniform vs joint
    model_df["rank_uniform"] = model_df["uniform_pnl"].rank(ascending=False).astype(int)
    model_df["rank_joint"] = model_df["joint_pnl"].rank(ascending=False).astype(int)
    model_df["rank_change"] = model_df["rank_uniform"] - model_df["rank_joint"]

    _subsection("C1: Model Rankings — Uniform vs Joint Optimization")

    display_cols = [
        "model_short",
        "uniform_pnl",
        "rank_uniform",
        "joint_pnl",
        "joint_strategy",
        "rank_joint",
        "rank_change",
        "uplift",
    ]
    print(df_to_md(model_df.sort_values("rank_uniform")[display_cols], float_fmt=",.0f"))

    rank_changes = (model_df["rank_change"] != 0).sum()
    print(f"\n{rank_changes}/{len(model_df)} models changed rank under reallocation.")

    # ── C2: Reallocation uplift by model ──
    _subsection("C2: Reallocation Uplift by Model")

    for _, row in model_df.sort_values("uplift", ascending=False).iterrows():
        pct = row["uplift"] / row["uniform_pnl"] * 100 if row["uniform_pnl"] != 0 else float("nan")
        print(
            f"- **{row['model_short']}**: ${row['uplift']:,.0f} uplift "
            f"({pct:+.1f}%) via {row['joint_strategy']} "
            f"(edge={row['joint_edge']}, KF={row['joint_kf']})"
        )

    _plot_model_ranking_shift(model_df)

    return model_df


def _plot_model_ranking_shift(model_df: pd.DataFrame) -> None:
    """Plot model PnL under uniform vs joint optimization (paired bar chart)."""
    plot_dir = ensure_plot_dir("joint")

    sorted_df = model_df.sort_values("uniform_pnl", ascending=False)
    models = sorted_df["model_short"].values
    uniform_pnl = sorted_df["uniform_pnl"].to_numpy(dtype=float)
    joint_pnl = sorted_df["joint_pnl"].to_numpy(dtype=float)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2, uniform_pnl, width, label="Best Uniform Config", color="#4C72B0", alpha=0.9
    )
    ax.bar(
        x + width / 2,
        joint_pnl,
        width,
        label="Best (Config × Strategy)",
        color="#55A868",
        alpha=0.9,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Combined PnL ($)")
    ax.set_title("Model PnL: Best Uniform Config vs Best Joint (Config × Strategy)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate uplift
    for i, (u, j) in enumerate(zip(uniform_pnl, joint_pnl, strict=True)):
        uplift = j - u
        if uplift > 0:
            ax.annotate(
                f"+${uplift:,.0f}",
                (x[i] + width / 2, j),
                fontsize=7,
                ha="center",
                va="bottom",
                color="#2E7D32",
            )

    fig.tight_layout()
    fig.savefig(plot_dir / "model_ranking_shift.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {plot_dir / 'model_ranking_shift.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Pareto Frontier (PnL vs MC CVaR)
# ═══════════════════════════════════════════════════════════════════════════════


def mc_cvar(
    scenario_pnl_by_year: dict[int, pd.DataFrame],
    entry_pnl_by_year: dict[int, pd.DataFrame],
    model: str,
    config_label: str,
    strategy_params: dict,
    alpha: float = 0.05,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute MC outcome-simulation CVaR-5% for a (model, config, strategy) triple.

    Unlike bootstrap resampling (which reshuffles observed category PnLs), this
    simulates actual Oscar-night outcomes: for each MC draw, independently sample
    one winner per category from a blended probability distribution, look up the
    per-nominee PnL, weight by the allocation strategy, and sum across categories
    and entry snapshots.

    Procedure per MC sample:
    1. For each (year, entry_snapshot):
       a. Compute allocation weights from entry_pnl signals.
       b. For each category, build blend probabilities (model+market)/2,
          scale nominee PnLs by the category allocation weight.
       c. Independently sample one winner per category.
    2. Sum weighted PnL across categories and entry snapshots → one realization.
    3. CVaR-5% = mean of worst alpha-fraction of realizations.

    Args:
        scenario_pnl_by_year: {year: scenario_pnl_df} with per-nominee pnl/probs.
        entry_pnl_by_year: {year: prepared_entry_pnl} with signals for weights.
        model: model_type to filter to.
        config_label: config to filter to.
        strategy_params: allocation strategy parameters.
        alpha: CVaR tail fraction (default 0.05 = 5%).
        n_samples: number of MC draws.
        rng: numpy random generator.

    Returns:
        CVaR at the given alpha level.
    """
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Collect CategoryScenario objects (with weighted PnLs) across all entries
    all_entry_scenarios: list[dict[str, CategoryScenario]] = []

    for year in YEARS:
        entry_data = entry_pnl_by_year[year]
        scen_data = scenario_pnl_by_year[year]

        entry_mc = entry_data[
            (entry_data["model_type"] == model) & (entry_data["config_label"] == config_label)
        ]
        scen_mc = scen_data[
            (scen_data["model_type"] == model) & (scen_data["config_label"] == config_label)
        ]

        for entry_snap, entry_group in entry_mc.groupby("entry_snapshot"):
            entry_group = entry_group.reset_index(drop=True)

            # Compute allocation weights for this entry group
            ga = _precompute_group_arrays(entry_group)
            weights = _compute_weights_numpy(
                ga["n"], ga["pnl"], ga["active"], ga["signals"], **strategy_params
            )
            # Map category → weight
            cat_weight = dict(zip(entry_group["category"].values, weights, strict=True))

            # Build CategoryScenario per category from scenario_pnl
            scen_entry = scen_mc[scen_mc["entry_snapshot"] == entry_snap]
            cat_scenarios: dict[str, CategoryScenario] = {}

            for cat, cat_scen in scen_entry.groupby("category"):
                cat_str = str(cat)
                w = cat_weight.get(cat_str, 1.0)

                nominees = cat_scen["nominee"].to_numpy()
                pnls = cat_scen["pnl"].to_numpy(dtype=np.float64) * w

                # Blend probabilities: (model_prob + market_prob) / 2, then normalize
                blend = (
                    cat_scen["model_prob"].to_numpy(dtype=np.float64)
                    + cat_scen["market_prob"].to_numpy(dtype=np.float64)
                ) / 2.0
                prob_sum = blend.sum()
                probs = blend / prob_sum if prob_sum > 0 else np.ones(len(blend)) / len(blend)

                cat_scenarios[cat_str] = CategoryScenario(winners=nominees, probs=probs, pnls=pnls)

            if cat_scenarios:
                all_entry_scenarios.append(cat_scenarios)

    if not all_entry_scenarios:
        return 0.0

    # MC simulation: sample across all entry-snapshot groups and sum
    total_pnls = np.zeros(n_samples)
    for cat_scenarios in all_entry_scenarios:
        total_pnls += sample_portfolio_pnl(cat_scenarios, n_samples, rng)

    return compute_cvar(total_pnls, alpha)


def analyze_pareto_frontier(
    data_by_year: dict[int, pd.DataFrame],
    combined: pd.DataFrame,
    scenario_pnl_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Section D: Pareto frontier of (PnL, CVaR) for avg_ensemble.

    For a representative subset of strategies, compute MC CVaR-5% for
    each (config, strategy) pair. Plot the Pareto frontier.

    Returns:
        DataFrame with columns: strategy, config_label, buy_edge_threshold,
        kelly_fraction, pnl_combined, cvar_5pct, is_pareto.
    """
    _section("D: Pareto Frontier — PnL vs MC CVaR")

    model = REC_MODEL
    ens_combined = combined[
        (combined["model_type"] == model) & (combined["strategy"].isin(PARETO_STRATEGIES))
    ].copy()

    print(
        f"Computing MC CVaR for {len(ens_combined)} (config × strategy) pairs "
        f"({N_MC_SAMPLES:,} MC samples each)..."
    )

    all_strats = get_all_strategies()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    pareto_rows: list[dict] = []
    for _, row in ens_combined.iterrows():
        strat_name = row["strategy"]
        config = row["config_label"]
        params = all_strats[strat_name]

        cvar = mc_cvar(
            scenario_pnl_by_year,
            data_by_year,
            model,
            config,
            params,
            n_samples=N_MC_SAMPLES,
            rng=rng,
        )
        pareto_rows.append(
            {
                "strategy": strat_name,
                "config_label": config,
                "buy_edge_threshold": row["buy_edge_threshold"],
                "kelly_fraction": row["kelly_fraction"],
                "pnl_combined": row["pnl_combined"],
                "cvar_5pct": cvar,
            }
        )

    pareto_df = pd.DataFrame(pareto_rows)

    # ── Identify Pareto-optimal points ──
    # A point is Pareto-optimal if no other point has both higher PnL and higher CVaR.
    # (Higher CVaR = less downside = better.)
    is_pareto = np.ones(len(pareto_df), dtype=bool)
    pnl_vals = pareto_df["pnl_combined"].values
    cvar_vals = pareto_df["cvar_5pct"].values

    for i in range(len(pareto_df)):
        for j in range(len(pareto_df)):
            if i == j:
                continue
            # j dominates i if j has >= PnL AND >= CVaR, with at least one strict
            if (
                pnl_vals[j] >= pnl_vals[i]
                and cvar_vals[j] >= cvar_vals[i]
                and (pnl_vals[j] > pnl_vals[i] or cvar_vals[j] > cvar_vals[i])
            ):
                is_pareto[i] = False
                break

    pareto_df["is_pareto"] = is_pareto

    _subsection("D1: Pareto-Optimal Points")

    pareto_optimal = pareto_df[pareto_df["is_pareto"]].sort_values("pnl_combined", ascending=False)
    print(f"{len(pareto_optimal)} Pareto-optimal points out of {len(pareto_df)} total.\n")
    print(df_to_md(pareto_optimal, float_fmt=",.0f"))

    # ── Compare uniform vs reallocation on the frontier ──
    _subsection("D2: Uniform vs Reallocation on the Frontier")

    uniform_pts = pareto_df[pareto_df["strategy"] == "uniform"]
    realloc_pts = pareto_df[pareto_df["strategy"] != "uniform"]

    if not uniform_pts.empty:
        best_uniform = uniform_pts.nlargest(1, "pnl_combined").iloc[0]
        print(
            f"Best uniform: PnL=${best_uniform['pnl_combined']:,.0f}, "
            f"CVaR=${best_uniform['cvar_5pct']:,.0f}"
        )

        # Find reallocation points that dominate the best uniform point
        dominators = realloc_pts[
            (realloc_pts["pnl_combined"] >= best_uniform["pnl_combined"])
            & (realloc_pts["cvar_5pct"] >= best_uniform["cvar_5pct"])
        ]
        if not dominators.empty:
            print(f"\n{len(dominators)} reallocation points dominate best uniform:\n")
            print(
                df_to_md(dominators.sort_values("pnl_combined", ascending=False), float_fmt=",.0f")
            )
        else:
            # Find iso-risk improvement (same CVaR range, better PnL)
            cvar_tol = abs(best_uniform["cvar_5pct"]) * 0.1  # 10% tolerance
            iso_risk = realloc_pts[
                (realloc_pts["cvar_5pct"] >= best_uniform["cvar_5pct"] - cvar_tol)
                & (realloc_pts["pnl_combined"] > best_uniform["pnl_combined"])
            ]
            if not iso_risk.empty:
                print(f"\n{len(iso_risk)} points improve PnL at similar risk (±10% CVaR):\n")
                print(df_to_md(iso_risk.nlargest(5, "pnl_combined"), float_fmt=",.0f"))
            else:
                print("\nNo reallocation points dominate or match best uniform's risk.")

    # ── Compute avg_rank for Pareto points ──
    _subsection("D3: Pareto — Avg Rank vs CVaR")

    avg_rank_by_strategy: dict[str, float] = {}
    for _, prow in pareto_df[pareto_df["is_pareto"]].iterrows():
        strat = prow["strategy"]
        cfg = prow["config_label"]
        key = f"{strat}|{cfg}"
        # Get this (strategy, config) row from combined to access per-year PnLs
        mask = (
            (combined["model_type"] == model)
            & (combined["strategy"] == strat)
            & (combined["config_label"] == cfg)
        )
        match = combined[mask]
        if match.empty:
            continue
        # Rank strategies within each year at this config
        cfg_data = combined[(combined["model_type"] == model) & (combined["config_label"] == cfg)]
        ranks = []
        for yr_col in ["pnl_2024", "pnl_2025"]:
            ranked = cfg_data[yr_col].rank(ascending=False)
            idx = cfg_data.index[cfg_data["strategy"] == strat]
            if len(idx) > 0:
                ranks.append(float(ranked.loc[idx[0]]))
        if ranks:
            avg_rank_by_strategy[key] = sum(ranks) / len(ranks)

    # Attach avg_rank to pareto_df
    pareto_df["avg_rank"] = pareto_df.apply(
        lambda r: avg_rank_by_strategy.get(f"{r['strategy']}|{r['config_label']}", float("nan")),
        axis=1,
    )

    pareto_rank_pts = pareto_df[pareto_df["is_pareto"]].dropna(subset=["avg_rank"])
    if not pareto_rank_pts.empty:
        print(
            df_to_md(
                pareto_rank_pts[
                    ["strategy", "config_label", "pnl_combined", "cvar_5pct", "avg_rank"]
                ].sort_values("avg_rank"),
                float_fmt=",.1f",
            )
        )

    _plot_pareto_frontier(pareto_df)
    _plot_pareto_avg_rank(pareto_df)

    return pareto_df


def _plot_pareto_frontier(pareto_df: pd.DataFrame) -> None:
    """Plot the Pareto frontier: PnL vs CVaR, color-coded by strategy type."""
    plot_dir = ensure_plot_dir("joint")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color map by strategy type
    colors = {
        "uniform": "#E74C3C",
        "equal_active": "#F39C12",
        "ev_100": "#2ECC71",
        "ev_50": "#27AE60",
        "ev_cap30": "#1ABC9C",
        "edge_100": "#3498DB",
        "edge_50": "#2980B9",
        "capital_100": "#9B59B6",
        "maxedge_100": "#E67E22",
        "npos_100": "#95A5A6",
    }

    for strat_name, group in pareto_df.groupby("strategy"):
        color = colors.get(str(strat_name), "#7F8C8D")
        label = str(strat_name)

        # Non-Pareto points: small, transparent
        non_pareto = group[~group["is_pareto"]]
        if not non_pareto.empty:
            ax.scatter(
                non_pareto["cvar_5pct"],
                non_pareto["pnl_combined"],
                color=color,
                alpha=0.25,
                s=20,
                label=None,
            )

        # Pareto points: large, opaque, with label
        pareto_pts = group[group["is_pareto"]]
        if not pareto_pts.empty:
            ax.scatter(
                pareto_pts["cvar_5pct"],
                pareto_pts["pnl_combined"],
                color=color,
                alpha=0.9,
                s=80,
                edgecolors="black",
                linewidth=0.8,
                label=f"{label} (Pareto)",
            )
        else:
            # Just show the label for the strategy
            ax.scatter([], [], color=color, alpha=0.5, s=30, label=label)

    # Draw Pareto frontier line
    pareto_optimal = pareto_df[pareto_df["is_pareto"]].sort_values("cvar_5pct")
    if len(pareto_optimal) > 1:
        ax.plot(
            pareto_optimal["cvar_5pct"],
            pareto_optimal["pnl_combined"],
            "k--",
            alpha=0.4,
            linewidth=1,
            label="Pareto frontier",
        )

    ax.set_xlabel("MC CVaR-5% ($)")
    ax.set_ylabel("Combined PnL ($)")
    ax.set_title("Pareto Frontier: PnL vs MC CVaR Downside Risk (avg_ensemble)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_frontier.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {plot_dir / 'pareto_frontier.png'}")


def _plot_pareto_avg_rank(pareto_df: pd.DataFrame) -> None:
    """Plot Pareto points: avg_rank (x, lower=better) vs CVaR-5% (y, higher=better)."""
    plot_dir = ensure_plot_dir("joint")

    pts = pareto_df[pareto_df["is_pareto"]].dropna(subset=["avg_rank"]).copy()
    if pts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "uniform": "#E74C3C",
        "equal_active": "#F39C12",
        "ev_100": "#2ECC71",
        "ev_50": "#27AE60",
        "ev_cap30": "#1ABC9C",
        "edge_100": "#3498DB",
        "edge_50": "#2980B9",
        "capital_100": "#9B59B6",
        "maxedge_100": "#E67E22",
        "npos_100": "#95A5A6",
    }

    for _, row in pts.iterrows():
        strat = str(row["strategy"])
        color = colors.get(strat, "#7F8C8D")
        ax.scatter(
            row["avg_rank"],
            row["cvar_5pct"],
            color=color,
            s=80,
            edgecolors="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=3,
        )
        label_text = f"{strat}\n{row['config_label']}"
        ax.annotate(
            label_text,
            (row["avg_rank"], row["cvar_5pct"]),
            fontsize=6,
            alpha=0.8,
            xytext=(6, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Avg Rank (lower = better)")
    ax.set_ylabel("MC CVaR-5% ($) (higher = better)")
    ax.set_title("Pareto Points: Avg Rank vs MC CVaR-5% (avg_ensemble)")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # lower rank = better → leftward

    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_avg_rank_vs_cvar.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_dir / 'pareto_avg_rank_vs_cvar.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Configuration Recommendation
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_recommendation(
    combined: pd.DataFrame,
    model_df: pd.DataFrame,
) -> None:
    """Section E: synthesize the joint analysis into a recommendation.

    Compare:
    1. Config sweep baseline: avg_ensemble, edge=0.20, KF=0.05, uniform
    2. Best (config, allocation) for avg_ensemble (joint optimization)
    3. Best overall triple across all models
    """
    _section("E: Configuration Recommendation")

    # ── Baseline: config sweep recommendation ──
    _subsection("E1: Comparison of Approaches")

    rows: list[dict] = []

    # 1. Config sweep baseline
    baseline = combined[
        (combined["model_type"] == REC_MODEL)
        & (combined["buy_edge_threshold"] == REC_EDGE)
        & (combined["kelly_fraction"] == REC_KF)
        & (combined["strategy"] == "uniform")
    ]
    if not baseline.empty:
        b = baseline.iloc[0]
        rows.append(
            {
                "approach": "Config sweep baseline",
                "model": short_model(REC_MODEL),
                "edge": REC_EDGE,
                "kf": REC_KF,
                "strategy": "uniform",
                "pnl_2024": b["pnl_2024"],
                "pnl_2025": b["pnl_2025"],
                "pnl_combined": b["pnl_combined"],
            }
        )

    # 2. Best (config, allocation) for avg_ensemble (non-oracle)
    ens_non_oracle = combined[
        (combined["model_type"] == REC_MODEL) & (~combined["strategy"].isin(["oracle"]))
    ]
    if not ens_non_oracle.empty:
        best_ens = ens_non_oracle.nlargest(1, "pnl_combined").iloc[0]
        rows.append(
            {
                "approach": "Best joint (avg_ens)",
                "model": short_model(REC_MODEL),
                "edge": best_ens["buy_edge_threshold"],
                "kf": best_ens["kelly_fraction"],
                "strategy": best_ens["strategy"],
                "pnl_2024": best_ens["pnl_2024"],
                "pnl_2025": best_ens["pnl_2025"],
                "pnl_combined": best_ens["pnl_combined"],
            }
        )

    # 3. Best overall triple (non-oracle)
    all_non_oracle = combined[~combined["strategy"].isin(["oracle"])]
    if not all_non_oracle.empty:
        best_all = all_non_oracle.nlargest(1, "pnl_combined").iloc[0]
        rows.append(
            {
                "approach": "Best overall triple",
                "model": short_model(best_all["model_type"]),
                "edge": best_all["buy_edge_threshold"],
                "kf": best_all["kelly_fraction"],
                "strategy": best_all["strategy"],
                "pnl_2024": best_all["pnl_2024"],
                "pnl_2025": best_all["pnl_2025"],
                "pnl_combined": best_all["pnl_combined"],
            }
        )

    rec_df = pd.DataFrame(rows)
    print(df_to_md(rec_df, float_fmt=",.1f"))

    # ── Separability verdict ──
    _subsection("E2: Separability Verdict")

    if len(rows) >= 2:
        baseline_pnl = rows[0]["pnl_combined"]
        joint_pnl = rows[1]["pnl_combined"]
        uplift = joint_pnl - baseline_pnl
        pct = (uplift / baseline_pnl * 100) if baseline_pnl != 0 else float("nan")

        print(f"Joint optimization uplift over baseline: ${uplift:,.0f} ({pct:+.1f}%)")

        baseline_edge = rows[0]["edge"]
        joint_edge = rows[1]["edge"]
        baseline_kf = rows[0]["kf"]
        joint_kf = rows[1]["kf"]

        if baseline_edge == joint_edge and baseline_kf == joint_kf:
            print(
                "\nConfig is UNCHANGED under joint optimization → "
                "config and allocation are **separable**."
            )
            print("Implication: pick config under uniform, then layer allocation on top.")
        else:
            print(
                f"\nConfig SHIFTED: edge {baseline_edge}→{joint_edge}, KF {baseline_kf}→{joint_kf}"
            )
            print(
                "Config and allocation are **entangled** — "
                "the two-step approach leaves money on the table."
            )

    # ── Model ranking stability ──
    _subsection("E3: Model Ranking Under Reallocation")

    if model_df is not None and not model_df.empty:
        rank_changes = (model_df["rank_change"] != 0).sum()
        if rank_changes == 0:
            print(
                "Model ranking is UNCHANGED under reallocation → "
                "model choice is **independent** of allocation."
            )
        else:
            print(
                f"{rank_changes}/{len(model_df)} models changed rank → "
                "model choice and allocation are partially entangled."
            )

        # Show top model under each regime
        top_uniform = model_df.loc[model_df["uniform_pnl"].idxmax()]
        top_joint = model_df.loc[model_df["joint_pnl"].idxmax()]
        print(
            f"\nTop model (uniform): {top_uniform['model_short']} "
            f"(${top_uniform['uniform_pnl']:,.0f})"
        )
        print(f"Top model (joint):   {top_joint['model_short']} (${top_joint['joint_pnl']:,.0f})")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV output
# ═══════════════════════════════════════════════════════════════════════════════


def save_outputs(
    combined: pd.DataFrame,
    edge_interaction: pd.DataFrame,
    model_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
) -> None:
    """Save analysis results to CSV files."""
    out_dir = ensure_output_dir()

    combined.to_csv(out_dir / "joint_sweep.csv", index=False)
    print(f"\nSaved: {out_dir / 'joint_sweep.csv'} ({len(combined):,} rows)")

    edge_interaction.to_csv(out_dir / "edge_interaction.csv", index=False)
    print(f"Saved: {out_dir / 'edge_interaction.csv'} ({len(edge_interaction):,} rows)")

    model_df.to_csv(out_dir / "model_interaction_joint.csv", index=False)
    print(f"Saved: {out_dir / 'model_interaction_joint.csv'} ({len(model_df):,} rows)")

    pareto_df.to_csv(out_dir / "pareto_frontier.csv", index=False)
    print(f"Saved: {out_dir / 'pareto_frontier.csv'} ({len(pareto_df):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("=" * 80)
    print("Analysis 4: Joint Optimization over (Config × Allocation × Model)")
    print("=" * 80)

    # ── Load data ──
    print("\nLoading data...")
    data_by_year = {year: prepare_data(year) for year in YEARS}
    for year, df in data_by_year.items():
        n_models = df["model_type"].nunique()
        n_configs = df["config_label"].nunique()
        n_entries = df["entry_snapshot"].nunique()
        n_cats = df["category"].nunique()
        print(
            f"  {year}: {n_models} models × {n_configs} configs × "
            f"{n_entries} entries × {n_cats} categories = {len(df):,} rows"
        )

    # ── Section A: Full sweep ──
    combined = run_full_sweep(data_by_year)
    analyze_full_sweep(combined)

    # ── Section B: Edge threshold interaction ──
    edge_interaction = analyze_edge_interaction(combined)

    # ── Section C: Model interaction ──
    model_df = analyze_model_interaction(combined)

    # ── Load scenario_pnl for MC CVaR ──
    print("\nLoading scenario_pnl for MC CVaR...")
    all_raw_data = load_all_data()
    scenario_pnl_by_year = {year: spnl for year, (_, spnl) in all_raw_data.items()}

    # ── Section D: Pareto frontier ──
    pareto_df = analyze_pareto_frontier(data_by_year, combined, scenario_pnl_by_year)

    # ── Section E: Recommendation ──
    analyze_recommendation(combined, model_df)

    # ── Save outputs ──
    save_outputs(combined, edge_interaction, model_df, pareto_df)

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
