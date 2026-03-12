"""Shared utilities for category allocation analysis.

Provides data loading, signal computation, allocation strategies, and portfolio
evaluation for the category allocation investigation. All analysis scripts
import from this module.

Data source: storage/d20260305_config_selection_sweep/ (targeted grid, post-refactor)
- 27 configs (9 edge × 3 KF) × 6 models = 162 model×config combos per year
- Fixed: fee=taker, kelly_mode=multi_outcome, allowed_directions=all
- 2024: 8 categories, 7 entry snapshots
- 2025: 9 categories, 9 entry snapshots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ─── Constants ───────────────────────────────────────────────────────────────

EXP_DIR = Path("storage/d20260305_config_selection_sweep")
OUTPUT_DIR = Path("storage/d20260305_portfolio_kelly")
BANKROLL = 1000.0
YEARS = [2024, 2025]

ALL_MODELS = [
    "avg_ensemble",
    "cal_sgbt",
    "clogit_cal_sgbt_ensemble",
    "clogit",
    "lr",
    "gbt",
]

MODEL_SHORT: dict[str, str] = {
    "avg_ensemble": "avg_ens",
    "cal_sgbt": "cal_sgbt",
    "clogit_cal_sgbt_ensemble": "clog_sgbt",
    "clogit": "clogit",
    "lr": "lr",
    "gbt": "gbt",
}

# Columns that uniquely identify a portfolio evaluation context
# (one row per category within this group = one portfolio decision)
GROUP_COLS = ["model_type", "config_label", "entry_snapshot"]

# Signal columns available for allocation
SIGNAL_NAMES = [
    "ev_pnl_blend",
    "capital_deployed",
    "n_positions",
    "mean_edge",
    "max_edge",
    "max_abs_edge",
]


# ─── Data loading ────────────────────────────────────────────────────────────


def load_entry_pnl(year: int) -> pd.DataFrame:
    """Load entry-level P&L from config selection sweep data.

    Returns one row per (category, model_type, entry_snapshot, config_label).
    Filtered to bankroll_mode='fixed' only.
    """
    path = EXP_DIR / str(year) / "results" / "entry_pnl.csv"
    df = pd.read_csv(path)
    return df[df["bankroll_mode"] == "fixed"].copy()


def load_scenario_pnl(year: int) -> pd.DataFrame:
    """Load scenario-level P&L (per-nominee) from config selection sweep data."""
    path = EXP_DIR / str(year) / "results" / "scenario_pnl.csv"
    return pd.read_csv(path)


def load_all_data() -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load entry_pnl and scenario_pnl for both years.

    Returns:
        {year: (entry_pnl_df, scenario_pnl_df)} for each year.
    """
    return {year: (load_entry_pnl(year), load_scenario_pnl(year)) for year in YEARS}


# ─── Signal computation ─────────────────────────────────────────────────────


def add_scenario_signals(
    entry_pnl: pd.DataFrame,
    scenario_pnl: pd.DataFrame,
) -> pd.DataFrame:
    """Add edge-based signals to entry_pnl.

    For each (category, model_type, entry_snapshot, config_label):
    - mean_edge: mean |model_prob - market_prob| of traded nominees (pnl != 0)
    - max_edge: max(model_prob - market_prob) across ALL nominees in category
      (YES-side only — the historical signal used in prior analyses)
    - max_abs_edge: max(|model_prob - market_prob|) across ALL nominees in
      category. Equivalently, the best raw edge in either YES or NO direction.
      A category with a strong NO opportunity (overpriced nominee) scores
      high on max_abs_edge even if max_edge is modest.
    - is_active: whether any position was taken (n_positions > 0)

    Categories with no trades get mean_edge=0.
    """
    join_cols = ["category", "model_type", "entry_snapshot", "config_label"]

    scen = scenario_pnl.copy()
    scen["edge"] = scen["model_prob"] - scen["market_prob"]
    scen["abs_edge"] = scen["edge"].abs()

    # Max edge across all nominees in category (YES-side buy opportunity)
    max_edge = (
        scen.groupby(join_cols)["edge"].max().reset_index().rename(columns={"edge": "max_edge"})
    )

    # Max absolute edge (best opportunity in either direction)
    max_abs_edge = (
        scen.groupby(join_cols)["abs_edge"]
        .max()
        .reset_index()
        .rename(columns={"abs_edge": "max_abs_edge"})
    )

    # Mean absolute edge of traded nominees (quality of executed trades)
    traded = scen[scen["pnl"] != 0]
    mean_edge = (
        traded.groupby(join_cols)["abs_edge"]
        .mean()
        .reset_index()
        .rename(columns={"abs_edge": "mean_edge"})
    )

    result = entry_pnl.merge(max_edge, on=join_cols, how="left")
    result = result.merge(max_abs_edge, on=join_cols, how="left")
    result = result.merge(mean_edge, on=join_cols, how="left")
    result["max_edge"] = result["max_edge"].fillna(0)
    result["max_abs_edge"] = result["max_abs_edge"].fillna(0)
    result["mean_edge"] = result["mean_edge"].fillna(0)
    result["is_active"] = result["n_positions"] > 0

    return result


def prepare_data(year: int) -> pd.DataFrame:
    """Load and prepare entry_pnl with all signals for a year.

    Convenience function that chains load + add_scenario_signals.
    """
    entry_pnl = load_entry_pnl(year)
    scenario_pnl = load_scenario_pnl(year)
    return add_scenario_signals(entry_pnl, scenario_pnl)


# ─── Allocation weight computation ──────────────────────────────────────────


def compute_weights(
    cat_data: pd.DataFrame,
    strategy: str,
    aggressiveness: float = 1.0,
    cap: float | None = None,
) -> pd.Series:
    """Compute per-category allocation weights for one portfolio decision.

    Args:
        cat_data: DataFrame with one row per category (from groupby on GROUP_COLS).
            Must have 'category', 'is_active', and signal columns.
        strategy: "uniform", "equal_active", or a signal column name from SIGNAL_NAMES.
        aggressiveness: blend factor. 0.0 = uniform, 1.0 = full strategy signal.
        cap: max fraction of total bankroll per category (e.g. 0.3 = max 30%).

    Returns:
        Series indexed like cat_data with weight per category.
        Weights are normalized so sum = n_categories (preserving total bankroll).
        A weight of 1.0 = uniform allocation ($BANKROLL per category).
        A weight of 2.0 = double bankroll for that category.
    """
    n = len(cat_data)
    idx = cat_data.index

    if strategy == "uniform":
        raw = pd.Series(1.0, index=idx)
    elif strategy == "equal_active":
        active_mask = cat_data["is_active"].values
        n_active = active_mask.sum()
        if n_active == 0:
            raw = pd.Series(1.0, index=idx)
        else:
            raw = pd.Series(0.0, index=idx)
            raw.iloc[active_mask] = n / n_active
    elif strategy == "oracle":
        # Hindsight-optimal: proportional to max(actual_pnl, 0)
        pnl_vals = cat_data["total_pnl"].clip(lower=0).values
        total = pnl_vals.sum()
        if total == 0:
            raw = pd.Series(1.0, index=idx)
        else:
            raw = pd.Series(pnl_vals / total * n, index=idx)
    else:
        # Signal-proportional, restricted to active categories
        active_mask = cat_data["is_active"].values
        signal_vals = np.maximum(cat_data[strategy].values, 0) * active_mask.astype(float)
        total_signal = signal_vals.sum()
        if total_signal == 0:
            # No signal → fall back to equal-among-active
            n_active = active_mask.sum()
            if n_active == 0:
                raw = pd.Series(1.0, index=idx)
            else:
                raw = pd.Series(0.0, index=idx)
                raw.iloc[active_mask] = n / n_active
        else:
            raw = pd.Series(signal_vals / total_signal * n, index=idx)

    # Blend toward uniform
    weights = (1.0 - aggressiveness) * 1.0 + aggressiveness * raw

    # Apply concentration cap (iterative redistribution)
    if cap is not None:
        max_w = cap * n
        w_arr = weights.values.copy()  # operate on numpy to avoid pandas index issues
        for _ in range(20):
            over_mask = w_arr > max_w
            if not over_mask.any():
                break
            excess = (w_arr[over_mask] - max_w).sum()
            w_arr[over_mask] = max_w
            n_under = (~over_mask).sum()
            if n_under == 0:
                break
            w_arr[~over_mask] += excess / n_under
        weights = pd.Series(w_arr, index=weights.index)

    return weights


# ─── Strategy registry ───────────────────────────────────────────────────────


def get_all_strategies() -> dict[str, dict[str, Any]]:
    """Return all allocation strategies to test.

    Each strategy is a dict with keys: strategy, aggressiveness, cap.
    Strategy name format: {signal}_{aggressiveness} or {signal}_{cap}.
    """
    strategies: dict[str, dict[str, Any]] = {
        "uniform": {"strategy": "uniform", "aggressiveness": 1.0, "cap": None},
        "equal_active": {"strategy": "equal_active", "aggressiveness": 1.0, "cap": None},
        "oracle": {"strategy": "oracle", "aggressiveness": 1.0, "cap": None},
    }

    # Signal-proportional variants
    signal_configs = [
        ("ev", "ev_pnl_blend"),
        ("edge", "mean_edge"),
        ("capital", "capital_deployed"),
        ("npos", "n_positions"),
        ("maxedge", "max_edge"),
        ("maxabsedge", "max_abs_edge"),
    ]

    for prefix, signal in signal_configs:
        # Aggressiveness blends
        for agg, label in [(1.0, "100"), (0.75, "75"), (0.50, "50"), (0.25, "25")]:
            strategies[f"{prefix}_{label}"] = {
                "strategy": signal,
                "aggressiveness": agg,
                "cap": None,
            }
        # Capped variants (full aggressiveness)
        for cap_val, cap_label in [(0.30, "cap30"), (0.50, "cap50")]:
            strategies[f"{prefix}_{cap_label}"] = {
                "strategy": signal,
                "aggressiveness": 1.0,
                "cap": cap_val,
            }

    return strategies


def get_prospective_strategies() -> dict[str, dict[str, Any]]:
    """Return only prospective (non-oracle) strategies."""
    all_strats = get_all_strategies()
    return {k: v for k, v in all_strats.items() if k != "oracle"}


# ─── Portfolio evaluation ────────────────────────────────────────────────────


def evaluate_portfolio_at_entry(
    entry_group: pd.DataFrame,
    strategy_params: dict[str, Any],
) -> float:
    """Compute portfolio PnL for one (model, config, entry_snapshot).

    Args:
        entry_group: DataFrame with one row per category.
        strategy_params: dict with keys: strategy, aggressiveness, cap.

    Returns:
        Weighted portfolio PnL = sum(cat_pnl × weight).
    """
    weights = compute_weights(entry_group, **strategy_params)
    return (entry_group["total_pnl"] * weights).sum()


def evaluate_strategy_across_entries(
    entry_pnl: pd.DataFrame,
    strategy_params: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate one allocation strategy across all (model, config, entry) combos.

    Returns DataFrame with columns:
        model_type, config_label, entry_snapshot, buy_edge_threshold,
        kelly_fraction, portfolio_pnl
    """
    results = []

    for (model, config, entry), group in entry_pnl.groupby(GROUP_COLS):
        pnl = evaluate_portfolio_at_entry(group, strategy_params)
        row0 = group.iloc[0]
        results.append(
            {
                "model_type": model,
                "config_label": config,
                "entry_snapshot": entry,
                "buy_edge_threshold": row0["buy_edge_threshold"],
                "kelly_fraction": row0["kelly_fraction"],
                "portfolio_pnl": pnl,
            }
        )

    return pd.DataFrame(results)


def evaluate_all_strategies_by_year(
    data_by_year: dict[int, pd.DataFrame],
    strategies: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Evaluate all strategies across all years.

    Args:
        data_by_year: {year: prepared_entry_pnl} from prepare_data().
        strategies: strategy dict (defaults to get_all_strategies()).

    Returns:
        Long-form DataFrame with columns:
            year, strategy, model_type, config_label, entry_snapshot,
            buy_edge_threshold, kelly_fraction, portfolio_pnl
    """
    if strategies is None:
        strategies = get_all_strategies()

    all_results = []
    for year, entry_pnl in data_by_year.items():
        for strat_name, params in strategies.items():
            result = evaluate_strategy_across_entries(entry_pnl, params)
            result["year"] = year
            result["strategy"] = strat_name
            all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


# ─── Aggregation helpers ─────────────────────────────────────────────────────


def aggregate_to_year_level(results: pd.DataFrame) -> pd.DataFrame:
    """Sum entry-level portfolio PnL to (year, strategy, model, config) level.

    Input: output of evaluate_all_strategies_by_year().
    Output: one row per (year, strategy, model_type, config_label).
    """
    return (
        results.groupby(
            [
                "year",
                "strategy",
                "model_type",
                "config_label",
                "buy_edge_threshold",
                "kelly_fraction",
            ]
        )
        .agg(
            portfolio_pnl=("portfolio_pnl", "sum"),
            n_entries=("entry_snapshot", "nunique"),
        )
        .reset_index()
    )


def aggregate_combined(year_level: pd.DataFrame) -> pd.DataFrame:
    """Compute combined (2024+2025) portfolio PnL per (strategy, model, config).

    Input: output of aggregate_to_year_level().
    Output: one row per (strategy, model_type, config_label) with per-year and combined PnL.
    """
    pivot = year_level.pivot_table(
        index=["strategy", "model_type", "config_label", "buy_edge_threshold", "kelly_fraction"],
        columns="year",
        values="portfolio_pnl",
        aggfunc="sum",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(columns={2024: "pnl_2024", 2025: "pnl_2025"})
    pivot["pnl_combined"] = pivot["pnl_2024"].fillna(0) + pivot["pnl_2025"].fillna(0)
    pivot["year_balance"] = pivot["pnl_2024"] / pivot["pnl_2025"]

    return pivot


# ─── Uniform baseline helper ────────────────────────────────────────────────


def compute_uniform_baseline(data_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute uniform-allocation portfolio PnL for all (model, config, entry, year).

    Returns same schema as evaluate_all_strategies_by_year but with strategy='uniform'.
    """
    uniform_params = {"strategy": "uniform", "aggressiveness": 1.0, "cap": None}
    return evaluate_all_strategies_by_year(data_by_year, {"uniform": uniform_params})


# ─── Noise injection ────────────────────────────────────────────────────────


def add_noise_to_signal(
    entry_pnl: pd.DataFrame,
    signal_col: str,
    noise_sigma: float,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return a copy of entry_pnl with multiplicative lognormal noise on a signal.

    noise_sigma=0.5 means the signal is multiplied by exp(N(0, 0.5)),
    roughly doubling/halving individual values.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = entry_pnl.copy()
    noise = np.exp(rng.normal(0, noise_sigma, size=len(result)))
    result[signal_col] = result[signal_col] * noise
    return result


# ─── Display helpers ─────────────────────────────────────────────────────────


def short_model(model: str) -> str:
    """Shorten model name for display."""
    return MODEL_SHORT.get(model, model)


def df_to_md(df: pd.DataFrame, float_fmt: str = ",.0f") -> str:
    """Convert DataFrame to markdown table string."""
    return df.to_markdown(index=False, floatfmt=float_fmt)


def ensure_output_dir() -> Path:
    """Create and return the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def ensure_plot_dir(subdir: str = "") -> Path:
    """Create and return a plot subdirectory."""
    plot_dir = OUTPUT_DIR / "plots"
    if subdir:
        plot_dir = plot_dir / subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir
