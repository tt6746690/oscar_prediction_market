"""Scenario-based scoring for buy-and-hold backtest config selection.

Unified CVaR framework for risk-constrained portfolio selection.  Every risk
measure is expressed as CVaR at some tail fraction α:

- **α = 0%** — deterministic worst case (no MC needed): ``cvar_0 = worst_pnl``
- **α = 5%, 10%, 25%** — Monte Carlo CVaR at increasing tail fractions

The optimization criterion is:

    max  EV(PnL)  s.t.  CVaR_α(PnL) >= -L * bankroll

where L is a loss bound fraction and α selects the risk measure.

Aggregation levels:
1. Entry-level: per (category, entry_point, config) — raw from entry_pnl.csv
2. Category-level: sum across entry points per (category, config)
3. Portfolio-level: sum across categories per (config, year)
4. Cross-year: average EV across 2024+2025, risk must pass in BOTH years

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.scenario_scoring --ceremony-year 2025
    # Or cross-year:
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.scenario_scoring --cross-year
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.portfolio_simulation import (
    CategoryScenario,
    compute_cvar,
    sample_portfolio_pnl,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    BUY_HOLD_EXP_DIR,
    YEAR_CONFIGS,
)

BANKROLL = 1000.0

#: Config columns used for grouping throughout the scoring pipeline.
CONFIG_COLS = [
    "model_type",
    "config_label",
    "fee_type",
    "kelly_fraction",
    "buy_edge_threshold",
    "kelly_mode",
    "bankroll_mode",
    "allowed_directions",
]

#: Default loss bound fractions for Pareto frontier sweep.
DEFAULT_LOSS_BOUNDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

#: CVaR tail fractions. α=0.0 is worst-case (deterministic, no MC needed).
CVAR_ALPHAS = [0.0, 0.05, 0.10, 0.25]

#: Standardized loss bounds for cross-year Pareto tables.
CROSS_YEAR_LOSS_BOUNDS = [0.10, 0.20, 0.30, 0.40, 0.50]


# ============================================================================
# Data loading
# ============================================================================


def load_entry_pnl(ceremony_year: int, *, exp_dir: Path | None = None) -> pd.DataFrame:
    """Load entry-level P&L with scenario columns from a year's results."""
    if exp_dir is not None:
        path = exp_dir / str(ceremony_year) / "results" / "entry_pnl.csv"
    else:
        cfg = YEAR_CONFIGS[ceremony_year]
        path = cfg.results_dir / "entry_pnl.csv"
    df = pd.read_csv(path)
    required = {"worst_pnl", "best_pnl", "ev_pnl_model", "ev_pnl_market", "ev_pnl_blend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"entry_pnl.csv for {ceremony_year} missing scenario columns: {missing}. "
            "Re-run backtests with updated run_backtests.py."
        )
    return df


def load_scenario_pnl(ceremony_year: int, *, exp_dir: Path | None = None) -> pd.DataFrame:
    """Load per-nominee scenario PnL from a year's results.

    Each row represents one (category, entry_snapshot, config, nominee) with
    the PnL if that nominee wins, plus model and market probabilities.
    Used for Monte Carlo CVaR computation.
    """
    if exp_dir is not None:
        path = exp_dir / str(ceremony_year) / "results" / "scenario_pnl.csv"
    else:
        cfg = YEAR_CONFIGS[ceremony_year]
        path = cfg.results_dir / "scenario_pnl.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"scenario_pnl.csv not found for {ceremony_year} at {path}. "
            "Re-run backtests with updated run_backtests.py."
        )
    return pd.read_csv(path)


# ============================================================================
# Portfolio-level aggregation
# ============================================================================


def compute_portfolio_scores(entry_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entry-level scenarios to portfolio level via per-entry normalization.

    Uses per-entry-time normalization:
    1. Sum across categories per entry time -> per-entry portfolio
    2. Average across entry times -> config-level score

    This represents "expected single-entry deployment performance" — the
    typical PnL from entering at one point in the season and deploying
    across all categories at that moment.

    Per-entry bankroll = N_categories * BANKROLL.  deployment_rate and
    return_pct are expressed relative to this per-entry bankroll.

    Returns one row per config with:
    - total_pnl, worst_pnl, best_pnl, ev_pnl_model/market/blend (per-entry averages)
    - capital_deployed, total_trades, total_fees (per-entry averages)
    - n_categories, n_entries
    - deployment_rate = capital_deployed / per_entry_bankroll
    - return_pct = total_pnl / per_entry_bankroll
    """
    n_categories = entry_df["category"].nunique()
    n_entries = entry_df["entry_snapshot"].nunique()

    # Step 1: Sum across categories per entry time -> per-entry portfolio
    per_entry = entry_df.groupby(["entry_snapshot"] + CONFIG_COLS, as_index=False).agg(
        total_pnl=("total_pnl", "sum"),
        worst_pnl=("worst_pnl", "sum"),
        best_pnl=("best_pnl", "sum"),
        ev_pnl_model=("ev_pnl_model", "sum"),
        ev_pnl_market=("ev_pnl_market", "sum"),
        ev_pnl_blend=("ev_pnl_blend", "sum"),
        capital_deployed=("capital_deployed", "sum"),
        total_trades=("total_trades", "sum"),
        total_fees=("total_fees", "sum"),
        n_categories=("category", "nunique"),
        entries_with_trades=("total_trades", lambda x: (x > 0).sum()),
        worst_category_pnl=("total_pnl", "min"),
    )

    # Step 2: Average across entry times -> config-level score
    portfolio_df = per_entry.groupby(CONFIG_COLS, as_index=False).agg(
        total_pnl=("total_pnl", "mean"),
        worst_pnl=("worst_pnl", "mean"),
        best_pnl=("best_pnl", "mean"),
        ev_pnl_model=("ev_pnl_model", "mean"),
        ev_pnl_market=("ev_pnl_market", "mean"),
        ev_pnl_blend=("ev_pnl_blend", "mean"),
        capital_deployed=("capital_deployed", "mean"),
        total_trades=("total_trades", "mean"),
        total_fees=("total_fees", "mean"),
        n_categories=("n_categories", "mean"),
        entries_with_trades=("entries_with_trades", "mean"),
        worst_category_pnl=("worst_category_pnl", "mean"),
    )

    # Per-entry bankroll: N_categories * BANKROLL (single deployment)
    per_entry_bankroll = n_categories * BANKROLL
    portfolio_df["total_bankroll_deployed"] = per_entry_bankroll
    portfolio_df["n_entries"] = n_entries
    portfolio_df["deployment_rate"] = (
        portfolio_df["capital_deployed"] / per_entry_bankroll * 100
    ).round(1)
    portfolio_df["return_pct"] = (portfolio_df["total_pnl"] / per_entry_bankroll * 100).round(1)
    portfolio_df["roic"] = np.where(
        portfolio_df["capital_deployed"] > 0,
        (portfolio_df["total_pnl"] / portfolio_df["capital_deployed"] * 100).round(1),
        0.0,
    )

    return portfolio_df.sort_values("ev_pnl_blend", ascending=False).reset_index(drop=True)


def compute_category_scores(entry_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to category level (sum across entries per category × config).

    Useful for per-category analysis and for computing portfolio worst case
    as sum of per-category worst cases.
    """
    cat_df = entry_df.groupby(["category"] + CONFIG_COLS, as_index=False).agg(
        total_pnl=("total_pnl", "sum"),
        worst_pnl=("worst_pnl", "sum"),
        best_pnl=("best_pnl", "sum"),
        ev_pnl_model=("ev_pnl_model", "sum"),
        ev_pnl_market=("ev_pnl_market", "sum"),
        ev_pnl_blend=("ev_pnl_blend", "sum"),
        capital_deployed=("capital_deployed", "sum"),
        total_trades=("total_trades", "sum"),
        total_fees=("total_fees", "sum"),
        n_entries=("entry_snapshot", "nunique"),
        entries_with_trades=("total_trades", lambda x: (x > 0).sum()),
    )
    return cat_df


def compute_temporal_scores(entry_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to entry-point level (sum across categories per entry × config).

    Shows how EV/risk evolve over the Oscar season as more precursor info arrives.
    """
    temporal_df = entry_df.groupby(["entry_snapshot"] + CONFIG_COLS, as_index=False).agg(
        total_pnl=("total_pnl", "sum"),
        worst_pnl=("worst_pnl", "sum"),
        best_pnl=("best_pnl", "sum"),
        ev_pnl_model=("ev_pnl_model", "sum"),
        ev_pnl_market=("ev_pnl_market", "sum"),
        ev_pnl_blend=("ev_pnl_blend", "sum"),
        capital_deployed=("capital_deployed", "sum"),
        total_trades=("total_trades", "sum"),
        n_categories=("category", "nunique"),
    )
    return temporal_df


# ============================================================================
# Pareto frontier
# ============================================================================


def compute_pareto_frontier(
    portfolio_df: pd.DataFrame,
    loss_bounds: list[float] | None = None,
    ev_column: str = "ev_pnl_blend",
    risk_col: str = "cvar_0",
) -> pd.DataFrame:
    """Find the Pareto-optimal config at each loss bound level.

    For each loss bound L, filters to configs where ``risk_col >= -L*B``
    (where B = total bankroll deployed), then picks the one with highest EV.

    The risk column can be any CVaR level: ``cvar_0`` (= worst_pnl,
    deterministic), ``cvar_5``, ``cvar_10``, ``cvar_25``.

    Args:
        portfolio_df: Portfolio-level scores (must contain ``risk_col``).
        loss_bounds: Loss bound fractions to sweep. Defaults to DEFAULT_LOSS_BOUNDS.
        ev_column: Which EV column to maximize (ev_pnl_model/market/blend).
        risk_col: Column used for the downside constraint.

    Returns:
        DataFrame with one row per loss bound, showing the best config and its metrics.
    """
    if loss_bounds is None:
        loss_bounds = DEFAULT_LOSS_BOUNDS

    rows: list[dict] = []
    total_bankroll = portfolio_df["total_bankroll_deployed"].iloc[0]

    for L in loss_bounds:
        max_loss = L * total_bankroll
        feasible = portfolio_df[portfolio_df[risk_col] >= -max_loss]

        if feasible.empty:
            rows.append(
                {
                    "loss_bound_pct": L * 100,
                    "max_loss_dollars": -max_loss,
                    "n_feasible": 0,
                    "best_ev": np.nan,
                    "best_config": "NONE",
                }
            )
            continue

        best_idx = feasible[ev_column].idxmax()
        best = feasible.loc[best_idx]

        rows.append(
            {
                "loss_bound_pct": L * 100,
                "max_loss_dollars": -max_loss,
                "n_feasible": len(feasible),
                "best_ev": best[ev_column],
                "best_total_pnl": best["total_pnl"],
                "best_risk": best[risk_col],
                "best_capital_deployed": best["capital_deployed"],
                "best_deployment_rate": best["deployment_rate"],
                "best_config": best["config_label"],
                "best_model": best["model_type"],
                "best_fee": best["fee_type"],
                "best_kf": best["kelly_fraction"],
                "best_edge": best["buy_edge_threshold"],
                "best_km": best["kelly_mode"],
                "best_side": best["allowed_directions"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Monte Carlo CVaR
# ============================================================================


def compute_portfolio_cvar(
    scenario_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    alphas: list[float] | None = None,
    n_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute portfolio-level CVaR via Monte Carlo for each config.

    For each (entry_snapshot, config), draws ``n_samples`` independent
    portfolios where each category's winner is sampled from blend
    probabilities (= (model_prob + market_prob) / 2, renormalized).
    CVaR_α = mean of bottom ⌊α × n_samples⌋ portfolio PnLs.

    Uses per-entry normalization: CVaR is averaged across entry times.

    The default of 10,000 samples achieves < 1% relative error on CVaR
    estimates (validated via ``calibrate_mc_sample_size``).  Use 100,000+
    for publication-quality results.

    Args:
        scenario_df: Per-nominee PnL and probabilities from scenario_pnl.csv.
        entry_df: Entry-level P&L (for config column mapping).
        alphas: Tail fractions for CVaR. Defaults to [0.05, 0.10, 0.25].
        n_samples: Number of MC samples per (entry, config).  Default
            10,000 balances speed (~10× faster than 100K) with accuracy
            (< 1% relative error).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with CONFIG_COLS + cvar_5, cvar_10, cvar_25 columns,
        one row per config.
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.25]

    rng = np.random.default_rng(seed)

    # Build config column mapping from entry_df
    config_map = entry_df[CONFIG_COLS].drop_duplicates()

    # Build structure: for each (entry, model_type, config_label),
    # collect categories with their nominees' PnLs and blend probs.
    entry_config_cats: dict[tuple[str, str, str], dict[str, CategoryScenario]] = {}

    grouped = scenario_df.groupby(["entry_snapshot", "model_type", "config_label", "category"])
    for (entry, model_type, config_label, category), grp in grouped:
        key = (str(entry), str(model_type), str(config_label))
        if key not in entry_config_cats:
            entry_config_cats[key] = {}

        pnls = np.asarray(grp["pnl"].values)
        winners = np.asarray(grp["nominee"].values)
        # blend_prob = (model_prob + market_prob) / 2, renormalized to sum to 1
        blend = (np.asarray(grp["model_prob"].values) + np.asarray(grp["market_prob"].values)) / 2.0
        total = blend.sum()
        if total > 0:
            blend = blend / total
        else:
            # Uniform if no probability info
            blend = np.ones(len(blend)) / len(blend)

        entry_config_cats[key][str(category)] = CategoryScenario(
            winners=winners, probs=blend, pnls=pnls
        )

    # MC simulation per (entry, config) using portfolio_simulation
    cvar_rows: list[dict] = []

    for (entry, model_type, config_label), cats in entry_config_cats.items():
        if not cats:
            continue

        portfolio_pnl = sample_portfolio_pnl(cats, n_samples, rng)

        row: dict = {
            "entry_snapshot": entry,
            "model_type": model_type,
            "config_label": config_label,
        }
        for alpha in alphas:
            cvar_val = compute_cvar(portfolio_pnl, alpha)
            col_name = f"cvar_{int(alpha * 100)}"
            row[col_name] = round(cvar_val, 2)

        cvar_rows.append(row)

    cvar_df = pd.DataFrame(cvar_rows)

    # Average CVaR across entry times (per-entry normalization)
    cvar_cols = [f"cvar_{int(a * 100)}" for a in alphas]
    portfolio_cvar = cvar_df.groupby(["model_type", "config_label"], as_index=False)[
        cvar_cols
    ].mean()

    # Merge full CONFIG_COLS from config_map
    portfolio_cvar = portfolio_cvar.merge(config_map, on=["model_type", "config_label"])

    # Round CVaR columns
    for col in cvar_cols:
        portfolio_cvar[col] = portfolio_cvar[col].round(2)

    return portfolio_cvar[CONFIG_COLS + cvar_cols].reset_index(drop=True)


def calibrate_mc_sample_size(
    scenario_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    sample_sizes: list[int] | None = None,
    n_trials: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Measure CVaR convergence across MC sample sizes.

    Picks 3 representative configs (top, median, bottom by ev_pnl_blend
    from portfolio scores) and computes CVaR at each sample size with
    ``n_trials`` different seeds.

    Returns:
        DataFrame with columns: config_label, model_type, sample_size,
        trial, cvar_value.  Suitable for plotting convergence.
    """
    if sample_sizes is None:
        sample_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    # Get portfolio scores to pick representative configs
    portfolio = compute_portfolio_scores(entry_df)
    n = len(portfolio)

    # Pick top, median, bottom
    indices = [0, n // 2, n - 1]
    rep_configs = portfolio.iloc[indices][["model_type", "config_label"]].values.tolist()

    rows: list[dict] = []
    for model_type, config_label in rep_configs:
        # Filter scenario_df for this config
        mask = (scenario_df["model_type"] == model_type) & (
            scenario_df["config_label"] == config_label
        )
        cfg_scenario = scenario_df[mask]

        # Filter entry_df for this config
        entry_mask = (entry_df["model_type"] == model_type) & (
            entry_df["config_label"] == config_label
        )
        cfg_entry = entry_df[entry_mask]

        for n_samp in sample_sizes:
            for trial in range(n_trials):
                trial_seed = trial * 1000 + n_samp % 1000
                cvar_result = compute_portfolio_cvar(
                    cfg_scenario,
                    cfg_entry,
                    alphas=[alpha],
                    n_samples=n_samp,
                    seed=trial_seed,
                )
                if not cvar_result.empty:
                    cvar_val = cvar_result[f"cvar_{int(alpha * 100)}"].iloc[0]
                    rows.append(
                        {
                            "config_label": config_label,
                            "model_type": model_type,
                            "sample_size": n_samp,
                            "trial": trial,
                            "cvar_value": cvar_val,
                        }
                    )

    return pd.DataFrame(rows)


# ============================================================================
# Cross-year analysis
# ============================================================================


def compute_cross_year_scores(
    portfolio_2024: pd.DataFrame,
    portfolio_2025: pd.DataFrame,
    cvar_2024: pd.DataFrame | None = None,
    cvar_2025: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge portfolio scores across years and compute combined metrics.

    Combined EV = average of 2024 and 2025 EV.
    Worst case must pass independently in both years.
    If CVaR DataFrames are provided, merges them and computes avg CVaR.

    Returns:
        DataFrame with one row per config, containing both years' metrics
        and combined scores.
    """
    # Suffix columns for merge
    p24 = portfolio_2024.copy()
    p25 = portfolio_2025.copy()

    # Merge CVaR into per-year portfolios before cross-year merge
    if cvar_2024 is not None:
        p24 = p24.merge(cvar_2024, on=CONFIG_COLS)
    if cvar_2025 is not None:
        p25 = p25.merge(cvar_2025, on=CONFIG_COLS)

    merged = p24.merge(p25, on=CONFIG_COLS, suffixes=("_2024", "_2025"))

    # Combined metrics
    for col in [
        "total_pnl",
        "worst_pnl",
        "best_pnl",
        "ev_pnl_model",
        "ev_pnl_market",
        "ev_pnl_blend",
        "capital_deployed",
        "total_trades",
    ]:
        merged[f"avg_{col}"] = (merged[f"{col}_2024"] + merged[f"{col}_2025"]) / 2

    # Both-year worst: the worst of the two years' worst cases
    merged["min_worst_pnl"] = merged[["worst_pnl_2024", "worst_pnl_2025"]].min(axis=1)

    # cvar_0 = worst_pnl (deterministic alias for unified CVaR framework)
    merged["cvar_0_2024"] = merged["worst_pnl_2024"]
    merged["cvar_0_2025"] = merged["worst_pnl_2025"]
    merged["avg_cvar_0"] = merged["avg_worst_pnl"]

    # Average CVaR columns if present
    cvar_cols_check = [c for c in p24.columns if c.startswith("cvar_")]
    for col in cvar_cols_check:
        col_2024 = f"{col}_2024"
        col_2025 = f"{col}_2025"
        if col_2024 in merged.columns and col_2025 in merged.columns:
            merged[f"avg_{col}"] = (merged[col_2024] + merged[col_2025]) / 2

    # Profitability flags
    merged["profitable_2024"] = merged["total_pnl_2024"] > 0
    merged["profitable_2025"] = merged["total_pnl_2025"] > 0
    merged["profitable_both"] = merged["profitable_2024"] & merged["profitable_2025"]

    return merged.sort_values("avg_ev_pnl_blend", ascending=False).reset_index(drop=True)


def compute_cross_year_pareto(
    cross_year_df: pd.DataFrame,
    loss_bounds: list[float] | None = None,
    ev_column: str = "avg_ev_pnl_blend",
    risk_col: str = "worst_pnl",
) -> pd.DataFrame:
    """Pareto frontier across years: risk must pass in BOTH years.

    For each loss bound L, filters to configs where:
    - {risk_col}_2024 >= -L * bankroll_2024
    - {risk_col}_2025 >= -L * bankroll_2025

    The risk column can be ``worst_pnl`` (deterministic worst case) or any
    CVaR level (``cvar_5``, ``cvar_10``, ``cvar_25``).

    Then picks the config with highest average EV.
    """
    if loss_bounds is None:
        loss_bounds = DEFAULT_LOSS_BOUNDS

    col_2024 = f"{risk_col}_2024"
    col_2025 = f"{risk_col}_2025"

    rows: list[dict] = []

    for L in loss_bounds:
        # Use per-year bankroll thresholds
        max_loss_2024 = L * cross_year_df["total_bankroll_deployed_2024"].iloc[0]
        max_loss_2025 = L * cross_year_df["total_bankroll_deployed_2025"].iloc[0]

        feasible = cross_year_df[
            (cross_year_df[col_2024] >= -max_loss_2024)
            & (cross_year_df[col_2025] >= -max_loss_2025)
        ]

        if feasible.empty:
            rows.append(
                {
                    "loss_bound_pct": L * 100,
                    "n_feasible": 0,
                    "best_ev": np.nan,
                    "best_config": "NONE",
                }
            )
            continue

        best_idx = feasible[ev_column].idxmax()
        best = feasible.loc[best_idx]

        rows.append(
            {
                "loss_bound_pct": L * 100,
                "n_feasible": len(feasible),
                "best_avg_ev": best[ev_column],
                "best_total_pnl_2024": best["total_pnl_2024"],
                "best_total_pnl_2025": best["total_pnl_2025"],
                "best_risk_2024": best[col_2024],
                "best_risk_2025": best[col_2025],
                "best_config": best["config_label"],
                "best_model": best["model_type"],
                "best_fee": best["fee_type"],
                "best_kf": best["kelly_fraction"],
                "best_edge": best["buy_edge_threshold"],
                "best_km": best["kelly_mode"],
                "best_side": best["allowed_directions"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Late-season analysis
# ============================================================================


def compute_late_season_scores(
    entry_df: pd.DataFrame,
    ceremony_year: int,
    max_days_before: int = 25,
) -> pd.DataFrame:
    """Portfolio scores using only late-season entry points.

    Filters to entry points within max_days_before days of the ceremony,
    then computes portfolio-level scores.  This gives the most relevant
    view for late-season 2026 deployment.

    Args:
        entry_df: Entry-level P&L with scenario columns.
        ceremony_year: Which year's ceremony to use for date filtering.
        max_days_before: Maximum days before ceremony to include.

    Returns:
        Portfolio-level scores for late-season entries only.
    """
    from oscar_prediction_market.data.awards_calendar import CALENDARS

    calendar = CALENDARS[ceremony_year]
    ceremony_date = calendar.oscar_ceremony_date_local

    # Parse entry dates from snapshot names (format: YYYY-MM-DD_event_name)
    entry_df = entry_df.copy()
    entry_df["entry_date"] = pd.to_datetime(entry_df["entry_snapshot"].str[:10], format="%Y-%m-%d")
    entry_df["days_before_ceremony"] = (
        pd.Timestamp(ceremony_date) - entry_df["entry_date"]
    ).dt.days

    late = entry_df[entry_df["days_before_ceremony"] <= max_days_before]
    if late.empty:
        return pd.DataFrame()

    return compute_portfolio_scores(late)


# ============================================================================
# CLI
# ============================================================================


def run_single_year(
    ceremony_year: int,
    *,
    skip_cvar: bool = False,
    mc_samples: int = 10_000,
    exp_dir: Path | None = None,
) -> None:
    """Run scenario scoring for one year and save results."""
    print(f"\n{'=' * 70}")
    print(f"Scenario Scoring: {ceremony_year}")
    print(f"{'=' * 70}")

    entry_df = load_entry_pnl(ceremony_year, exp_dir=exp_dir)
    print(f"Loaded {len(entry_df)} entry rows")
    print(f"  Categories: {entry_df['category'].nunique()}")
    print(f"  Entry points: {entry_df['entry_snapshot'].nunique()}")
    print(f"  Configs: {entry_df['config_label'].nunique()}")
    print(f"  Models: {entry_df['model_type'].nunique()}")

    if exp_dir is not None:
        results_dir = exp_dir / str(ceremony_year) / "results"
    else:
        results_dir = YEAR_CONFIGS[ceremony_year].results_dir

    # Portfolio scores (per-entry normalized)
    portfolio_df = compute_portfolio_scores(entry_df)
    portfolio_df.to_csv(results_dir / "portfolio_scores.csv", index=False)
    print(f"\nSaved portfolio_scores.csv ({len(portfolio_df)} configs)")

    # Category scores
    cat_df = compute_category_scores(entry_df)
    cat_df.to_csv(results_dir / "category_scores.csv", index=False)
    print(f"Saved category_scores.csv ({len(cat_df)} rows)")

    # Temporal scores
    temporal_df = compute_temporal_scores(entry_df)
    temporal_df.to_csv(results_dir / "temporal_scores.csv", index=False)
    print(f"Saved temporal_scores.csv ({len(temporal_df)} rows)")

    # Late-season scores
    late_df = compute_late_season_scores(entry_df, ceremony_year, max_days_before=25)
    if not late_df.empty:
        late_df.to_csv(results_dir / "late_season_scores.csv", index=False)
        print(f"Saved late_season_scores.csv ({len(late_df)} configs)")

    # CVaR (Monte Carlo)
    cvar_df = None
    if not skip_cvar:
        try:
            scenario_df = load_scenario_pnl(ceremony_year, exp_dir=exp_dir)
            print(f"\nLoaded {len(scenario_df)} scenario P&L rows")

            # MC calibration
            print("Running MC sample size calibration...")
            calib_df = calibrate_mc_sample_size(scenario_df, entry_df)
            calib_df.to_csv(results_dir / "mc_calibration.csv", index=False)
            print(f"Saved mc_calibration.csv ({len(calib_df)} rows)")

            # CVaR computation
            print(f"Computing portfolio CVaR (n_samples={mc_samples:,})...")
            cvar_df = compute_portfolio_cvar(scenario_df, entry_df, n_samples=mc_samples)
            cvar_df.to_csv(results_dir / "portfolio_cvar.csv", index=False)
            print(f"Saved portfolio_cvar.csv ({len(cvar_df)} configs)")

        except FileNotFoundError as e:
            print(f"\nSkipping CVaR: {e}")

    # Build merged portfolio with unified CVaR columns (cvar_0 = worst_pnl)
    merged_portfolio = portfolio_df.copy()
    merged_portfolio["cvar_0"] = merged_portfolio["worst_pnl"]
    if cvar_df is not None:
        merged_portfolio = merged_portfolio.merge(cvar_df, on=CONFIG_COLS)

    # Pareto frontiers: EV variants with worst-case constraint (cvar_0)
    for ev_col, label in [
        ("ev_pnl_model", "model"),
        ("ev_pnl_market", "market"),
        ("ev_pnl_blend", "blend"),
    ]:
        frontier = compute_pareto_frontier(merged_portfolio, ev_column=ev_col, risk_col="cvar_0")
        frontier.to_csv(results_dir / f"pareto_frontier_{label}.csv", index=False)
        print(f"Saved pareto_frontier_{label}.csv")

    # Pareto frontiers: blend EV with all CVaR alpha levels
    available_alphas = [0.0]  # worst-case always available
    if cvar_df is not None:
        available_alphas = list(CVAR_ALPHAS)

    for alpha in available_alphas:
        alpha_int = int(alpha * 100)
        risk_col = f"cvar_{alpha_int}"
        alpha_label = f"cvar{alpha_int:02d}"

        frontier = compute_pareto_frontier(
            merged_portfolio, ev_column="ev_pnl_blend", risk_col=risk_col
        )
        fname = f"pareto_frontier_{alpha_label}_blend.csv"
        frontier.to_csv(results_dir / fname, index=False)
        print(f"Saved {fname}")

    # Print top configs
    print(f"\n{'─' * 70}")
    print("Top 10 configs by EV (blend):")
    print(f"{'─' * 70}")
    top = portfolio_df.head(10)
    for _, row in top.iterrows():
        print(
            f"  {row['model_type']:<12s} {row['config_label'][:40]:<40s}"
            f"  EV=${row['ev_pnl_blend']:>+8.0f}"
            f"  Worst=${row['worst_pnl']:>+8.0f}"
            f"  Actual=${row['total_pnl']:>+8.0f}"
            f"  Deploy={row['deployment_rate']:>5.1f}%"
        )

    # Print unified CVaR Pareto frontiers
    for alpha in available_alphas:
        alpha_int = int(alpha * 100)
        risk_col = f"cvar_{alpha_int}"
        alpha_pct = f"CVaR {alpha_int}%" if alpha > 0 else "Worst-case"

        frontier_print = compute_pareto_frontier(
            merged_portfolio, ev_column="ev_pnl_blend", risk_col=risk_col
        )
        print(f"\n{'─' * 70}")
        print(f"Pareto Frontier (EV blend, {alpha_pct} constraint):")
        print(f"{'─' * 70}")
        for _, row in frontier_print.iterrows():
            if row["n_feasible"] == 0:
                print(f"  L={row['loss_bound_pct']:>5.0f}%: No feasible configs")
            else:
                print(
                    f"  L={row['loss_bound_pct']:>5.0f}%: "
                    f"EV=${row['best_ev']:>+8.0f}  "
                    f"{alpha_pct}=${row['best_risk']:>+8.0f}  "
                    f"Actual=${row['best_total_pnl']:>+8.0f}  "
                    f"({row['n_feasible']} feasible)  "
                    f"{row['best_model']}/{row['best_side']}"
                )


def run_cross_year(
    *,
    skip_cvar: bool = False,
    mc_samples: int = 10_000,
    exp_dir: Path | None = None,
) -> None:
    """Run cross-year analysis and save results."""
    print(f"\n{'=' * 70}")
    print("Cross-Year Scenario Scoring: 2024 + 2025")
    print(f"{'=' * 70}")

    entry_2024 = load_entry_pnl(2024, exp_dir=exp_dir)
    entry_2025 = load_entry_pnl(2025, exp_dir=exp_dir)

    portfolio_2024 = compute_portfolio_scores(entry_2024)
    portfolio_2025 = compute_portfolio_scores(entry_2025)

    # CVaR
    cvar_2024 = None
    cvar_2025 = None
    if not skip_cvar:
        try:
            scenario_2024 = load_scenario_pnl(2024, exp_dir=exp_dir)
            print(f"Computing CVaR for 2024 (n_samples={mc_samples:,})...")
            cvar_2024 = compute_portfolio_cvar(
                scenario_2024,
                entry_2024,
                n_samples=mc_samples,
            )
            print(f"Computed CVaR for 2024 ({len(cvar_2024)} configs)")
        except FileNotFoundError as e:
            print(f"Skipping 2024 CVaR: {e}")
        try:
            scenario_2025 = load_scenario_pnl(2025, exp_dir=exp_dir)
            print(f"Computing CVaR for 2025 (n_samples={mc_samples:,})...")
            cvar_2025 = compute_portfolio_cvar(
                scenario_2025,
                entry_2025,
                n_samples=mc_samples,
            )
            print(f"Computed CVaR for 2025 ({len(cvar_2025)} configs)")
        except FileNotFoundError as e:
            print(f"Skipping 2025 CVaR: {e}")

    cross_year_dir = exp_dir if exp_dir is not None else BUY_HOLD_EXP_DIR

    cross_year = compute_cross_year_scores(
        portfolio_2024,
        portfolio_2025,
        cvar_2024=cvar_2024,
        cvar_2025=cvar_2025,
    )
    cross_year.to_csv(cross_year_dir / "cross_year_scenario_scores.csv", index=False)
    print(f"Saved cross_year_scenario_scores.csv ({len(cross_year)} configs)")

    # Cross-year Pareto frontier: EV variants with worst-case constraint
    for ev_col, label in [
        ("avg_ev_pnl_model", "model"),
        ("avg_ev_pnl_market", "market"),
        ("avg_ev_pnl_blend", "blend"),
    ]:
        frontier = compute_cross_year_pareto(cross_year, ev_column=ev_col, risk_col="worst_pnl")
        frontier.to_csv(cross_year_dir / f"cross_year_pareto_{label}.csv", index=False)
        print(f"Saved cross_year_pareto_{label}.csv")

    # Cross-year CVaR Pareto: unified loop over alpha levels
    available_alphas = [0.0]  # worst-case (= cvar_0)
    if cvar_2024 is not None and cvar_2025 is not None:
        available_alphas = list(CVAR_ALPHAS)

    for alpha in available_alphas:
        alpha_int = int(alpha * 100)
        risk_col = f"cvar_{alpha_int}" if alpha > 0 else "worst_pnl"
        alpha_label = f"cvar{alpha_int:02d}"

        frontier = compute_cross_year_pareto(
            cross_year,
            loss_bounds=CROSS_YEAR_LOSS_BOUNDS,
            risk_col=risk_col,
        )
        fname = f"cross_year_pareto_{alpha_label}_blend.csv"
        frontier.to_csv(cross_year_dir / fname, index=False)
        print(f"Saved {fname}")

    # Print top configs
    print(f"\n{'─' * 70}")
    print("Top 10 by average EV (blend):")
    print(f"{'─' * 70}")
    top = cross_year.head(10)
    for _, row in top.iterrows():
        print(
            f"  {row['model_type']:<12s}"
            f"  AvgEV=${row['avg_ev_pnl_blend']:>+8.0f}"
            f"  Worst24=${row['worst_pnl_2024']:>+7.0f}"
            f"  Worst25=${row['worst_pnl_2025']:>+7.0f}"
            f"  Act24=${row['total_pnl_2024']:>+7.0f}"
            f"  Act25=${row['total_pnl_2025']:>+7.0f}"
            f"  Both={'Y' if row['profitable_both'] else 'N'}"
        )

    # Print unified cross-year CVaR Pareto frontiers
    for alpha in available_alphas:
        alpha_int = int(alpha * 100)
        risk_col = f"cvar_{alpha_int}" if alpha > 0 else "worst_pnl"
        alpha_pct = f"CVaR {alpha_int}%" if alpha > 0 else "Worst-case"

        frontier_print = compute_cross_year_pareto(
            cross_year,
            loss_bounds=CROSS_YEAR_LOSS_BOUNDS,
            risk_col=risk_col,
        )
        print(f"\n{'─' * 70}")
        print(f"Cross-Year Pareto ({alpha_pct} constraint, BOTH years):")
        print(f"{'─' * 70}")
        for _, row in frontier_print.iterrows():
            if row["n_feasible"] == 0:
                print(f"  L={row['loss_bound_pct']:>5.0f}%: No feasible configs")
            else:
                print(
                    f"  L={row['loss_bound_pct']:>5.0f}%: "
                    f"AvgEV=${row['best_avg_ev']:>+8.0f}  "
                    f"{alpha_pct}24=${row['best_risk_2024']:>+7.0f}  "
                    f"{alpha_pct}25=${row['best_risk_2025']:>+7.0f}  "
                    f"({row['n_feasible']} feasible)  "
                    f"{row['best_model']}/{row['best_side']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario-based config scoring")
    parser.add_argument(
        "--ceremony-year",
        type=int,
        choices=sorted(YEAR_CONFIGS.keys()),
        help="Run scoring for a single year.",
    )
    parser.add_argument(
        "--cross-year",
        action="store_true",
        help="Run cross-year (2024+2025) analysis.",
    )
    parser.add_argument(
        "--skip-cvar",
        action="store_true",
        help="Skip Monte Carlo CVaR computation (faster).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10_000,
        help="Number of MC samples for CVaR estimation (default: 10000).",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help=(
            "Override experiment directory for input/output. "
            "Reads from <exp-dir>/<year>/results/ and writes cross-year "
            "output to <exp-dir>/. Default: storage/d20260225_buy_hold_backtest."
        ),
    )
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir) if args.exp_dir else None

    if args.ceremony_year:
        run_single_year(
            args.ceremony_year,
            skip_cvar=args.skip_cvar,
            mc_samples=args.mc_samples,
            exp_dir=exp_dir,
        )

    if args.cross_year:
        run_cross_year(skip_cvar=args.skip_cvar, mc_samples=args.mc_samples, exp_dir=exp_dir)

    if not args.ceremony_year and not args.cross_year:
        # Default: run both years + cross-year
        for year in [2024, 2025]:
            run_single_year(
                year,
                skip_cvar=args.skip_cvar,
                mc_samples=args.mc_samples,
                exp_dir=exp_dir,
            )
        run_cross_year(skip_cvar=args.skip_cvar, mc_samples=args.mc_samples, exp_dir=exp_dir)


if __name__ == "__main__":
    main()
