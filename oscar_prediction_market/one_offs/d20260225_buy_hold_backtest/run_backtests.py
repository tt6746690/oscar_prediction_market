"""Run buy-and-hold backtests: buy once at each entry point, hold to settlement.

For each (category, model, entry_point, config), creates a single MarketSnapshot
from the snapshot's predictions + market prices at entry timestamp, runs it
through the existing BacktestEngine, and settles against the known winner.

This is a strict subset of the rebalancing engine: with exactly one moment,
no rebalancing/selling can occur. The engine buys once and holds to settlement.

Models:
    Always runs all 6 models: 4 individual (lr, clogit, gbt, cal_sgbt) plus
    2 ensembles (avg_ensemble, clogit_cal_sgbt_ensemble).  Use ``--fast``
    for a reduced subset during development.

Usage::

    cd "$(git rev-parse --show-toplevel)"

    # Fast mode (~30s, 18 configs × 3 categories × 2 individual models):
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.run_backtests --fast

    # Full grid (all 6 models × all categories × 615 configs):
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.run_backtests
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.one_offs.d20260220_backtest_strategies.evaluation import (
    evaluate_model_accuracy,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    YEAR_CONFIGS,
    YearConfig,
)
from oscar_prediction_market.trading.backtest import (
    BacktestEngine,
    MarketSnapshot,
)
from oscar_prediction_market.trading.backtest_configs import (
    generate_trading_configs,
)
from oscar_prediction_market.trading.name_matching import (
    match_nominees,
    validate_matching,
)
from oscar_prediction_market.trading.oscar_data import (
    build_market_prices,
    estimate_category_spreads,
    fetch_and_cache_market_data,
    get_winner_kalshi_name,
)
from oscar_prediction_market.trading.oscar_moments import build_entry_moment
from oscar_prediction_market.trading.oscar_prediction_source import (
    load_all_snapshot_predictions,
    load_ensemble_predictions,
    load_nomination_dataset,
    translate_predictions,
)
from oscar_prediction_market.trading.schema import PositionDirection
from oscar_prediction_market.trading.temporal_model import (
    SnapshotInfo,
)

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Default delay: inferred lag +6h (same as the recommended d20260220 config)
DEFAULT_LAG_HOURS = 6.0

BANKROLL = 1000.0
ENSEMBLE_SHORT_NAME = "avg_ensemble"

#: The 4 individual model types used in backtests.
BACKTEST_MODEL_TYPES: list[ModelType] = [
    ModelType.LOGISTIC_REGRESSION,
    ModelType.CONDITIONAL_LOGIT,
    ModelType.GRADIENT_BOOSTING,
    ModelType.CALIBRATED_SOFTMAX_GBT,
]

#: Custom ensemble: average of clogit and cal_sgbt (the two best individual
#: models across years).  Defined here so both run_backtests and analysis
#: scripts use the same label.
CLOGIT_CAL_SGBT_ENSEMBLE_LABEL = "clogit_cal_sgbt_ensemble"
CLOGIT_CAL_SGBT_ENSEMBLE_TYPES = [
    ModelType.CONDITIONAL_LOGIT,
    ModelType.CALIBRATED_SOFTMAX_GBT,
]


# ============================================================================
# Ensemble definitions
# ============================================================================

#: All ensembles to run.  Each entry is ``(label, model_types)``.
#: The avg_ensemble averages all 4 individual models; the clogit_cal_sgbt_ensemble
#: averages just the two strongest.
ALL_ENSEMBLES: list[tuple[str, list[ModelType]]] = [
    (ENSEMBLE_SHORT_NAME, list(BACKTEST_MODEL_TYPES)),
    (CLOGIT_CAL_SGBT_ENSEMBLE_LABEL, CLOGIT_CAL_SGBT_ENSEMBLE_TYPES),
]


# ============================================================================
# Per-category backtest
# ============================================================================


def run_category_buy_hold(
    category: OscarCategory,
    snapshot_keys: list[SnapshotInfo],
    lag_hours: float,
    year_config: YearConfig,
    model_types: list[ModelType],
    ensembles: list[tuple[str, list[ModelType]]],
    fast: bool = False,
    config_grid: str = "full",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run buy-and-hold backtest for one category across all models.

    Runs each individual model in ``model_types`` plus each ensemble in
    ``ensembles``.  For each (model, entry_point, config): builds one
    MarketSnapshot, runs the BacktestEngine, and settles.

    Returns:
        (pnl_rows, accuracy_rows, model_vs_market_rows, scenario_rows)
    """
    cat_slug = category.slug

    print(f"\n{'=' * 70}")
    print(f"Category: {category.name} ({cat_slug})")
    print(f"  Entry points: {len(snapshot_keys)}")
    print(f"{'=' * 70}")

    # Compute snapshot availability inline (inferred lag)
    snapshot_availability = {
        s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
    }

    # Fetch market data (always fetch hourly for inferred-lag entry)
    calendar = year_config.calendar
    fetch_result = fetch_and_cache_market_data(
        category,
        ceremony_year=year_config.ceremony_year,
        start_date=calendar.oscar_nominations_date_local,
        end_date=calendar.oscar_ceremony_date_local,
        market_data_dir=year_config.market_data_dir,
        fetch_hourly=True,
    )
    if fetch_result is None:
        return [], [], [], []

    market, daily_candles, hourly_candles, trades_df = fetch_result
    spread_by_ticker, mean_spread = estimate_category_spreads(trades_df)
    trading_configs = generate_trading_configs(
        bankroll=BANKROLL,
        spread_penalty=mean_spread,
        fast=fast,
        grid=config_grid,
    )

    # Load nomination records & build name-mapping data (once per category)
    snapshot_key_strs = [k.dir_name for k in snapshot_keys]
    dataset = load_nomination_dataset(year_config.datasets_dir, category, snapshot_key_strs[0])

    pnl_rows: list[dict] = []
    accuracy_rows: list[dict] = []
    model_vs_market_rows: list[dict] = []
    scenario_rows: list[dict] = []

    # --- Build model_runs: (label, preds_by_snapshot) for all models ---
    model_runs: list[tuple[str, dict[str, dict[str, float]]]] = []

    for model_type in model_types:
        short_name = model_type.short_name
        preds = load_all_snapshot_predictions(
            category,
            model_type,
            snapshot_key_strs,
            year_config.models_dir,
            dataset=dataset,
            ceremony_year=year_config.ceremony_year,
        )
        model_runs.append((short_name, preds))

    for ens_label, ens_types in ensembles:
        preds = load_ensemble_predictions(
            category,
            ens_types,
            snapshot_key_strs,
            year_config.models_dir,
            dataset=dataset,
            ceremony_year=year_config.ceremony_year,
        )
        model_runs.append((ens_label, preds))

    # --- Run each model through the backtest grid ---
    for short_name, preds_by_snapshot in model_runs:
        print(f"\n  Model: {short_name}")

        first_snap_preds = next(iter(preds_by_snapshot.values()))
        model_names = list(first_snap_preds.keys())

        # Name matching
        nominee_map = match_nominees(
            model_names=model_names,
            kalshi_names=list(market.nominee_tickers.keys()),
            category=category,
            ceremony_year=year_config.ceremony_year,
        )
        if not nominee_map:
            print(f"    WARNING: No name matches for {short_name}")
            continue

        kalshi_names = list(market.nominee_tickers.keys())
        validate_matching(
            nominee_map, model_names, kalshi_names, category, year_config.ceremony_year
        )
        matched_kalshi_names = set(nominee_map.values())

        # Translate predictions from model-name space to kalshi-name space
        preds_by_snapshot = translate_predictions(preds_by_snapshot, nominee_map)

        try:
            if year_config.winners is None:
                raise ValueError(f"No winners for {year_config.ceremony_year} — cannot settle")
            winner_kalshi_name = get_winner_kalshi_name(
                category, matched_kalshi_names, year_config.winners
            )
        except ValueError as e:
            print(f"    ERROR: {e}")
            continue
        print(f"    Winner: {winner_kalshi_name}")

        # Spread penalties
        spread_penalties = {
            kn: spread_by_ticker[t]
            for kn in matched_kalshi_names
            if (t := market.nominee_tickers.get(kn)) and t in spread_by_ticker
        }

        # Market prices by date
        market_prices_by_date = build_market_prices(daily_candles)

        # Model accuracy per snapshot
        for snap_key_str in snapshot_key_strs:
            snap_preds = preds_by_snapshot.get(snap_key_str, {})
            if not snap_preds:
                continue
            filtered_preds = {
                name: prob for name, prob in snap_preds.items() if name in matched_kalshi_names
            }
            acc = evaluate_model_accuracy(filtered_preds, winner_kalshi_name)
            accuracy_rows.append(
                {
                    "category": cat_slug,
                    "model_type": short_name,
                    "snapshot_key": snap_key_str,
                    **acc,
                }
            )

            # Model vs market divergence
            snap_date_str = snap_key_str[:10]
            market_on_snap = market_prices_by_date.get(snap_date_str, {})
            if market_on_snap and filtered_preds:
                total_market = sum(market_on_snap.values())
                for name in matched_kalshi_names:
                    model_p = filtered_preds.get(name, 0)
                    market_p = market_on_snap.get(name, 0) if total_market > 0 else 0
                    model_vs_market_rows.append(
                        {
                            "category": cat_slug,
                            "model_type": short_name,
                            "snapshot_key": snap_key_str,
                            "nominee": name,
                            "model_prob": round(model_p, 4),
                            "market_prob": round(market_p, 4),
                            "divergence": round(model_p - market_p, 4),
                            "is_winner": name == winner_kalshi_name,
                        }
                    )

        # Build one entry moment per snapshot
        entry_moments: list[tuple[SnapshotInfo, MarketSnapshot]] = []
        for key in snapshot_keys:
            moment = build_entry_moment(
                snapshot_key=key,
                snapshot_availability=snapshot_availability,
                preds_by_snapshot=preds_by_snapshot,
                matched_names=matched_kalshi_names,
                market_prices_by_date=market_prices_by_date,
                hourly_candles=hourly_candles,
            )
            if moment is not None:
                entry_moments.append((key, moment))

        print(f"    Entry points with data: {len(entry_moments)}")
        if not entry_moments:
            continue

        # Run each entry point × config combination
        n_configs = len(trading_configs)
        n_entries = len(entry_moments)
        print(
            f"    Running {n_entries} entries × {n_configs} configs = {n_entries * n_configs} backtests"
        )

        for key, moment in entry_moments:
            avail_at = snapshot_availability[key.dir_name]
            avail_et_str = avail_at.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET")

            for cfg in trading_configs:
                # Single-moment engine run: buy once, hold to settlement
                engine = BacktestEngine(cfg)
                result = engine.run(
                    moments=[moment],
                    spread_penalties=spread_penalties if spread_penalties else None,
                )

                # Compute capital actually spent on contracts
                if result.portfolio_history:
                    capital_deployed = sum(
                        pos.outlay_dollars
                        for pos in result.portfolio_history[-1].positions
                        if pos.contracts > 0
                    )
                else:
                    capital_deployed = 0.0

                # Settle: if no trades (empty settlements), PnL = 0.
                # Otherwise fail-fast — winner must be in the universe.
                if result.total_trades == 0:
                    final_cash = BANKROLL
                    actual_pnl = 0.0
                    return_pct = 0.0
                    worst_pnl = 0.0
                    best_pnl = 0.0
                    ev_pnl_model = 0.0
                    ev_pnl_market = 0.0
                    ev_pnl_blend = 0.0
                    n_positions = 0
                else:
                    settlement = result.settle(winner_kalshi_name)
                    final_cash = round(settlement.final_cash, 2)
                    actual_pnl = round(settlement.total_pnl, 2)
                    return_pct = settlement.return_pct

                    # Scenario-level metrics across all possible winners
                    all_pnls = {name: s.total_pnl for name, s in result.settlements.items()}
                    worst_pnl = round(min(all_pnls.values()), 2)
                    best_pnl = round(max(all_pnls.values()), 2)

                    # EV using model probabilities (from entry moment predictions)
                    settlement_names = set(all_pnls.keys())
                    raw_model = {
                        name: moment.predictions.get(name, 0.0) for name in settlement_names
                    }
                    model_total = sum(raw_model.values())
                    if model_total > 0:
                        ev_pnl_model = round(
                            sum(
                                (p / model_total) * all_pnls[name] for name, p in raw_model.items()
                            ),
                            2,
                        )
                    else:
                        ev_pnl_model = 0.0

                    # EV using market-implied probabilities (YES prices)
                    raw_market = {name: moment.prices.get(name, 0.0) for name in settlement_names}
                    market_total = sum(raw_market.values())
                    if market_total > 0:
                        ev_pnl_market = round(
                            sum(
                                (p / market_total) * all_pnls[name]
                                for name, p in raw_market.items()
                            ),
                            2,
                        )
                    else:
                        ev_pnl_market = 0.0

                    ev_pnl_blend = round((ev_pnl_model + ev_pnl_market) / 2, 2)

                    # Per-winner scenario rows
                    for name, pnl_val in all_pnls.items():
                        scenario_rows.append(
                            {
                                "category": cat_slug,
                                "model_type": short_name,
                                "entry_snapshot": key.dir_name,
                                "config_label": cfg.label,
                                "nominee": name,
                                "pnl": round(pnl_val, 2),
                                "model_prob": (
                                    round(raw_model[name] / model_total, 6)
                                    if model_total > 0
                                    else 0.0
                                ),
                                "market_prob": (
                                    round(raw_market[name] / market_total, 6)
                                    if market_total > 0
                                    else 0.0
                                ),
                            }
                        )

                    n_positions = sum(
                        1 for pos in result.portfolio_history[-1].positions if pos.contracts > 0
                    )

                row = {
                    "category": cat_slug,
                    "model_type": short_name,
                    "entry_snapshot": key.dir_name,
                    "entry_events": key.dir_name,
                    "entry_timestamp": avail_et_str,
                    "config_label": cfg.label,
                    "fee_type": cfg.trading.fee_type.value,
                    "kelly_fraction": cfg.trading.kelly.kelly_fraction,
                    "buy_edge_threshold": cfg.trading.kelly.buy_edge_threshold,
                    "min_price": cfg.trading.min_price,
                    "kelly_mode": cfg.trading.kelly.kelly_mode.value,
                    "bankroll_mode": cfg.simulation.bankroll_mode.value,
                    "allowed_directions": (
                        "yes"
                        if cfg.trading.allowed_directions == frozenset({PositionDirection.YES})
                        else "no"
                        if cfg.trading.allowed_directions == frozenset({PositionDirection.NO})
                        else "all"
                    ),
                    "total_trades": result.total_trades,
                    "total_fees": round(result.total_fees_paid, 2),
                    "capital_deployed": round(capital_deployed, 2),
                    "n_positions": n_positions,
                    "final_cash": final_cash,
                    "total_pnl": actual_pnl,
                    "return_pct": return_pct,
                    "worst_pnl": worst_pnl,
                    "best_pnl": best_pnl,
                    "ev_pnl_model": ev_pnl_model,
                    "ev_pnl_market": ev_pnl_market,
                    "ev_pnl_blend": ev_pnl_blend,
                }

                pnl_rows.append(row)

    return pnl_rows, accuracy_rows, model_vs_market_rows, scenario_rows


# ============================================================================
# Fast mode categories/models
# ============================================================================

_FAST_CATEGORIES = [
    OscarCategory.BEST_PICTURE,
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
]
_FAST_MODEL_TYPES = [
    ModelType.CALIBRATED_SOFTMAX_GBT,
    ModelType.LOGISTIC_REGRESSION,
]


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the buy-and-hold backtest pipeline.

    Always runs all 6 models (4 individual + 2 ensembles) in full mode.
    Use ``--fast`` for a reduced subset during development.
    """
    parser = argparse.ArgumentParser(description="Buy-and-hold backtests")
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=2025,
        choices=sorted(YEAR_CONFIGS.keys()),
        help="Ceremony year to backtest (default: 2025).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Category slugs to run (default: all for the year).",
    )
    parser.add_argument(
        "--inferred-lag-hours",
        type=float,
        default=6.0,
        help="Hours after event end before signal is available (default: 6).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory (default: year-specific dir).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 18 configs × 3 categories × 2 individual models, no ensembles.",
    )
    parser.add_argument(
        "--config-grid",
        type=str,
        choices=["full", "targeted"],
        default="full",
        help=(
            "Config grid to use: 'full' (588 configs, default) or "
            "'targeted' (27 configs: taker fees, multi_outcome, all directions). "
            "Ignored when --fast is set."
        ),
    )
    args = parser.parse_args()

    year_config = YEAR_CONFIGS[args.ceremony_year]
    results_dir = Path(args.results_dir) if args.results_dir else year_config.results_dir
    lag_hours: float = args.inferred_lag_hours
    config_grid: str = args.config_grid

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Categories and models: full mode runs all 6 models, fast mode runs 2
    if args.fast:
        default_categories = [c for c in _FAST_CATEGORIES if c in year_config.categories]
        model_types = _FAST_MODEL_TYPES
        ensembles: list[tuple[str, list[ModelType]]] = []
    else:
        default_categories = list(year_config.categories)
        model_types = list(BACKTEST_MODEL_TYPES)
        ensembles = ALL_ENSEMBLES

    if args.categories:
        categories = [c for c in year_config.categories if c.slug in args.categories]
        if not categories:
            print(f"ERROR: No matching categories for {args.categories}")
            return
    else:
        categories = default_categories

    # Derive snapshot keys (entry points)
    snapshot_keys = year_config.snapshot_keys()
    snapshot_availability = {
        s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
    }

    n_models = len(model_types) + len(ensembles)
    print("=" * 70)
    print(f"Buy-and-Hold Backtest: {args.ceremony_year} Ceremony")
    print("=" * 70)
    print(f"Categories: {[c.slug for c in categories]}")
    print(f"Models: {n_models} ({len(model_types)} individual + {len(ensembles)} ensembles)")
    print(f"Signal delay: inferred lag +{lag_hours}h")
    print("Trading configs: category-specific shared grid")

    print(f"\nEntry points ({len(snapshot_keys)}):")
    for key in snapshot_keys:
        avail = snapshot_availability.get(key.dir_name)
        avail_str = avail.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET") if avail else "N/A"
        print(f"  {key.dir_name} → entry at {avail_str}")

    # Run backtests per category
    all_pnl: list[dict] = []
    all_accuracy: list[dict] = []
    all_model_vs_market: list[dict] = []
    all_scenario: list[dict] = []

    for category in categories:
        pnl, acc, mvm, scen = run_category_buy_hold(
            category,
            snapshot_keys,
            lag_hours=lag_hours,
            year_config=year_config,
            model_types=model_types,
            ensembles=ensembles,
            fast=args.fast,
            config_grid=config_grid,
        )
        all_pnl.extend(pnl)
        all_accuracy.extend(acc)
        all_model_vs_market.extend(mvm)
        all_scenario.extend(scen)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)

    if all_pnl:
        pnl_df = pd.DataFrame(all_pnl)

        entry_csv = results_dir / "entry_pnl.csv"
        pnl_df.to_csv(entry_csv, index=False)
        print(f"\nSaved {len(pnl_df)} P&L rows to entry_pnl.csv")

        # Aggregate: sum across entry points per (category, model, config)
        agg_cols = [
            "category",
            "model_type",
            "config_label",
            "fee_type",
            "kelly_fraction",
            "buy_edge_threshold",
            "min_price",
            "kelly_mode",
            "bankroll_mode",
            "allowed_directions",
        ]
        agg_df = pnl_df.groupby(agg_cols, as_index=False).agg(
            total_pnl=("total_pnl", "sum"),
            worst_pnl=("worst_pnl", "sum"),
            best_pnl=("best_pnl", "sum"),
            ev_pnl_model=("ev_pnl_model", "sum"),
            ev_pnl_market=("ev_pnl_market", "sum"),
            ev_pnl_blend=("ev_pnl_blend", "sum"),
            total_fees=("total_fees", "sum"),
            capital_deployed=("capital_deployed", "sum"),
            total_trades=("total_trades", "sum"),
            n_entries=("entry_snapshot", "nunique"),
            entries_with_trades=("total_trades", lambda x: (x > 0).sum()),
        )
        agg_df["total_bankroll_deployed"] = agg_df["n_entries"] * BANKROLL
        agg_df["return_pct"] = round(
            agg_df["total_pnl"] / agg_df["total_bankroll_deployed"] * 100, 1
        )
        agg_df.to_csv(results_dir / "aggregate_pnl.csv", index=False)
        print(f"Saved {len(agg_df)} aggregate P&L rows to aggregate_pnl.csv")

    if all_accuracy:
        acc_df = pd.DataFrame(all_accuracy)
        acc_df.to_csv(results_dir / "model_accuracy.csv", index=False)
        print(f"Saved {len(acc_df)} accuracy rows")

    if all_model_vs_market:
        mvm_df = pd.DataFrame(all_model_vs_market)
        mvm_df.to_csv(results_dir / "model_vs_market.csv", index=False)
        print(f"Saved {len(mvm_df)} model-vs-market rows")

    if all_scenario:
        scenario_df = pd.DataFrame(all_scenario)
        scenario_df.to_csv(results_dir / "scenario_pnl.csv", index=False)
        print(f"Saved {len(scenario_df)} scenario P&L rows to scenario_pnl.csv")

    # Summary: best aggregate config per model × category
    if all_pnl:
        pnl_df = pd.DataFrame(all_pnl)
        print("\n" + "=" * 70)
        print("P&L Summary: Best per-entry P&L by model × category")
        print("=" * 70)

        entry_best = pnl_df.loc[
            pnl_df.groupby(["category", "model_type", "entry_snapshot"])["total_pnl"].idxmax()
        ]
        for (cat, model), group in entry_best.groupby(["category", "model_type"]):
            actual_pnl = group["total_pnl"].sum()
            total_trades = group["total_trades"].sum()
            n_entries = len(group)
            print(
                f"  {cat:<25} {model:<12} "
                f"entries={n_entries}  trades={total_trades:>3}  "
                f"sum P&L=${actual_pnl:>+8.2f}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
