"""Run daily buy-once backtests for all categories x model types x trading configs.

This is the core backtest runner for the multi-category backtest strategies
experiment. For each (category, model_type), it:

1. Loads model predictions from temporal snapshots
2. Fetches Kalshi daily prices and trade history
3. Matches model nominee names to Kalshi market names
4. Estimates spreads from trade data
5. Runs buy-once backtests across a grid of trading parameter configs
6. Settles against known winners
7. Saves results to CSV

The key design decision is **buy-once**: each nominee gets at most one purchase
per season. If the model says BUY on Feb 1 and still says BUY on Feb 5, we do
NOT buy more. This is enforced by the BacktestEngine's position tracking.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260220_backtest_strategies.run_backtests --ceremony-year 2025

    # Fast mode (focused subset, ~20s):
    uv run python -m oscar_prediction_market.one_offs.\\
d20260220_backtest_strategies.run_backtests --ceremony-year 2025 --fast

Configuration is hardcoded — no CLI args needed beyond ceremony year.
All paths are relative to the experiment storage directory.
"""

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardsCalendar,
)
from oscar_prediction_market.data.oscar_winners import WINNERS_BY_YEAR
from oscar_prediction_market.data.schema import (
    OscarCategory,
)
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.one_offs.d20260220_backtest_strategies import (
    BACKTEST_MODEL_TYPES,
    ENSEMBLE_SHORT_NAME,
    MODELED_CATEGORIES,
)
from oscar_prediction_market.one_offs.d20260220_backtest_strategies.evaluation import (
    evaluate_model_accuracy,
    run_market_favorite_baseline,
)
from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestEngine,
)
from oscar_prediction_market.trading.backtest_configs import (
    generate_trading_configs,
)
from oscar_prediction_market.trading.name_matching import (
    validate_matching,
)
from oscar_prediction_market.trading.oscar_data import (
    build_market_prices,
    estimate_category_spreads,
    fetch_and_cache_market_data,
    get_winner_kalshi_name,
)
from oscar_prediction_market.trading.oscar_market import Candle
from oscar_prediction_market.trading.oscar_moments import (
    build_trading_moments,
)
from oscar_prediction_market.trading.oscar_prediction_source import (
    build_ensemble_source,
    build_model_source,
    load_nomination_dataset,
)
from oscar_prediction_market.trading.temporal_model import (
    SnapshotInfo,
    TemporalModel,
    get_snapshot_sequence,
    get_trading_dates,
)

logger = logging.getLogger(__name__)

# Eastern Time for display
_ET = ZoneInfo("America/New_York")

# Bankroll per category — consistent across all configs
BANKROLL = 1000.0


# ============================================================================
# Per-source backtest (shared by individual models and ensemble)
# ============================================================================


def run_single_backtest(
    source: TemporalModel,
    category: OscarCategory,
    trading_dates: list[date],
    snapshot_keys: list[SnapshotInfo],
    snapshot_availability: dict[str, datetime],
    matched_kalshi_names: set[str],
    winner_kalshi_name: str,
    market_prices_by_date: dict[str, dict[str, float]],
    spread_penalties: dict[str, float],
    mean_spread: float,
    hourly_candles: list[Candle] | None,
    use_hourly: bool,
    trading_configs: list[BacktestConfig],
    snapshot_key_strs: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run backtest for one source across the full config grid.

    This is the extracted core shared by both the individual-model loop
    and the ensemble path.  It:
    1. Computes model accuracy at each snapshot
    2. Computes model-vs-market divergence
    3. Builds TradingMoments from the source
    4. Runs the backtest engine across the config grid
    5. Settles against the known winner

    Returns:
        (pnl_rows, accuracy_rows, model_vs_market_rows)
    """
    cat_slug = category.slug
    source_name = source.name

    pnl_rows: list[dict] = []
    accuracy_rows: list[dict] = []
    model_vs_market_rows: list[dict] = []

    # --- Model accuracy at each snapshot ---
    for snap_key_str in snapshot_key_strs:
        avail_ts = snapshot_availability.get(snap_key_str)
        if avail_ts is None:
            continue
        try:
            preds = source.get_predictions(avail_ts)
        except KeyError:
            continue
        acc = evaluate_model_accuracy(preds, winner_kalshi_name)
        accuracy_rows.append(
            {
                "category": cat_slug,
                "model_type": source_name,
                "snapshot_key": snap_key_str,
                **acc,
            }
        )

        # Model vs market for this snapshot — use event date for price lookup
        snap_date_str = snap_key_str[:10]
        market_on_snap = market_prices_by_date.get(snap_date_str, {})
        if market_on_snap and preds:
            total_market = sum(market_on_snap.values())
            for name in matched_kalshi_names:
                model_p = preds.get(name, 0)
                market_p = market_on_snap.get(name, 0) if total_market > 0 else 0
                model_vs_market_rows.append(
                    {
                        "category": cat_slug,
                        "model_type": source_name,
                        "snapshot_key": snap_key_str,
                        "nominee": name,
                        "model_prob": round(model_p, 4),
                        "market_prob": round(market_p, 4),
                        "divergence": round(model_p - market_p, 4),
                        "is_winner": name == winner_kalshi_name,
                    }
                )

    # --- Build TradingMoment list ---
    moments = build_trading_moments(
        trading_dates=trading_dates,
        source=source,
        market_prices_by_date=market_prices_by_date,
        hourly_candles=hourly_candles,
        use_hourly=use_hourly,
    )

    print(f"    Trading moments: {len(moments)}")

    if not moments:
        return pnl_rows, accuracy_rows, model_vs_market_rows

    # --- Run across trading config grid ---
    for cfg in trading_configs:
        engine = BacktestEngine(cfg)
        result = engine.run(
            moments=moments,
            spread_penalties=spread_penalties if spread_penalties else None,
        )

        # Settle against known winner
        try:
            settlement = result.settle(winner_kalshi_name)
        except KeyError:
            settlement = None

        row = {
            "category": cat_slug,
            "model_type": source_name,
            "config_label": cfg.label,
            "fee_type": cfg.trading.fee_type.value,
            "kelly_fraction": cfg.trading.kelly.kelly_fraction,
            "buy_edge_threshold": cfg.trading.kelly.buy_edge_threshold,
            "min_price": cfg.trading.min_price,
            "kelly_mode": cfg.trading.kelly.kelly_mode.value,
            "bankroll_mode": cfg.simulation.bankroll_mode.value,
            "trading_side": cfg.label.split("_side=")[1],
            "total_trades": result.total_trades,
            "total_fees": round(result.total_fees_paid, 2),
            "n_snapshots_used": len(result.portfolio_history),
        }

        if settlement:
            row.update(
                {
                    "final_cash": round(settlement.final_cash, 2),
                    "total_pnl": round(settlement.total_pnl, 2),
                    "return_pct": settlement.return_pct,
                }
            )
        else:
            actual_cash = result.final_wealth
            actual_pnl = actual_cash - BANKROLL
            row.update(
                {
                    "final_cash": round(actual_cash, 2),
                    "total_pnl": round(actual_pnl, 2),
                    "return_pct": round(actual_pnl / BANKROLL * 100, 1),
                }
            )

        pnl_rows.append(row)

    return pnl_rows, accuracy_rows, model_vs_market_rows


# ============================================================================
# Per-category backtest
# ============================================================================


def run_category_backtest(
    category: OscarCategory,
    snapshot_keys: list[SnapshotInfo],
    lag_hours: float,
    ceremony_year: int,
    calendar: AwardsCalendar,
    winners: dict[OscarCategory, str],
    models_dir: Path,
    datasets_dir: Path,
    market_data_dir: Path,
    nominee_maps_dir: Path,
    model_types: list[ModelType] | None = None,
    fast: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Run backtest for one category across all model types and trading configs.

    Uses ``build_model_source()`` / ``build_ensemble_source()`` to construct
    ``TemporalModel`` instances, then delegates to ``run_single_backtest()``
    for each source.  Always runs an equal-weight ensemble in addition to
    individual models.

    Args:
        category: Oscar category to backtest.
        trading_configs: Grid of trading parameter combinations.
        snapshot_keys: Per-event snapshot keys from ``get_snapshot_keys()``.
        lag_hours: Hours after event end before signal is tradeable.
        ceremony_year: Ceremony year (e.g. 2025).
        calendar: Awards calendar for the ceremony year.
        winners: ``{category: winner_name}`` for settlement.
        models_dir: Root models directory.
        datasets_dir: Root datasets directory.
        market_data_dir: Market data cache directory.
        nominee_maps_dir: Directory to save nominee maps.
        model_types: Which models to run. Defaults to BACKTEST_MODEL_TYPES.

    Returns:
        (pnl_rows, accuracy_rows, model_vs_market_rows, spread_report)
    """
    cat_slug = category.slug
    snapshot_key_strs = [k.dir_name for k in snapshot_keys]
    trading_dates = get_trading_dates(calendar)

    if model_types is None:
        model_types = BACKTEST_MODEL_TYPES

    print(f"\n{'=' * 70}")
    print(f"Category: {category.name} ({cat_slug})")
    print(f"  Snapshots: {len(snapshot_keys)}")
    print(f"  Trading days: {len(trading_dates)}")
    print(f"{'=' * 70}")

    # Compute snapshot availability (once per category, shared across models)
    snapshot_availability = {
        s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
    }
    use_hourly = True  # always use hourly prices with inferred lag

    # Load nomination records & build name-mapping data (once per category)
    dataset = load_nomination_dataset(datasets_dir, category, snapshot_key_strs[0])

    # --- Fetch market data & estimate spreads ---
    fetch_result = fetch_and_cache_market_data(
        category,
        ceremony_year=ceremony_year,
        start_date=calendar.oscar_nominations_date_local,
        end_date=calendar.oscar_ceremony_date_local,
        market_data_dir=market_data_dir,
        fetch_hourly=use_hourly,
    )
    if fetch_result is None:
        return [], [], [], {}

    market, daily_candles, hourly_candles, trades_df = fetch_result
    spread_by_ticker, mean_spread = estimate_category_spreads(trades_df)
    trading_configs = generate_trading_configs(
        bankroll=BANKROLL,
        spread_penalty=mean_spread,
        fast=fast,
    )

    pnl_rows: list[dict] = []
    accuracy_rows: list[dict] = []
    model_vs_market_rows: list[dict] = []

    # --- Individual models ---
    for model_type in model_types:
        short_name = model_type.short_name
        print(f"\n  Model: {short_name} ({model_type.value})")

        result = build_model_source(
            category,
            model_type,
            market,
            snapshot_key_strs,
            snapshot_availability,
            models_dir=models_dir,
            dataset=dataset,
            ceremony_year=ceremony_year,
        )
        source, nominee_map = result

        # Save nominee map
        nominee_maps_dir.mkdir(parents=True, exist_ok=True)
        map_path = nominee_maps_dir / f"{cat_slug}_{short_name}_{ceremony_year}.json"
        map_path.write_text(json.dumps(nominee_map, indent=2))

        kalshi_names = list(market.nominee_tickers.keys())
        model_names = list(nominee_map.keys())
        validate_matching(nominee_map, model_names, kalshi_names, category, ceremony_year)

        # Get winner kalshi name
        matched_kalshi_names = set(nominee_map.values())
        try:
            winner_kalshi_name = get_winner_kalshi_name(category, matched_kalshi_names, winners)
        except ValueError as e:
            print(f"    ERROR: {e}")
            continue
        print(f"    Winner (kalshi name): {winner_kalshi_name}")

        # Spread penalties and prices mapped to kalshi names
        spread_penalties = {
            kn: spread_by_ticker[t]
            for kn in matched_kalshi_names
            if (t := market.nominee_tickers.get(kn)) and t in spread_by_ticker
        }
        market_prices_by_date = build_market_prices(daily_candles)

        pnl, acc, mvm = run_single_backtest(
            source=source,
            category=category,
            trading_dates=trading_dates,
            snapshot_keys=snapshot_keys,
            snapshot_availability=snapshot_availability,
            matched_kalshi_names=matched_kalshi_names,
            winner_kalshi_name=winner_kalshi_name,
            market_prices_by_date=market_prices_by_date,
            spread_penalties=spread_penalties,
            mean_spread=mean_spread,
            hourly_candles=hourly_candles,
            use_hourly=use_hourly,
            trading_configs=trading_configs,
            snapshot_key_strs=snapshot_key_strs,
        )
        pnl_rows.extend(pnl)
        accuracy_rows.extend(acc)
        model_vs_market_rows.extend(mvm)

    # --- Ensemble model (always run) ---
    if model_types:
        print(f"\n  Model: {ENSEMBLE_SHORT_NAME} (equal-weight avg of {len(model_types)} models)")

        ens_result = build_ensemble_source(
            category,
            model_types,
            market,
            snapshot_key_strs,
            snapshot_availability,
            models_dir=models_dir,
            dataset=dataset,
            ceremony_year=ceremony_year,
        )
        ens_source, ens_nominee_map = ens_result
        kalshi_names = list(market.nominee_tickers.keys())
        model_names = list(ens_nominee_map.keys())
        validate_matching(ens_nominee_map, model_names, kalshi_names, category, ceremony_year)

        ens_matched = set(ens_nominee_map.values())
        try:
            winner_kalshi_name = get_winner_kalshi_name(category, ens_matched, winners)
        except ValueError as e:
            print(f"    ERROR: {e}")
            winner_kalshi_name = None  # type: ignore[assignment]  # intentional fallback

        if winner_kalshi_name:
            print(f"    Winner (kalshi name): {winner_kalshi_name}")
            spread_penalties = {
                kn: spread_by_ticker[t]
                for kn in ens_matched
                if (t := market.nominee_tickers.get(kn)) and t in spread_by_ticker
            }
            market_prices_by_date = build_market_prices(daily_candles)

            pnl, acc, mvm = run_single_backtest(
                source=ens_source,
                category=category,
                trading_dates=trading_dates,
                snapshot_keys=snapshot_keys,
                snapshot_availability=snapshot_availability,
                matched_kalshi_names=ens_matched,
                winner_kalshi_name=winner_kalshi_name,
                market_prices_by_date=market_prices_by_date,
                spread_penalties=spread_penalties,
                mean_spread=mean_spread,
                hourly_candles=hourly_candles,
                use_hourly=use_hourly,
                trading_configs=trading_configs,
                snapshot_key_strs=snapshot_key_strs,
            )
            pnl_rows.extend(pnl)
            accuracy_rows.extend(acc)
            model_vs_market_rows.extend(mvm)

    return (
        pnl_rows,
        accuracy_rows,
        model_vs_market_rows,
        {
            "category": cat_slug,
            "mean_spread": mean_spread,
            "spread_by_ticker": {k: round(v, 4) for k, v in spread_by_ticker.items()},
        },
    )


# ============================================================================
# Main
# ============================================================================


# Categories and models for --fast mode: most liquid / most interesting
_FAST_CATEGORIES = [
    OscarCategory.BEST_PICTURE,
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
]
_FAST_MODEL_TYPES = [
    ModelType.CALIBRATED_SOFTMAX_GBT,
    ModelType.LOGISTIC_REGRESSION,
]


def main() -> None:
    """Run the full backtest pipeline."""
    parser = argparse.ArgumentParser(description="Run multi-category backtests")
    parser.add_argument(
        "--ceremony-year",
        type=int,
        required=True,
        help="Ceremony year to backtest (e.g. 2025).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Category slugs to run (e.g. best_picture directing). Default: all.",
    )
    parser.add_argument(
        "--lag-hours",
        type=float,
        default=6.0,
        help=(
            "Hours after event end before signal is available for trading. "
            "Default: 6.0 (recommended based on timing leakage audit)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results. Default: derived from ceremony year.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Fast mode: run a focused subset of configs (18), categories (3), "
            "and models (2) for quick iteration. ~100x faster than full grid."
        ),
    )
    args = parser.parse_args()

    ceremony_year: int = args.ceremony_year

    # Derive year-specific paths and data
    calendar: AwardsCalendar = CALENDARS[ceremony_year]
    winners = WINNERS_BY_YEAR[ceremony_year]

    exp_dir = Path("storage/d20260220_backtest_strategies")
    models_dir = exp_dir / str(ceremony_year) / "models"
    datasets_dir = exp_dir / str(ceremony_year) / "datasets"
    results_dir = (
        Path(args.results_dir) if args.results_dir else exp_dir / str(ceremony_year) / "results"
    )
    market_data_dir = exp_dir / "market_data"
    nominee_maps_dir = exp_dir / "nominee_maps"

    lag_hours: float = args.lag_hours
    delay_label = f"inferred lag +{lag_hours}h"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine which categories and models to run
    if args.fast:
        default_categories = _FAST_CATEGORIES
        model_types: list[ModelType] | None = _FAST_MODEL_TYPES
    else:
        default_categories = list(MODELED_CATEGORIES)
        model_types = None  # = BACKTEST_MODEL_TYPES (default in run_category_backtest)

    if args.categories:
        categories = [cat for cat in MODELED_CATEGORIES if cat.slug in args.categories]
        if not categories:
            print(f"ERROR: No matching categories for {args.categories}")
            return
    else:
        categories = default_categories

    print("=" * 70)
    print(f"Multi-Category Backtest: {ceremony_year} Ceremony")
    print("=" * 70)
    print(f"Categories: {[c.slug for c in categories]}")
    print(f"Signal delay: {delay_label}")
    if args.fast:
        model_names = [m.short_name for m in (model_types or BACKTEST_MODEL_TYPES)]
        print(f"Fast mode: reduced config grid, models={model_names}")

    # Derive snapshot keys (per-event)
    snapshot_keys = get_snapshot_sequence(calendar)
    print(f"\nPer-event snapshots ({len(snapshot_keys)}):")
    for key in snapshot_keys:
        print(f"  {key.dir_name}")

    # Show snapshot availability
    snapshot_availability = {
        s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
    }
    print(f"\nSignal delay mode: {delay_label}")
    print("  Snapshot availability:")
    for key in snapshot_keys:
        avail = snapshot_availability[key.dir_name]
        avail_et = avail.astimezone(_ET)
        print(f"    {key.dir_name} → available {avail_et.strftime('%Y-%m-%d %H:%M ET')}")

    print("\nTrading config grid: category-specific shared grid")

    # Run backtests per category
    all_pnl: list[dict] = []
    all_accuracy: list[dict] = []
    all_model_vs_market: list[dict] = []
    all_spread_reports: list[dict] = []
    all_baseline: list[dict] = []

    for category in categories:
        pnl, acc, mvm, spread = run_category_backtest(
            category,
            snapshot_keys,
            lag_hours=lag_hours,
            ceremony_year=ceremony_year,
            calendar=calendar,
            winners=winners,
            models_dir=models_dir,
            datasets_dir=datasets_dir,
            market_data_dir=market_data_dir,
            nominee_maps_dir=nominee_maps_dir,
            model_types=model_types,
            fast=args.fast,
        )
        all_pnl.extend(pnl)
        all_accuracy.extend(acc)
        all_model_vs_market.extend(mvm)
        if spread:
            all_spread_reports.append(spread)

        # Market favorite baseline
        baseline = run_market_favorite_baseline(category, winners=winners, calendar=calendar)
        all_baseline.extend(baseline)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)

    if all_pnl:
        pnl_df = pd.DataFrame(all_pnl)
        pnl_df.to_csv(results_dir / "daily_pnl.csv", index=False)
        print(f"\nSaved {len(all_pnl)} P&L rows to daily_pnl.csv")

    if all_accuracy:
        acc_df = pd.DataFrame(all_accuracy)
        acc_df.to_csv(results_dir / "model_accuracy.csv", index=False)
        print(f"Saved {len(all_accuracy)} accuracy rows to model_accuracy.csv")

    if all_model_vs_market:
        mvm_df = pd.DataFrame(all_model_vs_market)
        mvm_df.to_csv(results_dir / "model_vs_market.csv", index=False)
        print(f"Saved {len(all_model_vs_market)} model-vs-market rows")

    if all_spread_reports:
        spread_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "spread_by_ticker"} for r in all_spread_reports]
        )
        spread_df.to_csv(results_dir / "spread_report.csv", index=False)
        with open(results_dir / "spread_detail.json", "w") as f:
            json.dump(all_spread_reports, f, indent=2)
        print(f"Saved spread reports for {len(all_spread_reports)} categories")

    if all_baseline:
        baseline_df = pd.DataFrame(all_baseline)
        baseline_df.to_csv(results_dir / "market_favorite_baseline.csv", index=False)
        print(f"Saved {len(all_baseline)} baseline rows")

    # Summary
    if all_pnl:
        pnl_df = pd.DataFrame(all_pnl)
        print("\n" + "=" * 70)
        print("P&L Summary (best config per model x category)")
        print("=" * 70)
        if "total_pnl" in pnl_df.columns:
            best = pnl_df.loc[pnl_df.groupby(["category", "model_type"])["total_pnl"].idxmax()]
            for _, row in best.iterrows():
                print(
                    f"  {row['category']:<25} {row['model_type']:<10} "
                    f"P&L=${row['total_pnl']:>+8.2f} "
                    f"({row['return_pct']:>+6.1f}%) "
                    f"trades={row['total_trades']}"
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
