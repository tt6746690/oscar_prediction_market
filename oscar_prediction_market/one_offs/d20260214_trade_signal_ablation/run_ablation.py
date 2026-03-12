"""Trade signal backtest with parameter ablation grid sweep.

Runs the backtest pipeline over a grid of trading parameters to find
optimal configuration. Reuses core simulation logic from the d20260214
backtest but adds:

- Grid sweep over fee_type, kelly_fraction, min_edge, sell_threshold,
  min_price
- Model weighting (LR, GBT, average, market blend)
- Parallel execution across configs
- Aggregated results across all configs

The model predictions come from temporal snapshots built with best configs
(lr_standard + wide grid + thresh 0.80, gbt_standard + gbt grid) — see
build_models.sh.

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260214_trade_signal_ablation.run_ablation \
        --configs-dir storage/d20260214_trade_signal_ablation/configs \
        --output-dir storage/d20260214_trade_signal_ablation/results \
        --snapshots-dir storage/d20260214_trade_signal_ablation/models \
        [--n-workers 4]
"""

import argparse
import json
import logging
import multiprocessing
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.one_offs.d20260214_trade_signal_ablation.price_helpers import (
    build_prices_by_date,
    deserialize_daily_prices,
    serialize_daily_prices,
)
from oscar_prediction_market.one_offs.d20260214_trade_signal_backtest.generate_signals import (
    BacktestConfig,
)
from oscar_prediction_market.one_offs.legacy_snapshot_loading import (
    estimate_spread_penalties,
    fetch_daily_prices,
    get_snapshot_dates,
    load_weighted_predictions,
)
from oscar_prediction_market.trading.backtest import (
    BacktestEngine,
    MarketSnapshot,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import (
    OscarMarket,
)
from oscar_prediction_market.trading.schema import BankrollMode

logger = logging.getLogger(__name__)


def _dicts_to_moments(
    predictions_by_date: dict[str, dict[str, float]],
    prices_by_date: dict[str, dict[str, float]],
) -> list[MarketSnapshot]:
    """Convert date-keyed dicts to sorted MarketSnapshot list (local helper)."""
    common_dates = sorted(set(predictions_by_date) & set(prices_by_date))
    return [
        MarketSnapshot(
            timestamp=datetime.combine(date.fromisoformat(d), datetime.min.time()),
            predictions=predictions_by_date[d],
            prices=prices_by_date[d],
        )
        for d in common_dates
    ]


# ============================================================================
# Single Config Runner
# ============================================================================


def run_single_config(
    config: BacktestConfig,
    daily_prices: pd.DataFrame,
    spread_penalties: dict[str, float],
    median_spread: float,
    snapshot_dates: list[str],
    models_dir: Path,
    config_id: str,
) -> dict[str, Any]:
    """Run backtest for a single config, returning serializable results.

    This is the core function called in parallel for each config in the grid.
    Delegates to BacktestEngine for the actual simulation.
    """
    model_type = config.model_types[0]  # Grid configs have exactly one model type
    # Grid configs should always have a concrete mode (never "both")
    raw_mode = config.bankroll_mode
    if raw_mode not in ("fixed", "dynamic"):
        raise ValueError(f"Grid configs must use 'fixed' or 'dynamic', got '{raw_mode}'")
    bankroll_mode = BankrollMode(raw_mode)

    # Build prices_by_date lookup
    prices_by_date = build_prices_by_date(daily_prices, snapshot_dates)

    # Load predictions for all snapshot dates
    predictions_by_date: dict[str, dict[str, float]] = {}
    for snap_date in snapshot_dates:
        market_prices = prices_by_date.get(snap_date)
        preds = load_weighted_predictions(
            models_dir,
            snap_date,
            model_type,
            market_prices,
            market_blend_alpha=config.market_blend_alpha,
            normalize_probabilities=config.normalize_probabilities,
        )
        if preds:
            predictions_by_date[snap_date] = preds

    # Build engine config
    engine_config = config.to_engine_config(
        spread_penalty=median_spread,
        bankroll_mode=bankroll_mode,
    )

    engine = BacktestEngine(engine_config)
    moments = _dicts_to_moments(predictions_by_date, prices_by_date)
    result = engine.run(
        moments=moments,
        spread_penalties=spread_penalties if spread_penalties else None,
    )

    # Build serializable snapshots
    snapshots: list[dict[str, Any]] = []
    running_fees = 0.0
    running_trades = 0
    for snap in result.portfolio_history:
        fills_at_snap = [f for f in result.trade_log if f.timestamp == snap.timestamp]
        running_trades += len(fills_at_snap)
        running_fees += sum(f.fee_dollars for f in fills_at_snap)
        snapshots.append(
            {
                "snapshot_date": snap.timestamp.isoformat(),
                "cash": snap.cash,
                "mark_to_market_value": snap.mark_to_market_value,
                "total_wealth": snap.total_wealth,
                "total_fees_paid": round(running_fees, 2),
                "total_trades": running_trades,
                "n_positions": snap.n_positions,
            }
        )

    # Summary metrics from engine result
    return {
        "config_id": config_id,
        "config": config.model_dump(),
        "model_type": model_type,
        "bankroll_mode": bankroll_mode,
        "fee_type": config.fee_type,
        "kelly_fraction": config.kelly_fraction,
        "min_edge": config.min_edge,
        "sell_edge_threshold": config.sell_edge_threshold,
        "min_price": config.min_price,
        "kelly_mode": config.kelly_mode,
        "market_blend_alpha": config.market_blend_alpha,
        "normalize_probabilities": config.normalize_probabilities,
        "final_wealth": round(result.final_wealth, 2),
        "total_return_pct": result.total_return_pct,
        "total_fees_paid": round(result.total_fees_paid, 2),
        "total_trades": result.total_trades,
        "n_snapshots": len(result.portfolio_history),
        "snapshots": snapshots,
        # Serialize SettlementResult objects to plain dicts for JSON output.
        "settlements": {k: v.model_dump() for k, v in result.settlements.items()},
    }


def _worker(args: tuple[str, dict, str]) -> dict[str, Any]:
    """Worker function for parallel execution."""
    config_id, config_dict, shared_data_path = args
    # Load shared data
    with open(shared_data_path) as f:
        shared = json.load(f)

    config = BacktestConfig.model_validate(config_dict)
    daily_prices = deserialize_daily_prices(shared["daily_prices"])

    return run_single_config(
        config=config,
        daily_prices=daily_prices,
        spread_penalties=shared["spread_penalties"],
        median_spread=shared["median_spread"],
        snapshot_dates=shared["snapshot_dates"],
        models_dir=Path(shared["models_dir"]),
        config_id=config_id,
    )


# ============================================================================
# Grid Runner
# ============================================================================


def run_grid(
    configs_dir: Path,
    output_dir: Path,
    snapshots_dir: str,
    price_start_date: str = "2025-12-01",
    price_end_date: str = "2026-02-15",
    n_workers: int | None = None,
) -> None:
    """Run backtest over all configs in configs_dir.

    1. Load and cache shared data (prices, spreads, snapshot dates)
    2. Load all config JSON files
    3. Run each config (optionally in parallel)
    4. Save aggregated results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load shared data
    models_dir = Path(snapshots_dir) / "models"
    logger.info("Loading shared data...")

    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_data.nominee_tickers)
    daily_prices = fetch_daily_prices(
        mkt,
        start_date=date.fromisoformat(price_start_date),
        end_date=date.fromisoformat(price_end_date),
    )

    # Estimate spreads
    spread_penalties, median_spread = estimate_spread_penalties(
        mkt,
        start_date=date.fromisoformat(price_start_date),
        end_date=date.fromisoformat(price_end_date),
    )

    # Get snapshot dates from LR model dir
    snapshot_dates = get_snapshot_dates(models_dir, "lr")
    logger.info("  %d snapshot dates", len(snapshot_dates))

    # 2. Load configs
    config_files = sorted(configs_dir.glob("*.json"))
    logger.info("  %d config files", len(config_files))

    configs: list[tuple[str, dict]] = []
    for cf in config_files:
        with open(cf) as f:
            config_dict = json.load(f)
        config_dict["snapshots_dir"] = snapshots_dir
        configs.append((cf.stem, config_dict))

    # 3. Save shared data for workers
    shared_data_path = output_dir / "_shared_data.json"
    with open(shared_data_path, "w") as f:
        json.dump(
            {
                "daily_prices": serialize_daily_prices(daily_prices),
                "spread_penalties": spread_penalties,
                "median_spread": median_spread,
                "snapshot_dates": snapshot_dates,
                "models_dir": str(models_dir),
            },
            f,
        )

    # 4. Run configs
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(configs))
    n_workers = max(1, n_workers)

    worker_args = [
        (config_id, config_dict, str(shared_data_path)) for config_id, config_dict in configs
    ]

    logger.info("Running %d configs with %d workers...", len(configs), n_workers)

    if n_workers == 1:
        results = [_worker(args) for args in worker_args]
    else:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_worker, worker_args)

    # 5. Save results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump({"n_configs": len(results), "results": results}, f, indent=2, default=str)
    logger.info("Saved %d results to %s", len(results), results_path)

    # 6. Save summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "config_id": r["config_id"],
                "model_type": r["model_type"],
                "fee_type": r["fee_type"],
                "kelly_fraction": r["kelly_fraction"],
                "min_edge": r["min_edge"],
                "sell_edge_threshold": r["sell_edge_threshold"],
                "min_price": r["min_price"],
                "kelly_mode": r.get("kelly_mode", "multi_outcome"),
                "market_blend_alpha": r.get("market_blend_alpha"),
                "normalize_probabilities": r.get("normalize_probabilities", False),
                "bankroll_mode": r["bankroll_mode"],
                "final_wealth": r["final_wealth"],
                "total_return_pct": r["total_return_pct"],
                "total_fees_paid": r["total_fees_paid"],
                "total_trades": r["total_trades"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    # Print top/bottom results
    if not summary_df.empty:
        print(f"\n{'=' * 80}")
        print(f"Ablation Results Summary ({len(results)} configs)")
        print(f"{'=' * 80}")

        sorted_df = summary_df.sort_values("total_return_pct", ascending=False)
        print("\nTop 10 configs:")
        print(sorted_df.head(10).to_string(index=False))

        print("\nBottom 5 configs:")
        print(sorted_df.tail(5).to_string(index=False))

        print(f"\nMean return: {summary_df['total_return_pct'].mean():.1f}%")
        print(f"Median return: {summary_df['total_return_pct'].median():.1f}%")
        print(f"Best return: {summary_df['total_return_pct'].max():.1f}%")
        print(f"Worst return: {summary_df['total_return_pct'].min():.1f}%")
        print(f"Profitable configs: {(summary_df['total_return_pct'] > 0).sum()}/{len(summary_df)}")

    # Cleanup
    shared_data_path.unlink(missing_ok=True)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run trade signal ablation grid sweep")
    parser.add_argument("--configs-dir", type=str, required=True, help="Dir with config JSONs")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="storage/d20260214_trade_signal_ablation",
        help="Dir with temporal model snapshots",
    )
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--price-start-date", type=str, default="2025-12-01")
    parser.add_argument("--price-end-date", type=str, default="2026-02-15")
    args = parser.parse_args()

    run_grid(
        configs_dir=Path(args.configs_dir),
        output_dir=Path(args.output_dir),
        snapshots_dir=args.snapshots_dir,
        price_start_date=args.price_start_date,
        price_end_date=args.price_end_date,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
