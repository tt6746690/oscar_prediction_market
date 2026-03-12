"""Backtest trade signal pipeline over temporal model snapshots.

Loads model predictions from each snapshot date, fetches historical Kalshi
prices, and simulates trading using the edge/kelly/signals pipeline.

Now delegates to the reusable backtest framework in
:mod:`trading.backtest`. This script handles:
- CLI interface and config loading
- Price and spread data fetching (Oscar-specific)
- Printing formatted signal reports
- Orchestrating the backtest engine with the fetched data

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260214_trade_signal_backtest.generate_signals \
        --config storage/d20260214_trade_signal_backtest/config.json \
        --output-dir storage/d20260214_trade_signal_backtest
"""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.one_offs.d20260214_trade_signal_ablation.price_helpers import (
    build_prices_by_date,
)
from oscar_prediction_market.one_offs.legacy_snapshot_loading import (
    estimate_spread_penalties,
    fetch_daily_prices,
    get_snapshot_dates,
    load_all_predictions_for_snapshots,
)
from oscar_prediction_market.trading.backtest import (
    BacktestConfig as _BacktestConfig,
)
from oscar_prediction_market.trading.backtest import (
    BacktestEngine,
    BacktestSimulationConfig,
    MarketSnapshot,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import OscarMarket
from oscar_prediction_market.trading.schema import (
    BankrollMode,
    FeeType,
    KellyMode,
    PositionDirection,
    TradingConfig,
)
from oscar_prediction_market.trading.signals import TradeSignal

logger = logging.getLogger(__name__)


# ============================================================================
# Printing Helpers
# ============================================================================


def _dicts_to_moments(
    predictions_by_date: dict[str, dict[str, float]],
    prices_by_date: dict[str, dict[str, float]],
) -> list[MarketSnapshot]:
    """Convert legacy dict-of-dicts to MarketSnapshot list (inlined from backtest.py)."""
    common_dates = sorted(set(predictions_by_date) & set(prices_by_date))
    return [
        MarketSnapshot(
            timestamp=datetime.combine(date.fromisoformat(d_str), datetime.min.time()),
            predictions=predictions_by_date[d_str],
            prices=prices_by_date[d_str],
        )
        for d_str in common_dates
    ]


def print_signal_report(signals: list[TradeSignal]) -> None:
    """Print a formatted signal list to stdout."""
    print(f"\n{'=' * 90}")
    print("Trade Signal Report")
    print(f"{'=' * 90}")
    print()

    if not signals:
        print("  No signals generated.")
        return

    # Active signals (non-HOLD)
    active = [s for s in signals if s.action != "hold"]
    if active:
        print(
            f"  {'Action':<6} {'Outcome':<30} {'Model':>6} {'Price':>6} "
            f"{'Edge':>7} {'Curr':>5} {'Tgt':>5} {'Δ':>5} {'$Δ':>8}  Reason"
        )
        print(f"  {'-' * 88}")
        for s in active:
            print(
                f"  {s.action:<6} {s.outcome:<30} {s.model_prob:>5.1%} "
                f"{s.execution_price * 100:>5.0f}¢ {s.net_edge:>+6.1%} "
                f"{s.current_contracts:>5} {s.target_contracts:>5} "
                f"{s.delta_contracts:>+5} ${s.outlay_dollars:>+7.2f}  {s.reason}"
            )
    else:
        print("  No active trades — all HOLD.")

    # Held positions (HOLD with contracts > 0)
    holds = [s for s in signals if s.action == "hold" and s.current_contracts > 0]
    if holds:
        print("\n  Held positions:")
        for s in holds:
            print(
                f"    {s.outcome:<30} {s.current_contracts} contracts  "
                f"edge={s.net_edge:>+.1%}  {s.reason}"
            )

    print(f"\n  Active trades: {len(active)}")


# ============================================================================
# Configuration (one-off specific — wraps library BacktestConfig)
# ============================================================================


class BacktestConfig(BaseModel):
    """Configuration for the trade signal backtest.

    This is the one-off config that wraps the library BacktestConfig
    with additional fields for data loading (snapshots_dir, price dates,
    model_types, spread estimation, etc.).
    """

    model_config = {"extra": "forbid"}

    bankroll_dollars: float = Field(default=1000, gt=0)
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    min_edge: float = Field(default=0.05, ge=0)
    max_position_per_outcome_dollars: float = Field(default=250, gt=0)
    max_total_exposure_dollars: float = Field(default=500, gt=0)
    sell_edge_threshold: float = Field(default=-0.03)
    spread_penalty_mode: Literal["fixed", "trade_data"] = Field(
        default="trade_data",
        description="How to estimate spread penalty",
    )
    fixed_spread_penalty: float = Field(
        default=0.02, ge=0, description="Penalty each way if mode='fixed' (dollars)"
    )
    model_types: list[str] = Field(default=["lr", "gbt", "average"])
    snapshots_dir: str = Field(default="storage/d20260211_temporal_model_snapshots")
    price_start_date: str = Field(default="2025-12-01")
    price_end_date: str = Field(default="2026-02-14")
    bankroll_mode: Literal["fixed", "dynamic", "both"] = Field(
        default="both",
        description="Bankroll evolution strategy",
    )
    fee_type: FeeType = Field(
        default=FeeType.TAKER,
        description="Fee schedule: 'taker' (7%) or 'maker' (1.75%)",
    )
    min_price: float = Field(
        default=0,
        ge=0,
        description="Skip contracts below this price (dollars).",
    )
    market_blend_alpha: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="If set, blend model with market: P = α * P_market + (1-α) * P_model.",
    )
    normalize_probabilities: bool = Field(
        default=False,
        description="Normalize model probabilities to sum to 1.",
    )
    kelly_mode: KellyMode = Field(
        default=KellyMode.MULTI_OUTCOME,
        description="Kelly sizing mode: 'independent' or 'multi_outcome'.",
    )

    def to_engine_config(
        self, spread_penalty: float, bankroll_mode: BankrollMode
    ) -> _BacktestConfig:
        """Convert to library BacktestConfig for a specific bankroll mode."""
        from oscar_prediction_market.trading.schema import KellyConfig

        return _BacktestConfig(
            trading=TradingConfig(
                kelly=KellyConfig(
                    bankroll=self.bankroll_dollars,
                    kelly_fraction=self.kelly_fraction,
                    kelly_mode=self.kelly_mode,
                    buy_edge_threshold=self.min_edge,
                    max_position_per_outcome=self.max_position_per_outcome_dollars,
                    max_total_exposure=self.max_total_exposure_dollars,
                ),
                sell_edge_threshold=self.sell_edge_threshold,
                fee_type=self.fee_type,
                limit_price_offset=0.0,
                min_price=self.min_price,
                allowed_directions=frozenset({PositionDirection.YES}),
            ),
            simulation=BacktestSimulationConfig(
                spread_penalty=spread_penalty,
                bankroll_mode=bankroll_mode,
            ),
        )


# ============================================================================
# Main Backtest
# ============================================================================


def run_backtest(config: BacktestConfig, output_dir: Path) -> dict:
    """Run the full backtest over all snapshots and model types.

    Fetches data, then delegates to BacktestEngine for each
    model_type × bankroll_mode combination.
    """
    models_dir = Path(config.snapshots_dir) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch price data
    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_data.nominee_tickers)
    daily_prices = fetch_daily_prices(
        mkt,
        start_date=date.fromisoformat(config.price_start_date),
        end_date=date.fromisoformat(config.price_end_date),
    )

    # 2. Estimate spreads from trade data
    if config.spread_penalty_mode == "trade_data":
        spread_penalties, median_spread = estimate_spread_penalties(
            mkt,
            start_date=date.fromisoformat(config.price_start_date),
            end_date=date.fromisoformat(config.price_end_date),
        )
    else:
        spread_penalties = {}
        median_spread = config.fixed_spread_penalty

    # 3. Get snapshot dates
    base_model_types = [mt for mt in config.model_types if mt != "average"]
    if not base_model_types:
        base_model_types = ["lr", "gbt"]
    snapshot_dates = get_snapshot_dates(models_dir, base_model_types[0])
    logger.info("Found %d snapshot dates: %s", len(snapshot_dates), snapshot_dates)

    # 4. Build prices by date
    prices_by_date = build_prices_by_date(daily_prices, snapshot_dates)

    # Build ticker map
    ticker_map: dict[str, str] = dict(bp_data.nominee_tickers)

    # 5. Run backtest for each model type × bankroll mode
    all_results: dict = {"config": config.model_dump(), "backtests": {}}

    bankroll_modes: list[BankrollMode] = (
        [BankrollMode.FIXED, BankrollMode.DYNAMIC]
        if config.bankroll_mode == "both"
        else [BankrollMode(config.bankroll_mode)]
    )

    for model_type in config.model_types:
        # Load all predictions for this model type
        predictions_by_date = load_all_predictions_for_snapshots(
            models_dir=models_dir,
            model_type=model_type,
            snapshot_dates=snapshot_dates,
            market_prices_by_date=prices_by_date,
            market_blend_alpha=config.market_blend_alpha,
            normalize_probabilities=config.normalize_probabilities,
        )

        for bankroll_mode in bankroll_modes:
            run_key = f"{model_type}_{bankroll_mode}"
            logger.info("\n== Backtest: model=%s, bankroll=%s ==", model_type, bankroll_mode)

            # Build engine config
            engine_config = config.to_engine_config(
                spread_penalty=median_spread,
                bankroll_mode=bankroll_mode,
            )

            engine = BacktestEngine(engine_config)
            moments = _dicts_to_moments(predictions_by_date, prices_by_date)
            result = engine.run(
                moments=moments,
                spread_penalties=(
                    spread_penalties if config.spread_penalty_mode == "trade_data" else None
                ),
                ticker_map=ticker_map,
            )

            # Serialize to JSON-compatible dicts
            snapshot_records = []
            running_fees = 0.0
            running_trades = 0
            for snap in result.portfolio_history:
                fills_at_snap = [f for f in result.trade_log if f.timestamp == snap.timestamp]
                running_trades += len(fills_at_snap)
                running_fees += sum(f.fee_dollars for f in fills_at_snap)
                record = {
                    "snapshot_date": snap.timestamp.isoformat(),
                    "model_type": model_type,
                    "bankroll_mode": bankroll_mode,
                    "cash": snap.cash,
                    "mark_to_market_value": snap.mark_to_market_value,
                    "total_wealth": snap.total_wealth,
                    "total_fees_paid": round(running_fees, 2),
                    "total_trades": running_trades,
                    "positions": {
                        p.outcome: {"contracts": p.contracts, "avg_cost": p.avg_cost}
                        for p in snap.positions
                    },
                }
                snapshot_records.append(record)

            all_results["backtests"][run_key] = {
                "snapshots": snapshot_records,
                # Serialize SettlementResult objects to plain dicts so downstream
                # analysis scripts can read them as JSON without Pydantic.
                "settlements": {k: v.model_dump() for k, v in result.settlements.items()},
            }

    # 6. Save results
    results_path = output_dir / "backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", results_path)

    return all_results


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Backtest trade signal pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to backtest config JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = BacktestConfig.model_validate_json(f.read())
    else:
        logger.info("Config not found at %s, using defaults", config_path)
        config = BacktestConfig()

    output_dir = Path(args.output_dir)
    run_backtest(config, output_dir)


if __name__ == "__main__":
    main()
