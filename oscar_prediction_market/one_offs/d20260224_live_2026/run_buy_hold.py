"""Run buy-and-hold backtests for 2026 live predictions.

Adapts the buy-hold backtest from d20260225 for live 2026 usage where
**the winner is unknown**.  Instead of settling against a known winner,
outputs scenario P&L for every possible winner via ``result.settlements``.

Uses the single recommended config (avg_ensemble) from the 2024–2025
cross-year robustness analysis.  All 6 models run with identical trading
parameters for comparison; only avg_ensemble is used for actual trades.

Outputs:
    scenario_pnl.csv
        One row per (category, model, entry_point, config, assumed_winner).
        Shows what the P&L would be if each nominee wins.
    model_vs_market.csv
        Model probabilities vs market prices per entry point.
    position_summary.csv
        Positions held per (category, model, entry_point, config).

Usage::

    cd "$(git rev-parse --show-toplevel)"

    # Run recommended config:
    uv run python -m oscar_prediction_market.one_offs.\\
d20260224_live_2026.run_buy_hold

    # Run specific categories:
    uv run python -m oscar_prediction_market.one_offs.\\
d20260224_live_2026.run_buy_hold --categories best_picture directing
"""

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.one_offs.d20260224_live_2026 import (
    AVAILABLE_SNAPSHOT_KEYS,
    BACKTEST_MODEL_TYPES,
)
from oscar_prediction_market.one_offs.d20260224_live_2026.recommended_configs import (
    RECOMMENDED_CONFIGS,
    get_recommended_configs,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    YEAR_CONFIGS,
    YearConfig,
)
from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestEngine,
    MarketSnapshot,
)
from oscar_prediction_market.trading.kalshi_client import KalshiPublicClient
from oscar_prediction_market.trading.name_matching import (
    match_nominees,
    validate_matching,
)
from oscar_prediction_market.trading.oscar_data import (
    build_market_prices,
    estimate_category_spreads,
    fetch_and_cache_market_data,
)
from oscar_prediction_market.trading.oscar_market import OscarMarket
from oscar_prediction_market.trading.oscar_moments import build_entry_moment
from oscar_prediction_market.trading.oscar_prediction_source import (
    load_all_snapshot_predictions,
    load_ensemble_predictions,
    load_nomination_dataset,
    translate_predictions,
)
from oscar_prediction_market.trading.schema import (
    PositionDirection,
)
from oscar_prediction_market.trading.temporal_model import SnapshotInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Default delay: inferred lag +6h (same as cross-year analysis)
DEFAULT_LAG_HOURS = 6.0

#: Per-category bankroll.
BANKROLL = 1000.0

#: Ensemble definitions — matches d20260225 but defined locally.
ALL_ENSEMBLES: list[tuple[str, list[ModelType]]] = [
    ("avg_ensemble", list(BACKTEST_MODEL_TYPES)),
    ("clogit_cal_sgbt_ensemble", [ModelType.CONDITIONAL_LOGIT, ModelType.CALIBRATED_SOFTMAX_GBT]),
]

# ============================================================================
# Model short name → list of configs
# ============================================================================


def get_configs_for_model(
    short_name: str,
    centers_only: bool = False,
    recommended_configs: dict[str, BacktestConfig] | None = None,
) -> list[BacktestConfig]:
    """Get the trading configs to test with a given model.

    All models run with the single recommended config. The ``centers_only``
    parameter is kept for backward compatibility but is now a no-op.
    """
    configs_by_option = recommended_configs or RECOMMENDED_CONFIGS
    return list(configs_by_option.values())


# ============================================================================
# Ensemble label → model types mapping
# ============================================================================

#: Map ensemble labels to their model types for easy lookup.
_ENSEMBLE_MAP: dict[str, list[ModelType]] = dict(ALL_ENSEMBLES)


# ============================================================================
# Live orderbook pricing
# ============================================================================


def fetch_live_prices(
    market: OscarMarket,
    matched_kalshi_names: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Fetch current orderbook mid-prices and spreads for all nominees.

    Uses the Kalshi orderbook API to get current bid/ask levels.
    Mid-price = (best_yes_bid + (100 - best_no_bid)) / 2 / 100.
    Half-spread = ((100 - best_no_bid) - best_yes_bid) / 2 / 100.

    Args:
        market: OscarMarket instance with nominee_tickers.
        matched_kalshi_names: Set of Kalshi names to fetch prices for.

    Returns:
        (prices, spreads): dicts keyed by kalshi name, values in dollars.
    """
    client = KalshiPublicClient()
    nominee_tickers = market.nominee_tickers

    prices: dict[str, float] = {}
    spreads: dict[str, float] = {}

    for kalshi_name in matched_kalshi_names:
        ticker = nominee_tickers.get(kalshi_name)
        if ticker is None:
            print(f"    WARNING: No ticker for '{kalshi_name}', using defaults")
            prices[kalshi_name] = 0.50
            spreads[kalshi_name] = 0.02
            continue

        ob = client.get_orderbook(ticker, depth=5)
        # Kalshi returns bids in ascending price order; best bid = highest = last
        best_yes_bid = ob.yes[-1][0] if ob.yes else 0
        best_no_bid = ob.no[-1][0] if ob.no else 0

        if best_yes_bid > 0 and best_no_bid > 0:
            # Both sides present
            best_yes_ask = 100 - best_no_bid
            mid = (best_yes_bid + best_yes_ask) / 2 / 100
            half_spread = (best_yes_ask - best_yes_bid) / 2 / 100
        elif best_yes_bid > 0:
            # Only YES bids — estimate mid from bid
            mid = best_yes_bid / 100
            half_spread = 0.02
        elif best_no_bid > 0:
            # Only NO bids — estimate mid from implied ask
            mid = (100 - best_no_bid) / 100
            half_spread = 0.02
        else:
            # Empty orderbook
            mid = 0.50
            half_spread = 0.02

        prices[kalshi_name] = mid
        spreads[kalshi_name] = half_spread
        print(
            f"    {kalshi_name:<30} mid=${mid:.3f}  spread=${half_spread:.3f}  "
            f"(bid={best_yes_bid}c  ask={100 - best_no_bid if best_no_bid else '?'}c)"
        )

    return prices, spreads


# ============================================================================
# Core backtest runner
# ============================================================================


def run_category_buy_hold(
    category: OscarCategory,
    snapshot_keys: list[SnapshotInfo],
    lag_hours: float,
    year_config: YearConfig,
    model_types: list[ModelType],
    ensembles: list[tuple[str, list[ModelType]]],
    centers_only: bool = False,
    live: bool = False,
    live_timestamp: datetime | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run buy-and-hold backtest for one category across all models.

    Unlike the d20260225 version, does NOT require a known winner.
    Instead outputs scenario P&L for every possible winner.

    Returns:
        (scenario_pnl_rows, model_vs_market_rows, position_rows)
    """
    cat_slug = category.slug
    snapshot_key_strs = [k.dir_name for k in snapshot_keys]

    print(f"\n{'=' * 70}")
    print(f"Category: {category.name} ({cat_slug})")
    if live:
        print("  Mode: LIVE orderbook pricing")
    else:
        print(f"  Entry points: {len(snapshot_keys)}")
    print(f"{'=' * 70}")

    # Snapshot availability
    snapshot_availability = {
        s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
    }

    # Fetch market data
    fetch_result = fetch_and_cache_market_data(
        category,
        ceremony_year=year_config.ceremony_year,
        start_date=year_config.calendar.oscar_nominations_date_local,
        end_date=year_config.calendar.oscar_ceremony_date_local,
        market_data_dir=year_config.market_data_dir,
        fetch_hourly=True,
    )
    if fetch_result is None:
        return [], [], []

    market, daily_candles, hourly_candles, trades_df = fetch_result
    spread_by_ticker, mean_spread = estimate_category_spreads(trades_df)
    recommended_configs = get_recommended_configs(
        bankroll=BANKROLL,
        spread_penalty=mean_spread,
    )

    # Load nomination records & build name-mapping data (once per category)
    dataset = load_nomination_dataset(year_config.datasets_dir, category, snapshot_key_strs[0])

    scenario_rows: list[dict] = []
    mvm_rows: list[dict] = []
    position_rows: list[dict] = []

    # --- Build model_runs: (label, preds_by_snapshot) ---
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

    # --- Run each model ---
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

        # Translate predictions to kalshi-name space
        preds_by_snapshot = translate_predictions(preds_by_snapshot, nominee_map)

        # Spread penalties & market prices
        spread_penalties = {
            kn: spread_by_ticker[t]
            for kn in matched_kalshi_names
            if (t := market.nominee_tickers.get(kn)) and t in spread_by_ticker
        }
        market_prices_by_date = build_market_prices(daily_candles)

        # ----- LIVE MODE: fetch orderbook prices, build single entry -----
        if live:
            assert live_timestamp is not None, "live_timestamp required in live mode"
            now_utc = live_timestamp
            live_snap_label = f"live_{now_utc.strftime('%Y-%m-%dT%H:%MZ')}"
            avail_et_str = now_utc.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET")

            print("    Fetching live orderbook prices...")
            live_prices, live_spreads = fetch_live_prices(market, matched_kalshi_names)

            mean_live_spread = (
                sum(live_spreads.values()) / len(live_spreads) if live_spreads else 0.02
            )

            # Use latest snapshot that has predictions
            latest_snap_key = None
            for sk in reversed(snapshot_key_strs):
                if sk in preds_by_snapshot and preds_by_snapshot[sk]:
                    latest_snap_key = sk
                    break
            if latest_snap_key is None:
                print("    WARNING: No predictions at any snapshot")
                continue
            snap_preds = preds_by_snapshot[latest_snap_key]
            print(f"    Using predictions from snapshot: {latest_snap_key}")
            # Model vs market (live prices — raw mid-prices, not normalized)
            for name in matched_kalshi_names:
                model_p = snap_preds.get(name, 0)
                market_p = live_prices.get(name, 0)
                hs = live_spreads.get(name, 0.02)
                mvm_rows.append(
                    {
                        "category": cat_slug,
                        "model_type": short_name,
                        "snapshot_key": live_snap_label,
                        "nominee": name,
                        "model_prob": round(model_p, 4),
                        "market_prob": round(market_p, 4),
                        "divergence": round(model_p - market_p, 4),
                        "half_spread": round(hs, 4),
                        "yes_bid": round(max(0.0, market_p - hs), 4),
                        "yes_ask": round(min(1.0, market_p + hs), 4),
                    }
                )

            # Build single live MarketSnapshot
            moment = MarketSnapshot(
                timestamp=now_utc,
                predictions={
                    name: prob for name, prob in snap_preds.items() if name in matched_kalshi_names
                },
                prices=live_prices,
            )

            # Get configs for this model
            trading_configs = get_configs_for_model(
                short_name,
                centers_only=centers_only,
                recommended_configs=recommended_configs,
            )

            n_configs = len(trading_configs)
            print(f"    Running 1 live entry × {n_configs} configs = {n_configs} backtests")

            # Use live orderbook spreads as per-nominee spread penalties
            live_spread_penalties = live_spreads

            print(f"    Live orderbook spread: {mean_live_spread:.4f}")
            print(f"    Trade-history spread:  {mean_spread:.4f}")

            for cfg in trading_configs:
                bt_config = cfg.model_copy(
                    update={
                        "simulation": cfg.simulation.model_copy(
                            update={"spread_penalty": mean_live_spread}
                        )
                    }
                )

                engine = BacktestEngine(bt_config)
                result = engine.run(
                    moments=[moment],
                    spread_penalties=(live_spread_penalties if live_spread_penalties else None),
                )

                if result.portfolio_history:
                    capital_deployed = sum(
                        pos.outlay_dollars
                        for pos in result.portfolio_history[-1].positions
                        if pos.contracts > 0
                    )
                else:
                    capital_deployed = 0.0

                base_row = {
                    "category": cat_slug,
                    "model_type": short_name,
                    "entry_snapshot": live_snap_label,
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
                }

                if result.settlements:
                    for winner_name, settlement in result.settlements.items():
                        scenario_rows.append(
                            {
                                **base_row,
                                "assumed_winner": winner_name,
                                "final_cash": round(settlement.final_cash, 2),
                                "total_pnl": round(settlement.total_pnl, 2),
                                "return_pct": settlement.return_pct,
                            }
                        )
                else:
                    scenario_rows.append(
                        {
                            **base_row,
                            "assumed_winner": "none",
                            "final_cash": BANKROLL,
                            "total_pnl": 0.0,
                            "return_pct": 0.0,
                        }
                    )

                if result.portfolio_history and result.trade_log:
                    last_snap = result.portfolio_history[-1]
                    for pos in last_snap.positions:
                        if pos.contracts != 0:
                            position_rows.append(
                                {
                                    "category": cat_slug,
                                    "model_type": short_name,
                                    "entry_snapshot": live_snap_label,
                                    "entry_timestamp": avail_et_str,
                                    "config_label": cfg.label,
                                    "outcome": pos.outcome,
                                    "direction": pos.direction.value,
                                    "contracts": pos.contracts,
                                    "avg_cost": round(pos.avg_cost, 4),
                                    "outlay_dollars": round(pos.outlay_dollars, 2),
                                }
                            )

            continue  # skip the non-live path below

        # ----- NON-LIVE MODE (original path) -----

        # Model vs market divergence (no is_winner since winner is unknown)
        for snap_key_str in snapshot_key_strs:
            snap_preds = preds_by_snapshot.get(snap_key_str, {})
            if not snap_preds:
                continue
            snap_date_str = snap_key_str[:10]
            market_on_snap = market_prices_by_date.get(snap_date_str, {})
            filtered_preds = {
                name: prob for name, prob in snap_preds.items() if name in matched_kalshi_names
            }
            if market_on_snap and filtered_preds:
                total_market = sum(market_on_snap.values())
                for name in matched_kalshi_names:
                    model_p = filtered_preds.get(name, 0)
                    market_p = market_on_snap.get(name, 0) if total_market > 0 else 0
                    hs = spread_penalties.get(name, mean_spread)
                    mvm_rows.append(
                        {
                            "category": cat_slug,
                            "model_type": short_name,
                            "snapshot_key": snap_key_str,
                            "nominee": name,
                            "model_prob": round(model_p, 4),
                            "market_prob": round(market_p, 4),
                            "divergence": round(model_p - market_p, 4),
                            "half_spread": round(hs, 4),
                            "yes_bid": round(max(0.0, market_p - hs), 4),
                            "yes_ask": round(min(1.0, market_p + hs), 4),
                        }
                    )

        # Get configs for this model
        trading_configs = get_configs_for_model(
            short_name,
            centers_only=centers_only,
            recommended_configs=recommended_configs,
        )

        # Build entry moments
        entry_moments: list[tuple[SnapshotInfo, MarketSnapshot]] = []
        for key in snapshot_keys:
            entry_moment = build_entry_moment(
                snapshot_key=key,
                snapshot_availability=snapshot_availability,
                preds_by_snapshot=preds_by_snapshot,
                matched_names=matched_kalshi_names,
                market_prices_by_date=market_prices_by_date,
                hourly_candles=hourly_candles,
            )
            if entry_moment is not None:
                entry_moments.append((key, entry_moment))

        print(f"    Entry points with data: {len(entry_moments)}")
        if not entry_moments:
            continue

        n_configs = len(trading_configs)
        n_entries = len(entry_moments)
        print(
            f"    Running {n_entries} entries × {n_configs} configs = "
            f"{n_entries * n_configs} backtests"
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

                # Capital deployed
                if result.portfolio_history:
                    capital_deployed = sum(
                        pos.outlay_dollars
                        for pos in result.portfolio_history[-1].positions
                        if pos.contracts > 0
                    )
                else:
                    capital_deployed = 0.0

                # Shared row fields
                base_row = {
                    "category": cat_slug,
                    "model_type": short_name,
                    "entry_snapshot": key.dir_name,
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
                }

                # Scenario P&L: one row per possible winner
                if result.settlements:
                    for winner_name, settlement in result.settlements.items():
                        scenario_rows.append(
                            {
                                **base_row,
                                "assumed_winner": winner_name,
                                "final_cash": round(settlement.final_cash, 2),
                                "total_pnl": round(settlement.total_pnl, 2),
                                "return_pct": settlement.return_pct,
                            }
                        )
                else:
                    # No positions opened
                    scenario_rows.append(
                        {
                            **base_row,
                            "assumed_winner": "none",
                            "final_cash": BANKROLL,
                            "total_pnl": 0.0,
                            "return_pct": 0.0,
                        }
                    )

                # Position summary: what positions were opened?
                if result.portfolio_history and result.trade_log:
                    last_snap = result.portfolio_history[-1]
                    for pos in last_snap.positions:
                        if pos.contracts != 0:
                            position_rows.append(
                                {
                                    "category": cat_slug,
                                    "model_type": short_name,
                                    "entry_snapshot": key.dir_name,
                                    "entry_timestamp": avail_et_str,
                                    "config_label": cfg.label,
                                    "outcome": pos.outcome,
                                    "direction": pos.direction.value,
                                    "contracts": pos.contracts,
                                    "avg_cost": round(pos.avg_cost, 4),
                                    "outlay_dollars": round(pos.outlay_dollars, 2),
                                }
                            )

    return scenario_rows, mvm_rows, position_rows


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run 2026 buy-and-hold backtest with recommended configs."""
    parser = argparse.ArgumentParser(description="2026 live buy-and-hold backtests")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Category slugs to run (default: all 2026 categories).",
    )
    parser.add_argument(
        "--inferred-lag-hours",
        type=float,
        default=DEFAULT_LAG_HOURS,
        help=f"Hours after event before signal is available (default: {DEFAULT_LAG_HOURS}).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory (default: storage/d20260224_live_2026/results/).",
    )
    parser.add_argument(
        "--centers-only",
        action="store_true",
        help="(no-op, kept for backward compatibility)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live orderbook mid-prices instead of historical candle data.",
    )
    args = parser.parse_args()

    year_config = YEAR_CONFIGS[2026]
    results_dir = (
        Path(args.results_dir) if args.results_dir else Path("storage/d20260224_live_2026/results")
    )
    lag_hours: float = args.inferred_lag_hours
    is_live: bool = args.live

    # Force centers-only in live mode (no neighborhood grids needed)
    centers_only: bool = args.centers_only or is_live

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Categories
    if args.categories:
        categories = [c for c in year_config.categories if c.slug in args.categories]
        if not categories:
            print(f"ERROR: No matching categories for {args.categories}")
            return
    else:
        categories = list(year_config.categories)

    # Models: all 4 individual + 2 ensembles
    model_types = list(BACKTEST_MODEL_TYPES)
    ensembles = ALL_ENSEMBLES

    # Snapshot keys — use AVAILABLE_SNAPSHOT_KEYS (filtered by _TODAY)
    # rather than year_config.snapshot_keys() which includes future events
    snapshot_keys = AVAILABLE_SNAPSHOT_KEYS
    snapshot_availability = {
        s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
    }

    n_models = len(model_types) + len(ensembles)
    if is_live:
        mode_str = "LIVE MODE — fetching current orderbook prices"
    else:
        mode_str = "single recommended config"

    print("=" * 70)
    print(f"2026 Live Buy-and-Hold Backtest ({mode_str})")
    print("=" * 70)
    print(f"Categories: {[c.slug for c in categories]}")
    print(f"Models: {n_models} ({len(model_types)} individual + {len(ensembles)} ensembles)")

    if is_live:
        now_utc = datetime.now(UTC)
        now_et = now_utc.astimezone(_ET)
        print(f"LIVE at {now_et.strftime('%Y-%m-%d %H:%M ET')}")
        print(f"Using latest snapshot predictions: {snapshot_keys[-1].dir_name}")
    else:
        print(f"Signal delay: inferred lag +{lag_hours}h")

    # Show config counts per model
    all_model_names = [mt.short_name for mt in model_types] + [label for label, _ in ensembles]
    for mn in all_model_names:
        configs = get_configs_for_model(mn, centers_only=centers_only)
        print(f"  {mn}: {len(configs)} configs")

    if not is_live:
        print(f"\nEntry points ({len(snapshot_keys)}):")
        for key in snapshot_keys:
            avail = snapshot_availability.get(key.dir_name)
            avail_str = avail.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET") if avail else "N/A"
            print(f"  {key.dir_name} → entry at {avail_str}")

    # Capture timestamp once for all categories (prevents minute-boundary straddling)
    live_ts = datetime.now(UTC) if is_live else None

    # Run backtests
    all_scenario: list[dict] = []
    all_mvm: list[dict] = []
    all_positions: list[dict] = []

    for category in categories:
        scenario, mvm, positions = run_category_buy_hold(
            category,
            snapshot_keys,
            lag_hours=lag_hours,
            year_config=year_config,
            model_types=model_types,
            ensembles=ensembles,
            centers_only=centers_only,
            live=is_live,
            live_timestamp=live_ts,
        )
        all_scenario.extend(scenario)
        all_mvm.extend(mvm)
        all_positions.extend(positions)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)

    def _merge_and_save(
        path: Path, new_df: pd.DataFrame, label_col: str = "entry_snapshot"
    ) -> None:
        """In live mode, merge new rows with existing data instead of overwriting.

        Replaces rows with matching live_* label, preserving historical entries.
        In non-live mode, writes the full DataFrame (overwrite).
        """
        if is_live and path.exists():
            existing = pd.read_csv(path)
            # Get the live label(s) from new data
            new_labels = set(new_df[label_col].unique())
            live_labels = {lbl for lbl in new_labels if str(lbl).startswith("live_")}
            if live_labels:
                # Drop old rows with matching live labels, keep everything else
                existing = existing[~existing[label_col].isin(live_labels)]
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged.to_csv(path, index=False)
            n_old = len(existing)
            print(f"  Merged {len(new_df)} new + {n_old} existing rows → {path.name}")
        else:
            new_df.to_csv(path, index=False)

    if all_scenario:
        scenario_df = pd.DataFrame(all_scenario)
        _merge_and_save(results_dir / "scenario_pnl.csv", scenario_df)
        print(f"\nSaved {len(scenario_df)} scenario P&L rows to scenario_pnl.csv")

        # Aggregate: expected P&L by nominee-weighted scenarios
        # (Users can compute weighted expected P&L externally using model probs)
        # Simple aggregate: mean P&L across all possible winners per config
        agg_cols = [
            "category",
            "model_type",
            "entry_snapshot",
            "config_label",
        ]
        agg_df = scenario_df.groupby(agg_cols, as_index=False).agg(
            mean_pnl=("total_pnl", "mean"),
            min_pnl=("total_pnl", "min"),
            max_pnl=("total_pnl", "max"),
            n_scenarios=("assumed_winner", "nunique"),
            total_trades=("total_trades", "first"),
            capital_deployed=("capital_deployed", "first"),
            fee_type=("fee_type", "first"),
            kelly_fraction=("kelly_fraction", "first"),
            buy_edge_threshold=("buy_edge_threshold", "first"),
            kelly_mode=("kelly_mode", "first"),
            allowed_directions=("allowed_directions", "first"),
        )
        _merge_and_save(results_dir / "scenario_pnl_agg.csv", agg_df)
        print(f"Saved {len(agg_df)} aggregated scenario P&L rows")

    if all_mvm:
        mvm_df = pd.DataFrame(all_mvm)
        _merge_and_save(results_dir / "model_vs_market.csv", mvm_df, label_col="snapshot_key")
        print(f"Saved {len(mvm_df)} model-vs-market rows")

    if all_positions:
        pos_df = pd.DataFrame(all_positions)
        _merge_and_save(results_dir / "position_summary.csv", pos_df)
        print(f"Saved {len(pos_df)} position rows")

    # Print summary
    if all_scenario:
        print("\n" + "=" * 70)
        print("Summary: Center configs – scenario P&L range by model × category")
        print("=" * 70)

        scenario_df = pd.DataFrame(all_scenario)
        center_labels = {cfg.label for cfg in RECOMMENDED_CONFIGS.values()}
        center_df = scenario_df[scenario_df["config_label"].isin(center_labels)]

        if not center_df.empty:
            for (cat, model, entry, _cfg_label), group in center_df.groupby(
                ["category", "model_type", "entry_snapshot", "config_label"]
            ):
                min_pnl = group["total_pnl"].min()
                max_pnl = group["total_pnl"].max()
                mean_pnl = group["total_pnl"].mean()
                trades = group["total_trades"].iloc[0]
                print(
                    f"  {cat:<25} {model:<28} {entry:<20} "
                    f"trades={trades:>3}  "
                    f"P&L: ${min_pnl:>+8.2f} / ${mean_pnl:>+8.2f} / ${max_pnl:>+8.2f} "
                    f"(min/mean/max)"
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
