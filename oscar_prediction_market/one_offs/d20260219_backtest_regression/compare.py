"""Regression test: verify the backtest refactor produces identical results.

This script re-runs the 24 golden configurations from the d20260219_backtest_refactor
branch and compares them against expected values captured before the refactor.

## What is being validated?

The refactor made structural changes:
  - SettlementResult: ``return_pct`` was a broken computed field (always 0.0);
    now a stored float set by the engine from ``initial_bankroll``.
  - BacktestResult.settlements changed from dict[str, dict[str, Any]] to
    dict[str, SettlementResult] -- the shape is identical, just typed.
  - Oscar-specific loaders moved from backtest/data_loader.py to trading/model_loader.py.
  - BacktestConfig renamed ``min_edge`` -> ``buy_edge_threshold``.

None of these changes should affect the numeric output of BacktestEngine.run().
This script verifies that assertion.

## How the golden fixture was captured

In Phase 1 of the refactor, 24 diverse configs were sampled from
``storage/d20260214_trade_signal_ablation/results/ablation_results.json``
(the full 878-config ablation grid). The expected values come directly from
that pre-refactor run.

## Spread estimation

The original ablation used ``spread_penalty_mode = "trade_data"``: it called
``estimate_spread_penalties()`` against Kalshi trade history for
2025-12-01 to 2026-02-15, and passed both per-outcome ``spread_penalties``
and ``median_spread`` to the engine.

We replicate this exactly here. Historical trade data for closed markets is
deterministic, so re-calling ``estimate_spread_penalties()`` for the same
date range returns the same values.

Usage
-----
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.\
one_offs.d20260219_backtest_regression.compare \
        [--fixture storage/d20260219_backtest_regression/golden_fixture.json] \
        [--models-dir storage/d20260214_trade_signal_ablation]
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.one_offs.d20260214_trade_signal_ablation.price_helpers import (
    build_prices_by_date,
)
from oscar_prediction_market.one_offs.legacy_snapshot_loading import (
    estimate_spread_penalties,
    fetch_daily_prices,
    get_snapshot_dates,
    load_weighted_predictions,
)
from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestSimulationConfig,
    MarketSnapshot,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import (
    OscarMarket,
)
from oscar_prediction_market.trading.schema import (
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
    TradingConfig,
)


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


# Date range used in the original ablation -- must match to reproduce spreads.
PRICE_START = date.fromisoformat("2025-12-01")
PRICE_END = date.fromisoformat("2026-02-15")

# Tolerances for numeric comparisons.
# Dollar amounts are rounded to 2 decimal places in the fixture, so 1.5c headroom
# is enough. The original run also rounded final_wealth / fees to 2dp.
DOLLAR_TOL = 0.015  # $0.015 -- covers rounding to nearest cent
PCT_TOL = 0.05  # 0.05 percentage points -- generous for rounded pct values
TRADE_COUNT_TOL = 0  # exact integer match


def _build_engine_config(ec: dict, median_spread: float) -> BacktestConfig:
    """Construct a BacktestConfig from a golden fixture engine_config entry.

    The golden fixture uses legacy field names:
      - ``min_edge``  →  ``buy_edge_threshold``
    Other fixture fields (``market_blend_alpha``, ``normalize_probabilities``,
    ``model_type``) are prediction-loading parameters, not engine parameters —
    stripped here.  ``spread_penalty`` was not stored in the fixture because it
    was derived from live trade data at run time; we re-inject it here.
    """
    return BacktestConfig(
        trading=TradingConfig(
            kelly=KellyConfig(
                bankroll=ec["bankroll_dollars"],
                kelly_fraction=ec["kelly_fraction"],
                buy_edge_threshold=ec["min_edge"],
                max_position_per_outcome=ec["max_position_per_outcome_dollars"],
                max_total_exposure=ec["max_total_exposure_dollars"],
                kelly_mode=KellyMode(ec["kelly_mode"]),
            ),
            sell_edge_threshold=ec["sell_edge_threshold"],
            fee_type=FeeType(ec["fee_type"]),
            limit_price_offset=0.0,
            min_price=ec.get("min_price", 0),
            allowed_directions=frozenset({PositionDirection.YES}),
        ),
        simulation=BacktestSimulationConfig(
            spread_penalty=median_spread,
            bankroll_mode=ec["bankroll_mode"],
        ),
    )


def _compare_scalar(name: str, actual: float, expected: float, tol: float) -> tuple[bool, str]:
    """Return (passed, message) for a single scalar comparison."""
    diff = abs(actual - expected)
    if diff <= tol:
        return True, f"  OK {name}: {actual} (expected {expected}, diff {diff:.4f})"
    return False, f"  FAIL {name}: {actual} (expected {expected}, diff {diff:.4f} > tol {tol})"


def _load_predictions_for_config(
    ec: dict,
    models_dir: Path,
    snapshot_dates: list[str],
    prices_by_date: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Load predictions for all snapshot dates for a given engine config.

    For market_blend and alpha-override configs, market prices are passed
    through for each snapshot date (as was done in the original ablation).
    """
    model_type = ec["model_type"]
    alpha = ec.get("market_blend_alpha")
    normalize = ec.get("normalize_probabilities", False)
    needs_market_prices = alpha is not None or model_type == "market_blend"

    predictions_by_date: dict[str, dict[str, float]] = {}
    for snap_date in snapshot_dates:
        market_prices = prices_by_date.get(snap_date, {}) if needs_market_prices else None
        preds = load_weighted_predictions(
            models_dir=models_dir,
            snapshot_date=snap_date,
            model_type=model_type,
            market_prices=market_prices if market_prices else None,
            market_blend_alpha=alpha,
            normalize_probabilities=normalize,
        )
        if preds:
            predictions_by_date[snap_date] = preds
    return predictions_by_date


def _run_config(
    entry: dict,
    models_dir: Path,
    snapshot_dates: list[str],
    prices_by_date: dict[str, dict[str, float]],
    spread_penalties: dict[str, float],
    median_spread: float,
) -> tuple[bool, list[str]]:
    """Re-run a single golden config and compare against expected values.

    Returns (passed, log_lines).
    """
    config_id = entry["config_id"]
    ec = entry["engine_config"]
    expected = entry["expected"]
    expected_snaps = entry["expected_snapshots"]

    lines: list[str] = [f"\n{chr(8212) * 60}", f"Config: {config_id}"]

    predictions_by_date = _load_predictions_for_config(
        ec, models_dir, snapshot_dates, prices_by_date
    )
    engine_config = _build_engine_config(ec, median_spread)
    moments = _dicts_to_moments(predictions_by_date, prices_by_date)
    result = BacktestEngine(engine_config).run(
        moments=moments,
        spread_penalties=spread_penalties if spread_penalties else None,
    )

    passed = True
    checks: list[tuple[bool, str]] = [
        _compare_scalar("final_wealth", result.final_wealth, expected["final_wealth"], DOLLAR_TOL),
        _compare_scalar(
            "total_return_pct", result.total_return_pct, expected["total_return_pct"], PCT_TOL
        ),
        _compare_scalar(
            "total_fees_paid", result.total_fees_paid, expected["total_fees_paid"], DOLLAR_TOL
        ),
        _compare_scalar(
            "total_trades", result.total_trades, expected["total_trades"], TRADE_COUNT_TOL
        ),
        _compare_scalar(
            "n_snapshots", len(result.portfolio_history), expected["n_snapshots"], TRADE_COUNT_TOL
        ),
    ]
    for ok, msg in checks:
        lines.append(msg)
        if not ok:
            passed = False

    # Per-snapshot comparison
    snap_map = {s.timestamp.isoformat(): s for s in result.portfolio_history}
    for exp_snap in expected_snaps:
        snap_date = exp_snap["snapshot_date"]
        if snap_date not in snap_map:
            lines.append(f"  FAIL snapshot {snap_date}: MISSING in actual result")
            passed = False
            continue
        actual_snap = snap_map[snap_date]
        snap_checks = [
            _compare_scalar(
                f"  snap[{snap_date}].cash", actual_snap.cash, exp_snap["cash"], DOLLAR_TOL
            ),
            _compare_scalar(
                f"  snap[{snap_date}].total_wealth",
                actual_snap.total_wealth,
                exp_snap["total_wealth"],
                DOLLAR_TOL,
            ),
            _compare_scalar(
                f"  snap[{snap_date}].n_positions",
                actual_snap.n_positions,
                exp_snap["n_positions"],
                TRADE_COUNT_TOL,
            ),
        ]
        for ok, msg in snap_checks:
            lines.append(msg)
            if not ok:
                passed = False

    status = "PASS" if passed else "FAIL"
    lines.insert(2, f"Status: {status}")
    return passed, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest refactor regression test.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("storage/d20260219_backtest_regression/golden_fixture.json"),
        help="Path to golden_fixture.json",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("storage/d20260214_trade_signal_ablation"),
        help="Directory containing model snapshots (used by original ablation).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Backtest Regression Test")
    print("=" * 60)

    # Load fixture
    with open(args.fixture) as f:
        fixture = json.load(f)
    configs = fixture["configs"]
    print(f"Loaded {len(configs)} golden configs from {args.fixture}")

    # -------------------------------------------------------------------------
    # Fetch shared data once -- same as the original ablation's shared data setup
    # -------------------------------------------------------------------------
    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    market = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_data.nominee_tickers)
    print(f"\nFetching daily prices: {PRICE_START} -> {PRICE_END}")
    raw_prices = fetch_daily_prices(market, start_date=PRICE_START, end_date=PRICE_END)
    models_dir = args.models_dir / "models"
    snapshot_dates = get_snapshot_dates(models_dir, "lr")
    prices_by_date = build_prices_by_date(raw_prices, snapshot_dates)
    print(f"  {len(raw_prices)} daily price rows, {len(prices_by_date)} snapshot dates")

    print(f"\nEstimating spread penalties from trade history: {PRICE_START} -> {PRICE_END}")
    spread_penalties, median_spread = estimate_spread_penalties(
        market, start_date=PRICE_START, end_date=PRICE_END
    )
    print(f"  median spread: {median_spread:.2f}c")
    if spread_penalties:
        per_outcome = ", ".join(f"{k}: {v:.1f}c" for k, v in list(spread_penalties.items())[:4])
        print(f"  per-outcome (first 4): {per_outcome} ...")

    print(f"  {len(snapshot_dates)} snapshot dates: {snapshot_dates[0]} -> {snapshot_dates[-1]}")

    # -------------------------------------------------------------------------
    # Run all configs
    # -------------------------------------------------------------------------
    print(f"\nRunning {len(configs)} configs against golden fixture...")
    results: list[tuple[str, bool, list[str]]] = []
    for entry in configs:
        passed, log_lines = _run_config(
            entry=entry,
            models_dir=models_dir,
            snapshot_dates=snapshot_dates,
            prices_by_date=prices_by_date,
            spread_penalties=spread_penalties,
            median_spread=median_spread,
        )
        results.append((entry["config_id"], passed, log_lines))
        for line in log_lines:
            print(line)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass}/{len(results)} configs passed")
    if n_fail:
        print("FAILED configs:")
        for config_id, ok, _ in results:
            if not ok:
                print(f"  - {config_id}")
    print("=" * 60)

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
