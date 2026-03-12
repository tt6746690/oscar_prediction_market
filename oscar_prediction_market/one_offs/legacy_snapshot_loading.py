"""Legacy prediction loading and market data utilities.

Path-based utilities for loading model predictions from the temporal snapshot
directory layout and for fetching market data from Kalshi.  Used by older
one-off experiments (d20260214, d20260219) that work with the legacy
``models_dir/{model_type}/{snapshot_date}/`` layout.

New code should use ``trading.temporal_model`` and the factory helpers in
``data_prep.py`` of the d20260220 backtest package.

Directory convention::

    models_dir/
    └── {model_type}/
        └── {snapshot_date}/
            └── {model_type}_{snapshot_date}/
                └── 5_final_predict/
                    └── predictions_test.csv
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from oscar_prediction_market.trading.edge import (
    estimate_spread_from_trades,
)
from oscar_prediction_market.trading.oscar_market import OscarMarket

logger = logging.getLogger(__name__)


# ============================================================================
# Snapshot directory utilities
# ============================================================================


def get_snapshot_dates(models_dir: Path, model_type: str) -> list[str]:
    """Get sorted list of available snapshot date strings for a model type.

    Scans subdirectories of ``models_dir/{model_type}/`` and returns their
    names as sorted strings (assumed ``YYYY-MM-DD``).
    """
    model_dir = models_dir / model_type
    if not model_dir.is_dir():
        return []
    return sorted(d.name for d in model_dir.iterdir() if d.is_dir())


# ============================================================================
# Prediction loaders
# ============================================================================


def load_snapshot_predictions(
    models_dir: Path,
    model_type: str,
    snapshot_date: str,
) -> dict[str, float]:
    """Load model predictions for a single snapshot date.

    Reads ``predictions_test.csv`` from the standard temporal snapshot layout.

    Returns:
        ``{outcome_title: probability}`` mapping, empty if file missing.
    """
    run_dir = models_dir / model_type / snapshot_date / f"{model_type}_{snapshot_date}"
    pred_path = run_dir / "5_final_predict" / "predictions_test.csv"

    if not pred_path.exists():
        logger.warning("No predictions at %s", pred_path)
        return {}

    df = pd.read_csv(pred_path)
    return dict(zip(df["title"], df["probability"], strict=True))


def load_average_predictions(
    models_dir: Path,
    snapshot_date: str,
    model_types: list[str],
) -> dict[str, float]:
    """Compute equal-weight average predictions across model types for one date.

    Outcomes that appear in only a subset of models are averaged over that
    subset (not penalised for missingness).
    """
    all_preds: dict[str, list[float]] = {}
    for mt in model_types:
        preds = load_snapshot_predictions(models_dir, mt, snapshot_date)
        for title, prob in preds.items():
            all_preds.setdefault(title, []).append(prob)

    return {title: sum(probs) / len(probs) for title, probs in all_preds.items() if probs}


def load_weighted_predictions(
    models_dir: Path,
    snapshot_date: str,
    model_type: str,
    market_prices: dict[str, float] | None = None,
    market_blend_alpha: float | None = None,
    normalize_probabilities: bool = False,
) -> dict[str, float]:
    """Load predictions supporting all model type variants and blends.

    Supported ``model_type`` values:

    ``'lr'``, ``'gbt'``
        Pure base model predictions.
    ``'avg'``
        Equal-weight average of LR and GBT.
    ``'market_blend'``
        1/3 LR + 1/3 GBT + 1/3 market implied probability.

    If ``market_blend_alpha`` is provided with ``market_prices``, applies:
    ``P = α * P_market + (1 - α) * P_model`` (overrides model_type).
    """
    if market_blend_alpha is not None and market_prices is not None:
        if model_type not in ("lr", "gbt"):
            raise ValueError(
                f"market_blend_alpha only valid for base models (lr, gbt), got '{model_type}'"
            )
        preds = load_snapshot_predictions(models_dir, model_type, snapshot_date)
        if not preds:
            return {}
        result: dict[str, float] = {}
        for outcome, model_p in preds.items():
            market_p = market_prices.get(outcome)
            if market_p is not None:
                result[outcome] = market_blend_alpha * market_p + (1 - market_blend_alpha) * model_p
            else:
                result[outcome] = model_p

    elif model_type in ("lr", "gbt"):
        result = load_snapshot_predictions(models_dir, model_type, snapshot_date)

    elif model_type == "avg":
        result = load_average_predictions(models_dir, snapshot_date, ["lr", "gbt"])

    elif model_type == "market_blend":
        lr_preds = load_snapshot_predictions(models_dir, "lr", snapshot_date)
        gbt_preds = load_snapshot_predictions(models_dir, "gbt", snapshot_date)
        if not lr_preds or not gbt_preds or not market_prices:
            result = load_average_predictions(models_dir, snapshot_date, ["lr", "gbt"])
        else:
            blended: dict[str, float] = {}
            all_outcomes = set(lr_preds.keys()) | set(gbt_preds.keys())
            for outcome in all_outcomes:
                values: list[float] = []
                if outcome in lr_preds:
                    values.append(lr_preds[outcome])
                if outcome in gbt_preds:
                    values.append(gbt_preds[outcome])
                if outcome in market_prices:
                    values.append(market_prices[outcome])
                if values:
                    blended[outcome] = sum(values) / len(values)
            result = blended

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    if normalize_probabilities and result:
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

    return result


def load_all_predictions_for_snapshots(
    models_dir: Path,
    model_type: str,
    snapshot_dates: list[str],
    market_prices_by_date: dict[str, dict[str, float]] | None = None,
    market_blend_alpha: float | None = None,
    normalize_probabilities: bool = False,
) -> dict[str, dict[str, float]]:
    """Load predictions for all snapshot dates for a given model type.

    Convenience wrapper around :func:`load_weighted_predictions`.
    """
    predictions_by_date: dict[str, dict[str, float]] = {}
    for snap_date in snapshot_dates:
        market_prices = market_prices_by_date.get(snap_date) if market_prices_by_date else None
        preds = load_weighted_predictions(
            models_dir=models_dir,
            snapshot_date=snap_date,
            model_type=model_type,
            market_prices=market_prices,
            market_blend_alpha=market_blend_alpha,
            normalize_probabilities=normalize_probabilities,
        )
        if preds:
            predictions_by_date[snap_date] = preds
    return predictions_by_date


# ============================================================================
# Market data utilities
# ============================================================================


def fetch_daily_prices(
    market: OscarMarket,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch daily closing prices from Kalshi for all outcomes in a market.

    Thin wrapper around ``OscarMarket.get_daily_prices()`` with logging.
    Returns a DataFrame for backward-compatibility with legacy one-off scripts.
    """
    logger.info("Fetching daily prices: %s to %s", start_date, end_date)
    candles = market.get_daily_prices(start_date=start_date, end_date=end_date)
    logger.info("  %d daily price rows", len(candles))
    return pd.DataFrame([{"date": c.date, "nominee": c.nominee, "close": c.close} for c in candles])


def estimate_spread_penalties(
    market: OscarMarket,
    start_date: date,
    end_date: date,
    default_spread: float = 0.04,
    min_trades_required: int = 20,
) -> tuple[dict[str, float], float]:
    """Estimate bid-ask spread penalties from historical Kalshi trade data.

    Returns:
        ``(spread_penalties_by_outcome, median_spread)`` in dollars.
    """
    logger.info("Estimating spreads from trade history: %s to %s", start_date, end_date)
    trades = market.fetch_trade_history(start_date=start_date, end_date=end_date)

    spread_penalties: dict[str, float] = {}
    if trades:
        ticker_spreads = estimate_spread_from_trades(
            trades,
            default_spread=default_spread,
            min_trades_required=min_trades_required,
        )
        for ticker, spread in ticker_spreads.items():
            outcome = market.ticker_to_nominee.get(ticker, ticker)
            spread_penalties[outcome] = spread
        logger.info("  Estimated spreads for %d outcomes", len(spread_penalties))
    else:
        logger.warning("  No trade data available, using default $%.2f", default_spread)

    if spread_penalties:
        median_spread = sorted(spread_penalties.values())[len(spread_penalties) // 2]
    else:
        median_spread = default_spread

    return spread_penalties, median_spread
