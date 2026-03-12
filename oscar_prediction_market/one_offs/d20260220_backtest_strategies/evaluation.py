"""Evaluation helpers for backtest experiments.

Model accuracy metrics and baselines that are independent of the
backtest engine. Used by the runner to annotate results CSVs.
"""

from __future__ import annotations

from oscar_prediction_market.data.awards_calendar import AwardsCalendar
from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.trading.kalshi_client import estimate_fee
from oscar_prediction_market.trading.oscar_market import Candle
from oscar_prediction_market.trading.schema import FeeType
from oscar_prediction_market.trading.temporal_model import get_trading_dates


def evaluate_model_accuracy(
    predictions: dict[str, float],
    winner: str,
) -> dict:
    """Evaluate model predictions for a single snapshot.

    Returns:
        Dict with rank, winner_prob, brier, accuracy.
    """
    if not predictions or winner not in predictions:
        return {"rank": None, "winner_prob": None, "brier": None, "accuracy": None}

    # Rank: 1 = model's top pick is the winner
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    rank = next(i + 1 for i, (name, _) in enumerate(sorted_preds) if name == winner)

    winner_prob = predictions[winner]

    # Brier score: mean squared error of probabilities
    brier = 0.0
    for name, prob in predictions.items():
        actual = 1.0 if name == winner else 0.0
        brier += (prob - actual) ** 2
    brier /= len(predictions)

    accuracy = 1 if rank == 1 else 0

    return {
        "rank": rank,
        "winner_prob": round(winner_prob, 4),
        "brier": round(brier, 4),
        "accuracy": accuracy,
    }


def run_market_favorite_baseline(
    category: OscarCategory,
    winners: dict[OscarCategory, str],
    calendar: AwardsCalendar,
    daily_candles: list[Candle] | None = None,
) -> list[dict]:
    """Run 'buy the market favorite' baseline for a category.

    Each trading day, buy the nominee with the highest Kalshi price.
    Settle against known winner. Includes maker fees (1.75%) to be
    comparable with the main backtest.

    Args:
        category: Oscar category.
        winners: ``{category: winner_name}`` dict for settlement.
        calendar: Awards calendar for deriving trading dates.
        daily_candles: Pre-fetched daily candles. If None or empty,
            returns an empty list (caller should have fetched market data).
    """
    cat_slug = category.slug
    trading_dates = get_trading_dates(calendar)
    winner = winners[category]

    if not daily_candles:
        return []

    # Build winner name set for robust matching
    winner_lower = winner.lower()

    rows = []
    for trading_day in trading_dates:
        day_candles = [c for c in daily_candles if c.date == trading_day]
        if not day_candles:
            continue

        # Find favorite (highest close price)
        fav = max(day_candles, key=lambda c: c.close)
        favorite = fav.nominee
        favorite_price = fav.close

        # Look up winner price
        winner_candle = next((c for c in day_candles if c.nominee.lower() == winner_lower), None)
        winner_price = winner_candle.close if winner_candle else 0

        # Fee for buying 1 contract at the favorite's price (maker)
        fee = estimate_fee(favorite_price, FeeType.MAKER, n_contracts=1)

        # P&L: buy 1 contract of favorite, including fee
        fav_is_winner = favorite.lower() == winner_lower or (
            winner_candle is not None and favorite == winner_candle.nominee
        )
        if fav_is_winner:
            pnl = 1.0 - favorite_price - fee  # Win: payout $1 - cost - fee
        else:
            pnl = -(favorite_price + fee)  # Lose: lose cost + fee

        rows.append(
            {
                "category": cat_slug,
                "date": str(trading_day),
                "favorite": favorite,
                "favorite_price": favorite_price,
                "winner_price": winner_price,
                "fee": fee,
                "pnl_per_contract": round(pnl, 4),
            }
        )

    return rows
