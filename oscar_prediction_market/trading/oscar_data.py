"""Oscar market-data helpers for trading and backtesting.

This module is intentionally narrow after the refactor: it owns the market
side of the pipeline only. Prediction loading, namespace remapping, and
model/Kalshi matching live in ``oscar_prediction_source.py`` so callers can
reason separately about "model-side adaptation" and "market-side data".
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.trading.edge import (
    estimate_spread_from_trades,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.name_matching import normalize_name
from oscar_prediction_market.trading.oscar_market import (
    Candle,
    OscarMarket,
    TradeRecord,
    _candle_list_adapter,
)

logger = logging.getLogger(__name__)


def build_market_prices(
    daily_candles: list[Candle],
) -> dict[str, dict[str, float]]:
    """Build ``{date_str: {kalshi_name: yes_price}}`` from daily candles.

    Returns prices keyed by the candle's nominee field (Kalshi registry names).
    Callers should ensure predictions are also in Kalshi-name space so that
    dict keys align without translation.
    """
    prices_by_date: dict[str, dict[str, float]] = {}

    for candle in daily_candles:
        if candle.close <= 0:
            continue
        prices_by_date.setdefault(str(candle.date), {})[candle.nominee] = candle.close

    return prices_by_date


def fetch_and_cache_market_data(
    category: OscarCategory,
    ceremony_year: int,
    start_date: date,
    end_date: date,
    market_data_dir: Path,
    fetch_hourly: bool = False,
) -> tuple[OscarMarket, list[Candle], list[Candle] | None, list[TradeRecord]] | None:
    """Fetch market data from Kalshi API and cache to disk.

    This helper is intentionally opinionated about caching because the Oscar
    backtest one-offs repeatedly reuse the same season/category market data.
    Centralizing the fetch+cache policy keeps those one-offs thin and ensures
    they all consume the same normalized candle/trade representations.
    """
    logger.info("Fetching market data for %s (%d)...", category.slug, ceremony_year)
    try:
        cat_data = OSCAR_MARKETS.get_category_data(category, ceremony_year)
    except (KeyError, ValueError) as exc:
        logger.warning("No market tickers for %s/%d: %s", category.name, ceremony_year, exc)
        return None

    market = OscarMarket(
        event_ticker=cat_data.event_ticker,
        nominee_tickers=cat_data.nominee_tickers,
    )

    price_start = start_date - timedelta(days=7)
    price_end = end_date + timedelta(days=1)
    daily_candles = market.get_daily_prices(price_start, price_end)
    trades = market.fetch_trade_history(price_start, price_end)

    if not daily_candles:
        logger.warning("No price data for %s", category.name)
        return None

    candle_dir = market_data_dir / "candles"
    candle_dir.mkdir(parents=True, exist_ok=True)
    event_ticker = market.event_ticker
    (candle_dir / f"{event_ticker}_candles.json").write_bytes(
        _candle_list_adapter.dump_json(daily_candles)
    )

    if trades:
        trade_dir = market_data_dir / "trades"
        trade_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "timestamp": trade.timestamp,
                    "date": trade.date,
                    "ticker": trade.ticker,
                    "nominee": trade.nominee,
                    "yes_price": trade.yes_price,
                    "no_price": trade.no_price,
                    "count": trade.count,
                    "taker_side": trade.taker_side,
                }
                for trade in trades
            ]
        ).to_parquet(trade_dir / f"{event_ticker}_trades.parquet")

    hourly_candles: list[Candle] | None = None
    if fetch_hourly:
        hourly_cache = candle_dir / f"{event_ticker}_hourly_candles.json"
        if hourly_cache.exists():
            logger.info("Loading cached hourly candles...")
            hourly_candles = _candle_list_adapter.validate_json(hourly_cache.read_bytes())
        else:
            logger.info("Fetching hourly candles from API...")
            hourly_candles = market.get_hourly_prices(price_start, price_end)
            if hourly_candles:
                hourly_cache.write_bytes(_candle_list_adapter.dump_json(hourly_candles))
            else:
                logger.warning("No hourly candle data available")
                hourly_candles = None

    return market, daily_candles, hourly_candles, trades


def estimate_category_spreads(
    trades: list[TradeRecord],
    default_spread: float = 0.03,
    min_trades: int = 10,
) -> tuple[dict[str, float], float]:
    """Estimate per-ticker spreads from trade history.

    The backtests need both the per-ticker estimate for outcome-level trading
    penalties and a category-level mean spread for grid generation and
    fallbacks. Keeping both outputs together avoids recomputing spread stats
    in each one-off.
    """
    logger.info("Estimating spreads...")
    if trades:
        spread_by_ticker = estimate_spread_from_trades(
            trades,
            default_spread=default_spread,
            min_trades_required=min_trades,
        )
    else:
        spread_by_ticker = {}

    mean_spread = (
        sum(spread_by_ticker.values()) / len(spread_by_ticker)
        if spread_by_ticker
        else default_spread / 2
    )
    return spread_by_ticker, mean_spread


def get_winner_kalshi_name(
    category: OscarCategory,
    matched_kalshi_names: set[str],
    winners: dict[OscarCategory, str],
) -> str:
    """Resolve the winner name into Kalshi-name space for settlement.

    Winner labels may already match exactly, or may match only after
    normalization (accent/case differences).  Doing this lookup in one
    place makes settlement behavior consistent across one-off runners.
    """
    raw_winner = winners[category]
    if raw_winner in matched_kalshi_names:
        return raw_winner

    norm_winner = normalize_name(raw_winner)
    for kalshi_name in matched_kalshi_names:
        if normalize_name(kalshi_name) == norm_winner:
            return kalshi_name

    raise ValueError(
        f"Cannot find winner '{raw_winner}' for {category.name} "
        f"in kalshi names: {sorted(matched_kalshi_names)}"
    )
