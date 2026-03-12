"""Build MarketSnapshot objects from Oscar prediction + market price data.

Bridges the gap between data loading (oscar_data) and the backtest engine.
Handles timezone conversions, daily/hourly price lookups, forward-fill
for missing nominees, and snapshot-availability-aware timestamp selection.

Two entry points:

- ``build_trading_moments()`` — builds a full sequence of MarketSnapshots
  for the **rebalancing** backtest (one per trading day).  The TemporalModel
  owns snapshot resolution internally.

- ``build_entry_moment()`` — builds a **single** MarketSnapshot for one
  buy-hold entry point (one snapshot). Handles matched-outcome filtering and
  forward-fill for missing nominees.

Both functions produce ``MarketSnapshot`` objects consumable by
``BacktestEngine.run()``.
"""

import logging
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from oscar_prediction_market.trading.backtest import MarketSnapshot
from oscar_prediction_market.trading.oscar_market import Candle
from oscar_prediction_market.trading.temporal_model import (
    SnapshotInfo,
    TemporalModel,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants — timezone & market mechanics
# ============================================================================

#: Eastern Time zone for computing moment timestamps.
_ET = ZoneInfo("America/New_York")

#: Default moment timestamp: 4 PM ET (roughly Kalshi market close).
#: For activation days in inferred-lag mode, if the signal becomes available
#: before market open (e.g., 4 AM ET after a late ceremony), we snap the
#: moment to 4 PM ET and use daily close prices rather than trying to find
#: intraday candles at low-volume hours. See ``build_trading_moments()`` docstring.
_MARKET_CLOSE_HOUR_ET = 16
_MARKET_CLOSE_MINUTE_ET = 0

#: Earliest hour (ET) at which we trust hourly candles to have meaningful
#: volume. Below this, we snap to 4 PM ET and use daily close.
_MIN_HOURLY_CANDLE_HOUR_ET = 9


# ============================================================================
# Internal helpers
# ============================================================================


def _make_daily_close_timestamp(trading_day: date) -> datetime:
    """Create a 4 PM ET → UTC timestamp for a trading day."""
    return datetime(
        trading_day.year,
        trading_day.month,
        trading_day.day,
        _MARKET_CLOSE_HOUR_ET,
        _MARKET_CLOSE_MINUTE_ET,
        tzinfo=_ET,
    ).astimezone(UTC)


def _get_daily_prices_with_fallback(
    trading_day: date,
    market_prices_by_date: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Look up daily close prices, falling back up to 7 days.

    Tries the exact ``trading_day`` first, then walks backward day by day
    up to 7 days.  This handles weekends, holidays, and gaps in daily
    price data.

    Args:
        trading_day: The date to look up.
        market_prices_by_date: ``{date_str: {model_name: price_dollars}}``.

    Returns:
        ``{model_name: price_dollars}``, empty dict if nothing found.
    """
    td_str = str(trading_day)
    prices = market_prices_by_date.get(td_str, {})
    if not prices:
        for fallback in range(1, 8):
            fb_date = trading_day - timedelta(days=fallback)
            prices = market_prices_by_date.get(str(fb_date), {})
            if prices:
                break
    return prices


def _get_hourly_price_at_timestamp(
    timestamp: datetime,
    hourly_candles: list[Candle],
    matched_names: set[str] | None = None,
) -> dict[str, float] | None:
    """Get nominee prices from hourly candles near a timestamp.

    Searches within +/- 2 hours for the closest candle per ticker.
    Returns ``{kalshi_name: price_dollars}`` or ``None`` if no candles found.

    Args:
        timestamp: UTC datetime to look up.
        hourly_candles: Hourly candle list.
        matched_names: If provided, only include these nominees.

    Returns:
        ``{kalshi_name: price_dollars}``, or ``None`` if unavailable.
    """
    if not hourly_candles:
        return None

    window_start = timestamp - timedelta(hours=1)
    window_end = timestamp + timedelta(hours=2)

    # Filter to candles within the window
    nearby = [c for c in hourly_candles if window_start <= c.timestamp <= window_end]
    if not nearby:
        return None

    # Group by nominee, pick closest candle at or after timestamp
    by_nominee: dict[str, list[Candle]] = {}
    for c in nearby:
        by_nominee.setdefault(c.nominee, []).append(c)

    prices: dict[str, float] = {}
    for nominee, candles in by_nominee.items():
        if matched_names is not None and nominee not in matched_names:
            continue
        # Prefer candle at or just after the timestamp
        at_or_after = [c for c in candles if c.timestamp >= timestamp]
        if at_or_after:
            best = min(at_or_after, key=lambda c: c.timestamp)
        else:
            best = max(candles, key=lambda c: c.timestamp)
        if best.close > 0:
            prices[nominee] = best.close

    return prices if prices else None


# ============================================================================
# Public API
# ============================================================================


def build_trading_moments(
    trading_dates: list[date],
    source: TemporalModel,
    market_prices_by_date: dict[str, dict[str, float]],
    hourly_candles: list[Candle] | None = None,
    use_hourly: bool = False,
) -> list[MarketSnapshot]:
    """Build the list of MarketSnapshots for one (category, source) pair.

    This is the single entry point for converting a ``TemporalModel`` +
    market prices into the ``list[MarketSnapshot]`` that
    ``BacktestEngine.run()`` consumes.

    The ``source`` (a ``TemporalModel``) owns snapshot resolution: when
    queried with a UTC timestamp, it returns predictions from the correct
    underlying snapshot.  This function only needs to determine the
    moment timestamp for each trading day and query the model.

    All data (predictions, prices) should be in Kalshi-name space.  Callers
    translate predictions from model names to Kalshi names once after
    ``match_nominees()``; ``build_market_prices()`` returns Kalshi-native
    prices directly.  This avoids repeated model↔kalshi translation.

    Use ``build_model_source()`` or ``build_ensemble_source()`` from
    ``oscar_prediction_source.py`` to construct
    a source that satisfies these requirements.

    **Activation-day handling (inferred-lag mode):**

    In inferred-lag mode, a snapshot becomes available at a specific UTC
    timestamp (e.g., DGA ends 06:30 UTC Feb 9 + 6h lag = 12:30 UTC Feb 9).
    The "activation day" is the ET date of that timestamp.

    If the signal becomes available before market open (``available_at`` in
    ET is before 9 AM), we snap the moment to **4 PM ET on the same day**
    and use daily close prices instead of hourly candles. Rationale:

    1. Pre-market hourly candles have zero or near-zero volume, making
       prices unreliable (close = 0).
    2. A real trader receiving a signal at 4 AM ET would wait until
       market open at 9:30 AM ET at the earliest.
    3. Using daily close (4 PM ET) is conservative and avoids stale
       overnight prices.

    When the signal is available **during market hours** (9 AM ET or later),
    we use the actual ``available_at`` timestamp and look up the hourly
    candle at that time for realistic intraday execution.

    Args:
        trading_dates: All trading days (sorted).
        source: ``TemporalModel`` providing predictions (owns snapshot resolution).
        market_prices_by_date: ``{date_str: {name: price_dollars}}`` (daily, kalshi names).
        hourly_candles: Hourly candle list (may be None).
        use_hourly: Whether inferred-lag mode is active.

    Returns:
        Sorted list of ``MarketSnapshot`` objects ready for ``BacktestEngine.run()``.
    """
    # Get snapshot availability from the model (SnapshotModel exposes this)
    snapshot_availability: dict[str, datetime] = {}
    if hasattr(source, "snapshot_availability"):
        snapshot_availability = source.snapshot_availability

    # Pre-compute activation days: ET dates when a new snapshot becomes available
    activation_days: dict[date, datetime] = {}
    for _snap_d, avail_at in snapshot_availability.items():
        act_day = avail_at.astimezone(_ET).date()
        if act_day not in activation_days or avail_at > activation_days[act_day]:
            activation_days[act_day] = avail_at

    moments: list[MarketSnapshot] = []
    for trading_day in trading_dates:
        # --- Determine moment timestamp ---
        use_hourly_for_day = False
        if use_hourly and trading_day in activation_days:
            avail_at = activation_days[trading_day]
            avail_et = avail_at.astimezone(_ET)
            if avail_et.hour >= _MIN_HOURLY_CANDLE_HOUR_ET:
                # Signal available during market hours → use actual timestamp
                moment_ts = avail_at
                use_hourly_for_day = True
            else:
                # Signal available pre-market (e.g. 4 AM ET) → snap to 4 PM ET
                # and use daily close. See docstring for rationale.
                moment_ts = _make_daily_close_timestamp(trading_day)
        else:
            moment_ts = _make_daily_close_timestamp(trading_day)

        # --- Query the model at this timestamp ---
        try:
            preds = source.get_predictions(moment_ts)
        except KeyError:
            continue
        if not preds:
            continue

        # --- Get market prices ---
        if use_hourly_for_day and hourly_candles is not None:
            try:
                hourly_prices = _get_hourly_price_at_timestamp(moment_ts, hourly_candles)
                if hourly_prices:
                    prices = hourly_prices
                else:
                    raise ValueError("No hourly candles matched")
            except ValueError as e:
                # Unexpected failure — fall back to daily close
                logger.warning(
                    "Hourly candle unavailable at %s: %s. Falling back to daily close.",
                    moment_ts.isoformat(),
                    e,
                )
                prices = _get_daily_prices_with_fallback(trading_day, market_prices_by_date)
        else:
            prices = _get_daily_prices_with_fallback(trading_day, market_prices_by_date)

        if not prices:
            continue

        moments.append(
            MarketSnapshot(
                timestamp=moment_ts,
                predictions=preds,
                prices=prices,
            )
        )

    return moments


def build_entry_moment(
    snapshot_key: SnapshotInfo,
    snapshot_availability: dict[str, datetime],
    preds_by_snapshot: dict[str, dict[str, float]],
    matched_names: set[str],
    market_prices_by_date: dict[str, dict[str, float]],
    hourly_candles: list[Candle] | None = None,
) -> MarketSnapshot | None:
    """Build a single MarketSnapshot for one entry point (snapshot).

    Uses the snapshot's availability timestamp as the entry time.
    Tries hourly prices first (during market hours), falls back to daily close.

    Unlike ``build_trading_moments`` (where the ``TemporalModel`` handles
    snapshot resolution internally), this function takes per-snapshot
    predictions directly and filters them to the matched outcome set.

    All data (predictions, prices) should be in Kalshi-name space.

    **Forward-fill for missing nominees:** If a matched nominee has no price
    in the current snapshot, we look backward through daily prices for the
    last known price.  This prevents nominees from silently dropping out
    when Kalshi has no candle data at the entry timestamp.

    Args:
        snapshot_key: The snapshot to build an entry moment for.
        snapshot_availability: ``{dir_name: available_at_utc}`` for all snapshots.
        preds_by_snapshot: ``{dir_name: {kalshi_name: probability}}`` predictions.
        matched_names: Set of Kalshi names that matched model predictions.
        market_prices_by_date: ``{date_str: {kalshi_name: price_dollars}}`` (daily).
        hourly_candles: Hourly candle list (may be None).

    Returns:
        A ``MarketSnapshot`` ready for ``BacktestEngine.run()``, or ``None``
        if predictions or prices are unavailable.
    """
    key_str = snapshot_key.dir_name
    avail_at = snapshot_availability.get(key_str)
    if avail_at is None:
        return None

    preds = preds_by_snapshot.get(key_str, {})
    if not preds:
        return None

    filtered_preds = {name: prob for name, prob in preds.items() if name in matched_names}
    if not filtered_preds:
        return None

    # Try hourly prices at entry timestamp (during market hours)
    prices: dict[str, float] | None = None
    avail_et = avail_at.astimezone(_ET)
    if hourly_candles is not None and avail_et.hour >= 9:
        prices = _get_hourly_price_at_timestamp(avail_at, hourly_candles, matched_names)

    # Fall back to daily close prices on the snapshot date or nearby
    if not prices:
        for offset in range(8):  # try up to 7 days forward
            day_str = str(snapshot_key.event_datetime_utc.date() + timedelta(days=offset))
            day_prices = market_prices_by_date.get(day_str)
            if day_prices:
                prices = day_prices
                break

    if not prices:
        return None

    # Forward-fill missing nominees: if a matched nominee has no price in
    # the current snapshot, look backward through daily prices for the last
    # known price.  This prevents nominees from silently dropping out when
    # Kalshi has no candle data at the entry timestamp.
    missing_names = matched_names - set(prices.keys())
    if missing_names:
        entry_date = snapshot_key.event_datetime_utc.date()
        sorted_dates = sorted(market_prices_by_date.keys(), reverse=True)
        for name in missing_names:
            for d_str in sorted_dates:
                if d_str > str(entry_date):
                    continue  # only look backward
                if name in market_prices_by_date[d_str]:
                    prices[name] = market_prices_by_date[d_str][name]
                    logger.info(
                        "FFILL: %s missing at %s, using last known price from %s: $%.2f",
                        name,
                        entry_date,
                        d_str,
                        prices[name],
                    )
                    break

    return MarketSnapshot(
        timestamp=avail_at,
        predictions=filtered_preds,
        prices=prices,
    )
