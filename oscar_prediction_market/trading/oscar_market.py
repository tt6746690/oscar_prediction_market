"""Oscar prediction market data on Kalshi.

Wraps the generic :class:`KalshiPublicClient` with Oscar-specific methods:
nominee↔ticker mappings, price fetching, candlestick history, orderbook
snapshots, and trade history.

Market metadata (tickers, nominees) lives in the :mod:`market_data` registry.
Use that module to look up tickers and construct ``OscarMarket`` instances.

Usage::

    from oscar_prediction_market.data.schema import OscarCategory
    from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
    from oscar_prediction_market.trading.oscar_market import OscarMarket

    # Build from registry (two-step)
    data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=data.event_ticker, nominee_tickers=data.nominee_tickers)

    # Or use the classmethod shorthand
    mkt = OscarMarket.from_registry_data(data)

    prices = mkt.get_prices()
    daily  = mkt.get_daily_prices(date(2025, 12, 1), date(2026, 3, 1))
"""

import datetime
import logging
from datetime import date
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, TypeAdapter, computed_field

from oscar_prediction_market.trading.kalshi_client import (
    KalshiPublicClient,
)
from oscar_prediction_market.trading.market_data import (
    CategoryCeremonyData,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


class Candle(BaseModel):
    """Single OHLC candlestick for one nominee/ticker.

    Represents one price bar from the Kalshi candlestick API. All price
    fields are converted to dollars at the Oscar-market boundary. Trading
    code in this package treats probabilities/prices as dollar fractions
    almost everywhere, so doing the cents->dollars conversion here prevents
    repeated ad hoc conversions downstream.

    Example::

        Candle(
            timestamp=datetime(2025, 2, 1, 21, 0, tzinfo=UTC),
            date=date(2025, 2, 1),
            ticker="KXOSCARPIC-26-SIN",
            nominee="Sinners",
            open=0.18, high=0.22, low=0.17, close=0.20,
            volume=350,
        )
    """

    model_config = {"extra": "forbid"}

    timestamp: datetime.datetime
    date: date
    ticker: str
    nominee: str
    open: float = Field(ge=0, le=1.0)
    high: float = Field(ge=0, le=1.0)
    low: float = Field(ge=0, le=1.0)
    close: float = Field(ge=0, le=1.0)
    volume: int = Field(ge=0)

    @property
    def yes_price(self) -> float:
        """Close price expressed as yes-contract dollars."""
        return self.close

    @property
    def no_price(self) -> float:
        """Implied no-contract price in dollars."""
        return 1.0 - self.close


class TradeRecord(BaseModel):
    """Executed trade normalized into dollar-price units.

    The Kalshi API exposes trade prices in cents, but spread estimation and
    trading simulation consume dollar fractions. Normalizing once here keeps
    the rest of the trading stack consistent with the candle/backtest
    convention and avoids mixed-unit bugs.
    """

    model_config = {"extra": "forbid"}

    timestamp: datetime.datetime
    date: date
    ticker: str
    nominee: str
    yes_price: float = Field(ge=0, le=1.0)
    count: int = Field(ge=0)
    taker_side: str

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def no_price(self) -> float:
        """Implied no-contract price in dollars."""
        return 1.0 - self.yes_price


_candle_list_adapter = TypeAdapter(list[Candle])


class OutcomePrice(BaseModel):
    """Current price snapshot for one nominee/outcome.

    Captures the best bid/ask on both YES and NO sides, plus last traded
    price and volume. NO prices are derived from YES: ``no_bid = 100 - yes_ask``
    and ``no_ask = 100 - yes_bid``.

    Example (nominee "Sinners" trading around 20c)::

        OutcomePrice(
            ticker="KXOSCARPIC-26-SIN",
            yes_bid=18, yes_ask=22,
            no_bid=78, no_ask=82,
            spread=4,
            last_price=20, volume=1523,
        )
    """

    model_config = {"extra": "forbid"}

    ticker: str
    yes_bid: int = Field(..., ge=0, le=100, description="Best bid for YES in cents")
    yes_ask: int = Field(..., ge=0, le=100, description="Best ask for YES in cents")
    no_bid: int = Field(..., ge=0, le=100, description="Best bid for NO (= 100 - yes_ask)")
    no_ask: int = Field(..., ge=0, le=100, description="Best ask for NO (= 100 - yes_bid)")
    spread: int = Field(..., ge=0, description="yes_ask - yes_bid in cents")
    last_price: int = Field(..., ge=0, le=100)
    volume: int = Field(..., ge=0)


# ============================================================================
# Helpers
# ============================================================================


def _date_to_ts(d: date) -> int:
    """Convert date to Unix timestamp (UTC midnight)."""
    return int(datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.UTC).timestamp())


# ============================================================================
# OscarMarket (parameterized by category)
# ============================================================================


class OscarMarket:
    """Oscar prediction market on Kalshi, parameterized by category.

    Wraps a :class:`KalshiPublicClient` and provides Oscar-specific
    methods: nominee lookups, daily/intraday price history, orderbook
    snapshots, and trade history. The design intent is that this class is
    the market-data boundary where raw Kalshi responses become Oscar-labeled,
    dollar-denominated records usable by the rest of the trading code.

    Parameters
    ----------
    event_ticker : str
        Kalshi event ticker (e.g. ``"KXOSCARPIC-26"``).
    nominee_tickers : dict[str, str]
        Maps nominee name → market ticker (e.g. ``{"Sinners": "KXOSCARPIC-26-SIN"}``).
    client : KalshiPublicClient, optional
        Existing client instance.  A new one is created if omitted.
    """

    def __init__(
        self,
        event_ticker: str,
        nominee_tickers: dict[str, str],
        client: KalshiPublicClient | None = None,
    ) -> None:
        self._event_ticker = event_ticker
        self._nominee_tickers = dict(nominee_tickers)  # defensive copy
        self.client = client or KalshiPublicClient()

    @property
    def event_ticker(self) -> str:
        """Kalshi event ticker."""
        return self._event_ticker

    @property
    def nominee_tickers(self) -> dict[str, str]:
        """Nominee name → ticker mapping."""
        return self._nominee_tickers

    @property
    def ticker_to_nominee(self) -> dict[str, str]:
        """Reverse mapping: ticker → nominee name."""
        return {v: k for k, v in self._nominee_tickers.items()}

    @classmethod
    def from_registry_data(
        cls,
        data: CategoryCeremonyData,
        client: KalshiPublicClient | None = None,
    ) -> "OscarMarket":
        """Construct from :class:`~market_data.CategoryCeremonyData`.

        Example::

            from oscar_prediction_market.trading.market_data import (
                OSCAR_MARKETS,
            )
            data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
            mkt = OscarMarket.from_registry_data(data)
        """
        return cls(
            event_ticker=data.event_ticker,
            nominee_tickers=data.nominee_tickers,
            client=client,
        )

    # -- Current prices --------------------------------------------------------

    def get_prices(self) -> dict[str, OutcomePrice]:
        """Get current prices for all nominees in this market.

        Returns dict mapping nominee name to :class:`OutcomePrice` with
        yes/no bid/ask, spread, last price, and volume.

        Example::

            >>> from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
            >>> data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
            >>> mkt = OscarMarket(event_ticker=data.event_ticker, nominee_tickers=data.nominee_tickers)
            >>> prices = mkt.get_prices()
            >>> prices["Sinners"].yes_bid
            18
            >>> prices["Sinners"].spread
            4
        """
        markets = self.client.get_event_markets(event_ticker=self.event_ticker)
        result: dict[str, OutcomePrice] = {}
        for m in markets:
            # KalshiMarket has extra="allow", so custom_strike may be present
            custom_strike = getattr(m, "custom_strike", None)
            if isinstance(custom_strike, dict):
                nominee = custom_strike.get("Nominee", m.ticker)
            else:
                nominee = m.ticker
            yes_bid = m.yes_bid
            yes_ask = m.yes_ask
            result[nominee] = OutcomePrice(
                ticker=m.ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=100 - yes_ask,
                no_ask=100 - yes_bid,
                spread=yes_ask - yes_bid,
                last_price=m.last_price,
                volume=m.volume,
            )
        return result

    # -- Candlestick history ---------------------------------------------------

    def fetch_candlestick_history(
        self,
        start_date: date,
        end_date: date,
        period_interval: int,
        tickers: list[str],
    ) -> list[Candle]:
        """Fetch OHLC candlestick data for nominees.

        API responses are cached at the client level (KalshiPublicClient).

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            period_interval: Candle granularity in minutes (1, 60, or 1440).
            tickers: Tickers to fetch.

        Returns:
            Sorted list of :class:`Candle` objects (by timestamp, ticker).
        """

        start_ts = _date_to_ts(start_date)
        end_ts = _date_to_ts(end_date + datetime.timedelta(days=1))

        logger.info(
            "Fetching candlestick history: %s to %s, interval=%d min, %d tickers",
            start_date,
            end_date,
            period_interval,
            len(tickers),
        )

        all_candles = self.client.get_batch_candlesticks(tickers, start_ts, end_ts, period_interval)

        result: list[Candle] = []
        for ticker, candles in all_candles.items():
            nominee = self.ticker_to_nominee.get(ticker, ticker)
            for candle in candles:
                ts = candle.get("end_period_ts", 0)
                dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
                price = candle.get("price", {})
                result.append(
                    Candle(
                        timestamp=dt,
                        date=dt.date(),
                        ticker=ticker,
                        nominee=nominee,
                        open=(price.get("open") or 0) / 100,
                        high=(price.get("high") or 0) / 100,
                        low=(price.get("low") or 0) / 100,
                        close=(price.get("close") or 0) / 100,
                        volume=candle.get("volume") or 0,
                    )
                )

        result.sort(key=lambda c: (c.timestamp, c.ticker))
        return result

    def get_daily_prices(
        self,
        start_date: date,
        end_date: date,
    ) -> list[Candle]:
        """Get daily closing prices for all nominees.

        Convenience wrapper: fetches daily (1440-min) candles.

        Returns:
            Sorted list of :class:`Candle` objects.
        """
        return self.fetch_candlestick_history(
            start_date=start_date,
            end_date=end_date,
            period_interval=1440,
            tickers=list(self.nominee_tickers.values()),
        )

    def get_hourly_prices(
        self,
        start_date: date,
        end_date: date,
    ) -> list[Candle]:
        """Get hourly OHLC candles for all nominees.

        Convenience wrapper: fetches 60-minute candles. Used for intraday
        price lookups at specific execution timestamps.

        Fetches tickers one at a time to avoid exceeding API response
        size limits (hourly data is 24× larger than daily).

        Returns:
            Sorted list of :class:`Candle` objects (empty list if no data).
        """
        all_candles: list[Candle] = []
        for ticker in self.nominee_tickers.values():
            candles = self.fetch_candlestick_history(
                start_date=start_date,
                end_date=end_date,
                period_interval=60,
                tickers=[ticker],
            )
            all_candles.extend(candles)

        all_candles.sort(key=lambda c: (c.timestamp, c.ticker))
        return all_candles

    def fetch_intraday_prices(
        self,
        event_date: date,
        event_time_et: str,
        hours_before: int,
        hours_after: int,
        period_interval: int,
        tickers: list[str],
    ) -> pd.DataFrame:
        """Fetch intraday price data around a specific event.

        Centers the fetch window around the event time and returns candles
        with a ``time_relative_hours`` column for easy pre/post analysis.

        Args:
            event_date: Date of the event (local ET date).
            event_time_et: Event time in ET as ``"HH:MM"``.
            hours_before: Hours of data before the event.
            hours_after: Hours of data after the event.
            period_interval: Candle granularity in minutes (1 or 60).
            tickers: Tickers to fetch.

        Returns:
            DataFrame with all candlestick columns plus ``event_timestamp``
            and ``time_relative_hours``.
        """
        import zoneinfo

        et = zoneinfo.ZoneInfo("America/New_York")
        hour, minute = (int(x) for x in event_time_et.split(":"))
        event_dt_et = datetime.datetime(
            event_date.year, event_date.month, event_date.day, hour, minute, tzinfo=et
        )
        event_dt_utc = event_dt_et.astimezone(datetime.UTC)

        start_dt = event_dt_utc - datetime.timedelta(hours=hours_before)
        end_dt = event_dt_utc + datetime.timedelta(hours=hours_after)

        start_date_fetch = start_dt.date()
        end_date_fetch = end_dt.date()

        logger.info(
            "Fetching intraday prices: event=%s %s ET, window=[-%dh, +%dh], interval=%d min",
            event_date,
            event_time_et,
            hours_before,
            hours_after,
            period_interval,
        )

        candles = self.fetch_candlestick_history(
            start_date=start_date_fetch,
            end_date=end_date_fetch,
            period_interval=period_interval,
            tickers=tickers,
        )

        if not candles:
            return pd.DataFrame()

        # Build DataFrame from candles, filter to window, add relative time
        rows: list[dict[str, Any]] = []
        for c in candles:
            if start_dt <= c.timestamp <= end_dt:
                rows.append(
                    {
                        "timestamp": c.timestamp,
                        "date": c.date,
                        "ticker": c.ticker,
                        "nominee": c.nominee,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                        "yes_price": c.yes_price,
                        "no_price": c.no_price,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["event_timestamp"] = event_dt_utc
        df["time_relative_hours"] = (df["timestamp"] - event_dt_utc).dt.total_seconds() / 3600.0
        return df.reset_index(drop=True)

    # -- Orderbook snapshots ---------------------------------------------------

    def fetch_current_orderbook(
        self,
        depth: int,
    ) -> pd.DataFrame:
        """Fetch current orderbook snapshots for all nominees.

        Returns one row per ticker with bid/ask distribution summary.

        Returns:
            DataFrame with columns: ticker, nominee, best_yes_bid, best_yes_ask,
            spread, midpoint, bid_depth, ask_depth.
        """
        tickers = list(self.nominee_tickers.values())

        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            nominee = self.ticker_to_nominee.get(ticker, ticker)
            try:
                ob = self.client.get_orderbook(ticker, depth=depth)
            except requests.RequestException:
                logger.warning("Failed to fetch orderbook for %s", ticker)
                continue

            yes_bids = ob.yes
            no_bids = ob.no

            # Kalshi returns bids in ascending price order; best bid = highest = last
            best_yes_bid = yes_bids[-1][0] if yes_bids else 0
            best_no_bid = no_bids[-1][0] if no_bids else 0
            best_yes_ask = 100 - best_no_bid if best_no_bid > 0 else 100

            spread = best_yes_ask - best_yes_bid if best_yes_bid > 0 else None
            midpoint = (best_yes_bid + best_yes_ask) / 2 if best_yes_bid > 0 else None

            bid_depth = sum(level[1] for level in yes_bids) if yes_bids else 0
            ask_depth = sum(level[1] for level in no_bids) if no_bids else 0

            rows.append(
                {
                    "ticker": ticker,
                    "nominee": nominee,
                    "best_yes_bid": best_yes_bid,
                    "best_yes_ask": best_yes_ask,
                    "spread": spread,
                    "midpoint": midpoint,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                }
            )

        return pd.DataFrame(rows)

    # -- Trade history ---------------------------------------------------------

    def fetch_trade_history(
        self,
        start_date: date,
        end_date: date,
    ) -> list[TradeRecord]:
        """Fetch executed trades for all nominees.

        API responses are cached at the client level (KalshiPublicClient).

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            Trade records with yes/no prices represented in dollars.

        Returning typed records instead of a DataFrame keeps the market-data
        boundary explicit and makes it harder for downstream code to rely on
        undocumented column conventions.
        """
        tickers = list(self.nominee_tickers.values())

        start_ts = _date_to_ts(start_date)
        end_ts = _date_to_ts(end_date + datetime.timedelta(days=1))

        logger.info(
            "Fetching trade history: %s to %s, %d tickers",
            start_date,
            end_date,
            len(tickers),
        )

        trades_out: list[TradeRecord] = []
        for ticker in tickers:
            nominee = self.ticker_to_nominee.get(ticker, ticker)
            try:
                trades = self.client.get_trades(ticker, start_ts=start_ts, end_ts=end_ts)
            except requests.RequestException:
                logger.warning("Failed to fetch trades for %s", ticker)
                continue

            for trade in trades:
                ts_str = trade.get("created_time", "")
                try:
                    dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue

                trades_out.append(
                    TradeRecord(
                        timestamp=dt,
                        date=dt.date(),
                        ticker=ticker,
                        nominee=nominee,
                        yes_price=(trade.get("yes_price", 0) or 0) / 100,
                        count=trade.get("count", 0) or 0,
                        taker_side=trade.get("taker_side", ""),
                    )
                )

        trades_out.sort(key=lambda trade: (trade.timestamp, trade.ticker))
        return trades_out

    # -- Full market snapshot --------------------------------------------------

    def get_market_snapshot_df(self) -> pd.DataFrame:
        """Get current comprehensive market snapshot.

        Combines price, orderbook, and volume data into a single DataFrame.

        Returns:
            DataFrame with one row per nominee: ticker, nominee, yes_bid,
            yes_ask, spread, midpoint, volume, last_price, bid_depth, ask_depth.
        """
        prices = self.get_prices()
        orderbook = self.fetch_current_orderbook(depth=10)

        price_rows: list[dict[str, Any]] = []
        for nominee, info in prices.items():
            price_rows.append(
                {
                    "ticker": info.ticker,
                    "nominee": nominee,
                    "yes_bid": info.yes_bid,
                    "yes_ask": info.yes_ask,
                    "no_bid": info.no_bid,
                    "no_ask": info.no_ask,
                    "spread": info.spread,
                    "last_price": info.last_price,
                    "volume": info.volume,
                }
            )

        price_df = pd.DataFrame(price_rows)

        if not orderbook.empty:
            price_df = price_df.merge(
                orderbook[["ticker", "midpoint", "bid_depth", "ask_depth"]],
                on="ticker",
                how="left",
            )

        return price_df.sort_values("yes_bid", ascending=False).reset_index(drop=True)
