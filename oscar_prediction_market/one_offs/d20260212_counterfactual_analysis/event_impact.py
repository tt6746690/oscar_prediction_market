"""Reusable tools for analyzing price impact around market-moving events.

Provides generic functions for:
1. Fetching and structuring candlestick (OHLC) data around a specific event datetime
2. Computing price changes before/after the event for each market
3. Summarizing position P&L given a set of fills

These are building blocks for analyzing any precursor award (DGA, BAFTA, SAG, PGA)
or other catalyst's impact on Kalshi Oscar Best Picture market prices.

Usage
-----
    from oscar_prediction_market.one_offs.d20260212_counterfactual_analysis.event_impact import (
        fetch_candles_around_event,
        compute_price_changes,
        summarize_position_pnl,
    )

    client = KalshiPublicClient()
    candles = fetch_candles_around_event(client, event_dt, tickers, ...)
    changes = compute_price_changes(candles, pre_date="02/07", post_date="02/08")
"""

import datetime

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.trading.kalshi_client import (
    KalshiPublicClient,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS

_TICKER_TO_NOMINEE = OSCAR_MARKETS.get_category_data(
    OscarCategory.BEST_PICTURE, 2026
).ticker_to_nominee


def _ts(year: int, month: int, day: int) -> int:
    """Convert date to Unix timestamp (UTC midnight)."""
    return int(datetime.datetime(year, month, day, tzinfo=datetime.UTC).timestamp())


# -- Candle fetching -----------------------------------------------------------


def fetch_daily_candles(
    client: KalshiPublicClient,
    tickers: list[str],
    days_back: int = 30,
) -> dict[str, list[dict]]:
    """Fetch daily OHLC candles for the given tickers.

    Returns {ticker: [candle_dict, ...]} where each candle has
    'end_period_ts', 'price' (with open/high/low/close), 'volume'.
    """
    now = datetime.datetime.now(tz=datetime.UTC)
    end_ts = int(now.timestamp())
    start_ts = int((now - datetime.timedelta(days=days_back)).timestamp())
    return client.get_batch_candlesticks(tickers, start_ts, end_ts, period_interval=1440)


def fetch_candles_around_event(
    client: KalshiPublicClient,
    event_date: datetime.date,
    tickers: list[str],
    period_interval: int = 60,
    days_before: int = 1,
    days_after: int = 1,
) -> dict[str, list[dict]]:
    """Fetch candles around a specific event date.

    Parameters
    ----------
    event_date : date
        The date of the event (e.g. DGA ceremony).
    tickers : list[str]
        Market tickers to fetch.
    period_interval : int
        Candle granularity in minutes: 1, 60, or 1440.
    days_before / days_after : int
        How many days before/after the event date to include.

    Returns
    -------
    dict mapping ticker -> list of candle dicts.
    """
    start_dt = event_date - datetime.timedelta(days=days_before)
    end_dt = event_date + datetime.timedelta(days=days_after)
    start_ts = _ts(start_dt.year, start_dt.month, start_dt.day)
    end_ts = _ts(end_dt.year, end_dt.month, end_dt.day)
    return client.get_batch_candlesticks(tickers, start_ts, end_ts, period_interval)


# -- Price change computation --------------------------------------------------


class PriceChange(BaseModel):
    """Price change for a single market between two dates."""

    model_config = {"extra": "forbid"}

    ticker: str
    nominee: str
    pre_close: int | None
    post_close: int | None
    change: int | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def change_str(self) -> str:
        if self.change is None:
            return "?"
        sign = "+" if self.change >= 0 else ""
        return f"{sign}{self.change}¢"


def _extract_close_prices(candles: list[dict]) -> dict[str, int | None]:
    """Extract {date_str -> close_price} from a list of candles.

    date_str is in "MM/DD" format. Close price is in cents.
    """
    prices: dict[str, int | None] = {}
    for c in candles:
        dt = datetime.datetime.fromtimestamp(c["end_period_ts"], tz=datetime.UTC)
        date_str = dt.strftime("%m/%d")
        prices[date_str] = c.get("price", {}).get("close")
    return prices


def compute_price_changes(
    candle_data: dict[str, list[dict]],
    pre_date: str,
    post_date: str,
) -> list[PriceChange]:
    """Compute price change for each ticker between pre_date and post_date.

    Parameters
    ----------
    candle_data : dict
        {ticker: [candle_dict, ...]} as returned by fetch_daily_candles.
    pre_date : str
        Date string in "MM/DD" format for the pre-event close.
    post_date : str
        Date string in "MM/DD" format for the post-event close.

    Returns
    -------
    List of PriceChange, sorted by post_close descending.
    """
    results: list[PriceChange] = []
    for ticker, candles in candle_data.items():
        nominee = _TICKER_TO_NOMINEE.get(ticker, ticker)
        prices = _extract_close_prices(candles)
        pre = prices.get(pre_date)
        post = prices.get(post_date)
        change = (post - pre) if (pre is not None and post is not None) else None
        results.append(
            PriceChange(
                ticker=ticker, nominee=nominee, pre_close=pre, post_close=post, change=change
            )
        )
    results.sort(key=lambda x: x.post_close or 0, reverse=True)
    return results


def get_close_on_date(candles: list[dict], date_str: str) -> int | None:
    """Get close price on a specific date (MM/DD format) from candle list."""
    prices = _extract_close_prices(candles)
    return prices.get(date_str)


# -- Position P&L -------------------------------------------------------------


class Fill(BaseModel):
    """A single trade fill."""

    model_config = {"extra": "forbid"}

    nominee: str
    side: str = Field(..., description="'yes' or 'no'")
    action: str = Field(..., description="'buy' or 'sell'")
    contracts: int = Field(..., ge=0)
    yes_price: float = Field(..., ge=0, le=1.0)
    fee_dollars: float = Field(..., ge=0)
    label: str = Field(default="", description="Optional label (e.g. 'pre-dga', 'post-dga')")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_contract(self) -> float:
        """Cost in dollars per contract."""
        if self.side == "yes":
            return self.yes_price
        return 1.0 - self.yes_price

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost(self) -> float:
        """Total outlay in dollars (excludes fees)."""
        return self.contracts * self.cost_per_contract


class PositionSummary(BaseModel):
    """Aggregated position info for one nominee."""

    model_config = {"extra": "forbid"}

    nominee: str
    side: str
    total_contracts: int = Field(..., ge=0)
    total_cost: float
    total_fees: float = Field(..., ge=0)
    realized_pnl: float
    is_open: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_pnl(self) -> float:
        return self.realized_pnl - self.total_fees


def summarize_position_pnl(fills: list[Fill]) -> list[PositionSummary]:
    """Aggregate fills into per-nominee position summaries.

    Groups buys and sells by nominee. Buy fills accumulate cost;
    sell fills contribute to realized P&L.

    Returns list of PositionSummary sorted by total_cost descending.
    """
    # Accumulate by nominee
    buys: dict[str, list[Fill]] = {}
    sells: dict[str, list[Fill]] = {}
    for f in fills:
        bucket = buys if f.action == "buy" else sells
        bucket.setdefault(f.nominee, []).append(f)

    summaries: list[PositionSummary] = []
    all_nominees = sorted(set(buys.keys()) | set(sells.keys()))

    for nominee in all_nominees:
        buy_fills = buys.get(nominee, [])
        sell_fills = sells.get(nominee, [])

        buy_contracts = sum(f.contracts for f in buy_fills)
        sell_contracts = sum(f.contracts for f in sell_fills)
        buy_cost = sum(f.total_cost for f in buy_fills)
        sell_revenue = sum(f.total_cost for f in sell_fills)
        total_fees = sum(f.fee_dollars for f in buy_fills + sell_fills)

        side = buy_fills[0].side if buy_fills else (sell_fills[0].side if sell_fills else "?")
        is_open = buy_contracts != sell_contracts
        realized_pnl = sell_revenue - buy_cost if sell_contracts > 0 else 0.0

        summaries.append(
            PositionSummary(
                nominee=nominee,
                side=side,
                total_contracts=buy_contracts,
                total_cost=buy_cost,
                total_fees=total_fees,
                realized_pnl=realized_pnl,
                is_open=is_open,
            )
        )

    summaries.sort(key=lambda x: x.total_cost, reverse=True)
    return summaries


# -- Printing helpers ----------------------------------------------------------


def print_price_changes(
    changes: list[PriceChange],
    pre_label: str = "Pre-event",
    post_label: str = "Post-event",
    extra_dates: dict[str, tuple[str, dict[str, list[dict]]]] | None = None,
) -> None:
    """Print a formatted table of price changes.

    Parameters
    ----------
    changes : list[PriceChange]
        Price changes to display.
    pre_label / post_label : str
        Column headers for the pre/post prices.
    extra_dates : dict, optional
        Additional date columns. Maps column_label -> (date_str, candle_data).
        candle_data is the same {ticker: [candles]} dict used elsewhere.
    """
    print(f"\n  {'Nominee':<30} {pre_label:>10} {post_label:>11} {'Change':>8}")
    print(f"  {'-' * 62}")

    for pc in changes:
        pre = f"{pc.pre_close}¢" if pc.pre_close is not None else "?"
        post = f"{pc.post_close}¢" if pc.post_close is not None else "?"
        print(f"  {pc.nominee:<30} {pre:>10} {post:>11} {pc.change_str:>8}")


def print_position_summary(summaries: list[PositionSummary]) -> None:
    """Print formatted position P&L summary."""
    print(
        f"\n  {'Nominee':<25} {'Side':<5} {'Cts':>5} {'Cost':>9}"
        f" {'Fees':>7} {'R. PnL':>8} {'Net':>9} {'Status':>7}"
    )
    print(f"  {'-' * 78}")

    total_cost = 0.0
    total_fees = 0.0
    total_rpnl = 0.0

    for s in summaries:
        status = "OPEN" if s.is_open else "CLOSED"
        print(
            f"  {s.nominee:<25} {s.side:<5} {s.total_contracts:>5}"
            f" ${s.total_cost:>7.2f} ${s.total_fees:>5.2f}"
            f" ${s.realized_pnl:>+7.2f} ${s.net_pnl:>+7.2f} {status:>7}"
        )
        total_cost += s.total_cost
        total_fees += s.total_fees
        total_rpnl += s.realized_pnl

    print(f"  {'-' * 78}")
    net = total_rpnl - total_fees
    print(
        f"  {'TOTAL':<25} {'':5} {'':>5}"
        f" ${total_cost:>7.2f} ${total_fees:>5.2f}"
        f" ${total_rpnl:>+7.2f} ${net:>+7.2f}"
    )
