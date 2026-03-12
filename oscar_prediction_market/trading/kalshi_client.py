"""Generic Kalshi API client.

Class hierarchy
---------------
- ``KalshiPublicClient``  -- unauthenticated (markets, candlesticks, trades, orderbook)
- ``KalshiClient``        -- authenticated read-only (positions, fills, orders, balance)
- (future) ``KalshiTradingClient`` -- authenticated read+write (create/cancel orders)

Authentication
--------------
Authenticated clients load credentials from files on disk:
- **~/.kalshi_access_key**: API Key ID (UUID)
- **~/.kalshi_private_key**: RSA private key (PEM file)

Override paths via env vars ``KALSHI_ACCESS_KEY_FILE`` / ``KALSHI_PRIVATE_KEY_FILE``.
See ``constants.py`` and ``.env`` for details.

Fee structure
-------------
Kalshi fees use a variance-based formula: ``ceil(rate × C × P × (1 − P))``
where P is the contract price in dollars and C is the number of contracts.
Fees peak at P = 50c (maximum uncertainty) and vanish at P = 0 or P = $1.

Rates:
- **Taker fee**: 7% (orders that immediately match)
- **Maker fee**: 1.75% (resting orders that are later filled)

Source: https://kalshi.com/docs/kalshi-fee-schedule.pdf

The API also returns exact fees on fills (``fee_cost``) and orders
(``taker_fees_dollars``, ``maker_fees_dollars``).

For Oscar Best Picture-specific code, see :mod:`oscar_market`.

Usage
-----
    from oscar_prediction_market.trading.kalshi_client import (
        KalshiPublicClient,
        KalshiClient,
    )

    pub = KalshiPublicClient()
    markets = pub.get_event_markets(event_ticker="KXOSCARPIC-26")

    client = KalshiClient()
    positions = client.get_positions(event_ticker="KXOSCARPIC-26")
"""

import base64
import datetime
import logging
import math
import time
from pathlib import Path
from typing import Any

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from diskcache import Cache
from pydantic import BaseModel, Field

from oscar_prediction_market.constants import (
    CACHE_DIR,
    KALSHI_ACCESS_KEY_FILE,
    KALSHI_PRIVATE_KEY_FILE,
)
from oscar_prediction_market.trading.schema import FeeType

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"


# -- Pydantic models for API responses ----------------------------------------


class Orderbook(BaseModel):
    """Kalshi dual-sided orderbook.

    Kalshi orderbooks contain YES bids and NO bids, each as a list of
    ``[price_cents, quantity]`` levels. NO bids at price P are economically
    equivalent to YES asks at price ``100 - P``.

    Example (Oscar BP nominee "Sinners" at ~20c)::

        Orderbook(
            yes=[[16, 100], [18, 50]],  # 100 YES at 16c, 50 YES at 18c (ascending)
            no=[[76, 20], [78, 30]],    # = YES ask at 24c (qty 20), 22c (qty 30)
        )

    Levels are in ascending price order (cheapest first).  Best bid = last:
    ``yes[-1][0]`` → 18c, ``100 - no[-1][0]`` → 22c (best YES ask).
    The spread is ``best_yes_ask - best_yes_bid`` → 22 - 18 = 4c.
    """

    model_config = {"extra": "forbid"}

    # The API returns both cent-based (yes/no) and dollar-based (yes_dollars/no_dollars)
    # representations of the same orderbook. All code uses the cent-based fields only.
    # The dollar fields are accepted to satisfy extra="forbid" but never read.
    yes: list[list[int]] = Field(default_factory=list, description="YES bid levels [[price, qty]]")
    no: list[list[int]] = Field(default_factory=list, description="NO bid levels [[price, qty]]")
    yes_dollars: list[list[str | int]] = Field(
        default_factory=list, description="YES dollar levels — accepted but unused"
    )
    no_dollars: list[list[str | int]] = Field(
        default_factory=list, description="NO dollar levels — accepted but unused"
    )


class KalshiMarket(BaseModel):
    """Kalshi market data (subset of API response fields we use).

    The Kalshi API returns ~30+ fields per market. This model captures the
    fields accessed by our trading code, with ``extra="allow"`` to silently
    pass through any others.

    Example (Oscar BP nominee)::

        KalshiMarket(
            ticker="KXOSCARPIC-26-SIN",
            event_ticker="KXOSCARPIC-26",
            title="Will Sinners win Best Picture at the 2026 Oscars?",
            yes_bid=18, yes_ask=22,
            last_price=20, volume=1523,
            status="active",
        )
    """

    model_config = {"extra": "allow"}

    ticker: str
    event_ticker: str = ""
    title: str = ""
    yes_bid: int = 0
    yes_ask: int = 0
    last_price: int = 0
    volume: int = 0
    status: str = ""


# Client-level cache for Kalshi API responses.
# Only historical/immutable endpoints are cached (candlesticks, trades).
# Live endpoints (markets, orderbook, portfolio) always fetch fresh data.
KALSHI_CACHE_DIR = CACHE_DIR / "kalshi"

# Path fragments that indicate historical/immutable data safe to cache.
# Everything else (live prices, orderbook, portfolio) is NOT cached.
_CACHEABLE_PATH_FRAGMENTS = {"/candlesticks", "/trades"}


# -- Fee schedule constants ---------------------------------------------------
# Source: https://kalshi.com/docs/kalshi-fee-schedule.pdf
#
# Formula: fees = round_up(rate × C × P × (1 - P))
#   P = contract price in dollars (e.g. 50 cents = 0.50)
#   C = number of contracts
#   round_up = ceiling to next cent
#
# The P*(1-P) term means fees peak at P=50c (max uncertainty) and vanish
# at the extremes (P=0 or P=$1). This is the variance of a Bernoulli(P).

TAKER_FEE_RATE = 0.07
MAKER_FEE_RATE = 0.0175

_FEE_RATES: dict[FeeType, float] = {
    FeeType.TAKER: TAKER_FEE_RATE,
    FeeType.MAKER: MAKER_FEE_RATE,
}


def estimate_fee(
    price: float,
    fee_type: FeeType,
    n_contracts: int = 1,
) -> float:
    """Estimate Kalshi trading fee in dollars.

    Uses the variance-based formula from the Kalshi fee schedule::

        fee = ceil_to_cent(rate × C × P × (1 − P))

    where P is the contract price in dollars [0.0, 1.0], C = n_contracts,
    and ``ceil_to_cent`` rounds up to the next cent.

    The P × (1 − P) term is the variance of a Bernoulli(P) random variable.
    Intuitively, fees are highest when the outcome is most uncertain (P ≈ $0.50)
    and lowest when the market is highly confident either way (P near $0 or $1).

    Examples at C=1:
      - P = $0.50: fee = ceil(0.07 × 0.50 × 0.50) = $0.02
      - P = $0.25: fee = ceil(0.07 × 0.25 × 0.75) = $0.02
      - P = $0.10: fee = ceil(0.07 × 0.10 × 0.90) = $0.01
      - P = $0.05: fee = ceil(0.07 × 0.05 × 0.95) = $0.01

    Args:
        price: Contract price in dollars (0.0-1.0 for binary options).
        n_contracts: Number of contracts in the order.
        fee_type: ``"taker"`` (0.07) for market orders that immediately match,
            ``"maker"`` (0.0175) for resting/limit orders filled later.

    Returns:
        Total fee in dollars (ceiling to nearest cent). Returns 0.0 when price
        is 0 or 1 (no uncertainty → no fee).
    """
    rate = _FEE_RATES[FeeType(fee_type)]
    fee_dollars = rate * n_contracts * price * (1 - price)
    return math.ceil(fee_dollars * 100) / 100


# -- Public client (unauthenticated) ------------------------------------------


class KalshiPublicClient:
    """Unauthenticated Kalshi API client.

    Provides access to public market data: prices, candlesticks, trades,
    and orderbook.  No credentials required.

    GET requests are cached to ``storage/cache/kalshi/`` by default to avoid
    redundant API calls. Real-time endpoints (orderbook, portfolio) bypass the
    cache. Set ``cache_dir=None`` to disable caching entirely.
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.2,
        cache_dir: Path | str | None = KALSHI_CACHE_DIR,
    ) -> None:
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        self._cache: Cache | None = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(cache_dir))

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _is_cacheable(self, path: str) -> bool:
        """Check if a request path should be cached (whitelist approach)."""
        return any(frag in path for frag in _CACHEABLE_PATH_FRAGMENTS)

    def _get(self, path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Make a GET request to the Kalshi API, with optional caching."""
        # Check cache first (only for unauthenticated, cacheable requests)
        if self._cache is not None and not headers and self._is_cacheable(path):
            cached = self._cache.get(path)
            if cached is not None:
                logger.debug("Cache hit: %s", path)
                return cached  # type: ignore[no-any-return]

        self._rate_limit()
        url = BASE_URL + path
        response = requests.get(url, headers=headers or {})
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # Cache the response (unauthenticated + cacheable only)
        if self._cache is not None and not headers and self._is_cacheable(path):
            self._cache.set(path, data)
            logger.debug("Cached: %s", path)

        return data

    # -- Market data -----------------------------------------------------------

    def get_market(self, ticker: str) -> KalshiMarket:
        """Get current market data for a single market.

        Example::

            >>> client = KalshiPublicClient()
            >>> m = client.get_market("KXOSCARPIC-26-SIN")
            >>> m.ticker
            'KXOSCARPIC-26-SIN'
            >>> m.yes_bid, m.yes_ask
            (18, 22)
        """
        data = self._get(f"{API_PREFIX}/markets/{ticker}")
        return KalshiMarket.model_validate(data["market"])

    def get_event_markets(self, event_ticker: str) -> list[KalshiMarket]:
        """Get all markets for an event.

        Example::

            >>> client = KalshiPublicClient()
            >>> markets = client.get_event_markets("KXOSCARPIC-26")
            >>> len(markets)  # one per nominee
            11
            >>> markets[0].ticker
            'KXOSCARPIC-26-ONE'
        """
        data = self._get(f"{API_PREFIX}/events/{event_ticker}")
        return [KalshiMarket.model_validate(m) for m in data["markets"]]

    def get_orderbook(self, ticker: str, depth: int = 0) -> Orderbook:
        """Get current orderbook (works without auth as of 2026-02).

        Example::

            >>> client = KalshiPublicClient()
            >>> ob = client.get_orderbook("KXOSCARPIC-26-SIN", depth=5)
            >>> ob.yes  # YES bid levels: [[price_cents, quantity], ...]
            [[18, 50], [16, 100]]
            >>> ob.no   # NO bid levels (YES ask = 100 - price)
            [[78, 30], [76, 20]]
        """
        path = f"{API_PREFIX}/markets/{ticker}/orderbook"
        if depth > 0:
            path += f"?depth={depth}"
        data = self._get(path)
        return Orderbook.model_validate(data.get("orderbook", {}))

    # -- Candlesticks ----------------------------------------------------------

    def get_batch_candlesticks(
        self,
        tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get candlestick data for multiple markets at once.

        Args:
            tickers: Market tickers to fetch.
            start_ts: Start Unix timestamp (inclusive).
            end_ts: End Unix timestamp (inclusive).
            period_interval: Candle granularity in minutes (1, 60, or 1440).
        """
        tickers_str = ",".join(tickers)
        path = (
            f"{API_PREFIX}/markets/candlesticks"
            f"?market_tickers={tickers_str}"
            f"&start_ts={start_ts}&end_ts={end_ts}"
            f"&period_interval={period_interval}"
        )
        data = self._get(path)
        result: dict[str, list[dict[str, Any]]] = {}
        for market in data.get("markets", []):
            result[market["market_ticker"]] = market.get("candlesticks", [])
        return result

    # -- Trades ----------------------------------------------------------------

    def get_trades(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get executed trades for a market, with automatic pagination.

        Fetches all pages of trades using cursor-based pagination. Logs a
        warning if more than 3 pages are needed (indicates a large time range).

        Args:
            ticker: Market ticker.
            start_ts: Start Unix timestamp (inclusive).
            end_ts: End Unix timestamp (inclusive).
            limit: Page size (max trades per request).
        """
        all_trades: list[dict[str, Any]] = []
        cursor: str | None = None
        page = 0

        while True:
            path = (
                f"{API_PREFIX}/markets/trades?ticker={ticker}&limit={limit}"
                f"&min_ts={start_ts}&max_ts={end_ts}"
            )
            if cursor:
                path += f"&cursor={cursor}"

            data = self._get(path)
            trades = data.get("trades", [])
            all_trades.extend(trades)
            page += 1

            cursor = data.get("cursor")
            if not cursor or len(trades) < limit:
                break

            if page == 3:
                logger.info(
                    "get_trades(%s): fetched %d trades across %d pages, continuing...",
                    ticker,
                    len(all_trades),
                    page,
                )

        if page > 3:
            logger.info(
                "get_trades(%s): completed with %d trades across %d pages",
                ticker,
                len(all_trades),
                page,
            )

        return all_trades


# -- Authenticated client (read-only) -----------------------------------------


class KalshiClient(KalshiPublicClient):
    """Authenticated Kalshi API client (read-only).

    Inherits all public endpoints and adds portfolio read access:
    positions, fills, orders, balance.
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.2,
    ) -> None:
        super().__init__(rate_limit_delay=rate_limit_delay)

        if not KALSHI_ACCESS_KEY_FILE.exists():
            raise ValueError(
                f"Kalshi access key file not found at {KALSHI_ACCESS_KEY_FILE}. "
                "Place your API key UUID there (or set KALSHI_ACCESS_KEY_FILE env var). "
                "Get credentials at https://kalshi.com/api"
            )
        self._api_key_id = KALSHI_ACCESS_KEY_FILE.read_text().strip()

        if not KALSHI_PRIVATE_KEY_FILE.exists():
            raise ValueError(
                f"Kalshi private key file not found at {KALSHI_PRIVATE_KEY_FILE}. "
                "Place your RSA private key PEM there (or set KALSHI_PRIVATE_KEY_FILE env var). "
                "Get credentials at https://kalshi.com/api"
            )
        pk_pem = KALSHI_PRIVATE_KEY_FILE.read_text().strip()
        key = serialization.load_pem_private_key(
            pk_pem.encode(), password=None, backend=default_backend()
        )
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError(f"Expected RSA private key, got {type(key).__name__}")
        self._private_key: rsa.RSAPrivateKey = key

    def _sign_request(self, timestamp: str, method: str, path: str) -> str:
        """Create RSA-PSS signature for authenticated requests."""
        path_without_query = path.split("?")[0]
        message = f"{timestamp}{method}{path_without_query}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _get_auth_headers(self, method: str, path: str) -> dict[str, str]:
        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        signature = self._sign_request(timestamp, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _auth_get(self, path: str) -> dict[str, Any]:
        """Make an authenticated GET request."""
        headers = self._get_auth_headers("GET", path)
        return self._get(path, headers=headers)

    # -- Positions -------------------------------------------------------------

    def get_positions(
        self,
        event_ticker: str | None = None,
        ticker: str | None = None,
    ) -> dict[str, Any]:
        """Get current positions.

        Returns dict with 'market_positions' and 'event_positions'.
        Each market_position has: ticker, position, total_traded,
        market_exposure_dollars, realized_pnl_dollars, fees_paid_dollars.
        """
        path = f"{API_PREFIX}/portfolio/positions?limit=100"
        if event_ticker:
            path += f"&event_ticker={event_ticker}"
        if ticker:
            path += f"&ticker={ticker}"
        return self._auth_get(path)

    # -- Fills -----------------------------------------------------------------

    def get_fills(
        self,
        ticker: str | None = None,
        event_ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get trade fills (matched orders).

        Each fill: ticker, side, action, count, yes_price, no_price,
        is_taker, fee_cost, created_time.
        """
        path = f"{API_PREFIX}/portfolio/fills?limit={limit}"
        if ticker:
            path += f"&ticker={ticker}"
        if min_ts is not None:
            path += f"&min_ts={min_ts}"
        if max_ts is not None:
            path += f"&max_ts={max_ts}"
        data = self._auth_get(path)
        fills: list[dict[str, Any]] = data.get("fills", [])
        if event_ticker:
            fills = [f for f in fills if f.get("ticker", "").startswith(event_ticker)]
        return fills

    # -- Orders ----------------------------------------------------------------

    def get_orders(
        self,
        event_ticker: str | None = None,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get orders (resting, executed, canceled).

        Each order: ticker, side, action, type, status, yes_price,
        fill_count, taker_fees_dollars, maker_fees_dollars, created_time.
        """
        path = f"{API_PREFIX}/portfolio/orders?limit={limit}"
        if event_ticker:
            path += f"&event_ticker={event_ticker}"
        if ticker:
            path += f"&ticker={ticker}"
        if status:
            path += f"&status={status}"
        data = self._auth_get(path)
        return data.get("orders", [])  # type: ignore[no-any-return]

    # -- Balance ---------------------------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        return self._auth_get(f"{API_PREFIX}/portfolio/balance")
