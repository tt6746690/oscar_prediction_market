"""Execution price and edge estimation for Kalshi binary options.

Core concept: **edge** is the difference between what our model thinks an
outcome is worth and what the market charges for it, net of transaction costs.
Positive edge means we believe the market is mispricing the contract in our
favor -- the fundamental prerequisite for any trade.

Two modes of operation:

- **Live**: walks the orderbook depth to compute a volume-weighted average
  price (VWAP) for a given order size, accounting for the fact that large
  orders consume multiple price levels.
- **Backtest**: accepts a fixed execution price (e.g., daily close) since
  historical orderbooks are unavailable.

The module also estimates bid-ask spreads from historical trade data to
penalize backtest execution prices -- without this, backtests overstate
performance by assuming we can always trade at the midpoint.

Usage::

    from oscar_prediction_market.trading.edge import (
        get_execution_price,
        Edge,
        estimate_spread_from_trades,
    )

    # Live: from orderbook
    result = get_execution_price(orderbook, side="buy", n_contracts=10)

    # Backtest: construct Edge directly
    edge = Edge(
        outcome="Sinners", direction="yes",
        model_prob=0.30, execution_price=0.25, fee_type="taker",
    )
"""

import statistics

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.trading.kalshi_client import (
    Orderbook,
    estimate_fee,
)
from oscar_prediction_market.trading.oscar_market import TradeRecord
from oscar_prediction_market.trading.schema import (
    FeeType,
    PositionDirection,
    Side,
)


class ExecutionResult(BaseModel):
    """Result of computing an execution price from orderbook depth."""

    model_config = {"extra": "forbid"}

    execution_price: float = Field(..., description="VWAP execution price in dollars")
    n_contracts_requested: int = Field(
        ..., ge=0, description="Number of contracts the caller wanted to fill"
    )
    n_contracts_fillable: int = Field(
        ..., ge=0, description="Max contracts actually fillable at available depth"
    )
    is_partial: bool = Field(
        ..., description="True if orderbook lacks sufficient depth for full fill"
    )
    levels_consumed: int = Field(..., ge=0, description="Number of orderbook price levels consumed")


class Edge(BaseModel):
    """Edge computation for a single outcome.

    Stored fields are the inputs to edge computation. Derived fields
    (implied_prob, fee, gross_edge, net_edge) are
    computed from the inputs so they stay consistent.

    All edges represent BUY opportunities. When we want to sell, we liquidate
    an existing position rather than compute a separate sell edge.

    Example (buy YES at $0.20 with model_prob=0.30)::

        Edge(
            outcome="Sinners",
            direction=PositionDirection.YES,
            model_prob=0.30,
            execution_price=0.20,
            fee_type=FeeType.TAKER,
        )
        # .implied_prob = 0.20
        # .fee = 0.02  (ceil(0.07 × 0.20 × 0.80 × 100) / 100)
        # .gross_edge = 0.30 - 0.20 = 0.10
        # .net_edge = 0.10 - 0.02 = 0.08

    Example (buy NO — model thinks The Brutalist is overpriced)::

        # Market: The Brutalist YES @ $0.20 (implied 20% win probability)
        # Model:  gives it 12% chance → we want to BUY NO
        Edge(
            outcome="The Brutalist",
            direction=PositionDirection.NO,
            model_prob=0.88,        # P(Brutalist loses) = 1 - 0.12
            execution_price=0.80,   # NO ask = 1.0 - 0.20
            fee_type=FeeType.TAKER,
        )
        # .implied_prob = 0.80  (NO price)
        # .gross_edge = 0.88 - 0.80 = 0.08  (8% edge)
    """

    model_config = {"extra": "forbid"}

    outcome: str = Field(..., description="Outcome name (e.g. 'Sinners')")
    direction: PositionDirection = Field(
        ...,
        description=(
            "Whether this edge is for a YES or NO contract. "
            "For NO, model_prob is the probability the nominee LOSES "
            "and execution_price is the NO price (1.0 - YES price)."
        ),
    )
    model_prob: float = Field(
        ..., ge=0, le=1, description="Model's probability estimate for this direction"
    )
    execution_price: float = Field(
        ..., ge=0, le=1.0, description="Price in dollars (caller should adjust for spread)"
    )
    fee_type: FeeType = Field(..., description="Taker (market) or maker (limit) fee schedule")

    @property
    def implied_prob(self) -> float:
        """Implied probability = execution_price."""
        return self.execution_price

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def fee(self) -> float:
        """Estimated fee in dollars for 1 contract under the configured fee schedule."""
        return estimate_fee(self.execution_price, fee_type=self.fee_type, n_contracts=1)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def gross_edge(self) -> float:
        """Edge before fees: model_prob - implied_prob."""
        return self.model_prob - self.implied_prob

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def net_edge(self) -> float:
        """Edge after fees: gross_edge - fee."""
        return self.gross_edge - self.fee


# ============================================================================
# Execution Price
# ============================================================================


def get_execution_price(
    orderbook: Orderbook,
    side: Side,
    n_contracts: int,
) -> ExecutionResult:
    """Compute depth-weighted execution price from an orderbook.

    Why VWAP matters: an orderbook might show "best ask = 25c" but only 5
    contracts are available at that price. If you want 20 contracts, you'll
    consume the first 5 at 25c, then maybe 10 at 26c, then 5 at 28c. The
    VWAP (volume-weighted average price) tells you what you'll actually pay
    on average: (5*25 + 10*26 + 5*28) / 20 = 26.25c.

    Kalshi's orderbook uses a "dual-sided" format with YES bids and NO bids.
    NO bids at price P are economically equivalent to YES asks at (100 - P).
    See :class:`~kalshi_client.Orderbook` for details.

    For buying YES: consume NO bids -> convert to YES asks -> cheapest first.
    For selling YES: consume YES bids -> best bid first.

    The orderbook uses cents internally (Kalshi API convention). The VWAP
    is computed in cents and converted to dollars at the end.

    Example::

        >>> ob = Orderbook(yes=[[18, 50], [16, 100]], no=[[78, 30], [76, 20]])
        >>> result = get_execution_price(ob, Side.BUY, n_contracts=10)
        >>> result.execution_price  # (100 - 78) / 100 = 0.22
        0.22

    Args:
        orderbook: Orderbook from ``KalshiPublicClient.get_orderbook()``.
        side: ``Side.BUY`` (take asks) or ``Side.SELL`` (hit bids) YES contracts.
        n_contracts: Number of contracts to fill.

    Returns:
        ExecutionResult with VWAP price and fill info.
    """
    if side == Side.BUY:
        # Buying YES = taking from no bids (converted to yes ask prices)
        # Convert: no bid at price P means yes ask at 100-P
        # Sort yes asks ascending (cheapest first for buyer)
        levels: list[list[int]] = sorted(
            [[100 - price, qty] for price, qty in orderbook.no],
            key=lambda x: x[0],
        )
    else:
        # Selling YES = hitting yes bids
        # Sort descending (best bid first for seller)
        levels = sorted(orderbook.yes, key=lambda x: x[0], reverse=True)

    if not levels:
        return ExecutionResult(
            execution_price=0,
            n_contracts_requested=n_contracts,
            n_contracts_fillable=0,
            is_partial=True,
            levels_consumed=0,
        )

    total_cost = 0.0
    total_filled = 0
    levels_consumed = 0

    for price, qty in levels:
        fill_at_level = min(qty, n_contracts - total_filled)
        total_cost += price * fill_at_level
        total_filled += fill_at_level
        levels_consumed += 1
        if total_filled >= n_contracts:
            break

    vwap = total_cost / total_filled if total_filled > 0 else 0

    return ExecutionResult(
        execution_price=round(vwap / 100, 4),
        n_contracts_requested=n_contracts,
        n_contracts_fillable=total_filled,
        is_partial=total_filled < n_contracts,
        levels_consumed=levels_consumed,
    )


# ============================================================================
# Spread Estimation from Historical Trades
# ============================================================================


def estimate_spread_from_trades(
    trades: list[TradeRecord],
    default_spread: float,
    min_trades_required: int,
) -> dict[str, float]:
    """Estimate effective bid-ask spread per ticker from trade history.

    The bid-ask spread is the gap between the best price someone will pay
    (bid) and the best price someone will sell at (ask). It represents the
    cost of immediacy -- you pay the spread every time you trade.

    In backtesting, we only have trade data (not live orderbooks), so we
    estimate the spread by looking at price differences between consecutive
    trades on opposite sides. If a buy-taker trade executes at 26c and the
    next sell-taker trade executes at 24c, the spread was roughly 2c.

    We take the median of these gaps (robust to outliers) and halve it to
    get the **one-way** spread penalty -- the cost we'd pay on each side of
    a round-trip trade.

    Returns spreads in **dollars**. Falls back to ``default_spread / 2`` when
    there aren't enough trades for a reliable estimate.

    The function deliberately uses a simple, robust estimator rather than a
    more ambitious microstructure model because the backtest only needs a
    reasonable execution penalty, not an exact reconstruction of historical
    orderbooks. The median side-switch gap captures the scale of spread costs
    while staying stable in the presence of noisy trade prints.

    Returns empty dict if ``trades`` is empty.

    Args:
        trades: Trade records from ``OscarMarket.fetch_trade_history()``.
        default_spread: Fallback full spread in dollars if insufficient data.
        min_trades_required: Minimum trades per ticker for reliable estimate.

    Returns:
        Dict mapping ticker to estimated one-way spread in dollars.
    """
    if not trades:
        return {}

    spreads: dict[str, float] = {}
    trades_by_ticker: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        trades_by_ticker.setdefault(trade.ticker, []).append(trade)

    for ticker, ticker_trades in trades_by_ticker.items():
        if len(ticker_trades) < min_trades_required:
            spreads[ticker] = default_spread / 2
            continue

        sorted_trades = sorted(ticker_trades, key=lambda trade: trade.timestamp)
        gaps = [
            abs(curr.yes_price - prev.yes_price)
            for prev, curr in zip(sorted_trades, sorted_trades[1:], strict=False)
            if curr.taker_side != prev.taker_side
        ]

        if len(gaps) >= 5:
            median_spread = statistics.median(gaps)
            spreads[ticker] = median_spread / 2
        else:
            spreads[ticker] = default_spread / 2

    return spreads
