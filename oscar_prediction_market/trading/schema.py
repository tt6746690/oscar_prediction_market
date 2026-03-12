"""Shared types and data models for the trading module.

Centralizes StrEnums, configuration models, and core data models used across
the trading pipeline (backtest engine, signal generation, Kelly sizing,
portfolio management). Import all fundamental types from here.

**Enums**: Side, PositionDirection, FeeType, TradeAction, KellyMode, BankrollMode
**Config**: KellyConfig, TradingConfig
**Data**: Position, MarketQuotes, KellyAllocation, Fill, SettlementResult

Design notes:

- All prices use *dollars* [0.0, 1.0] for Kalshi binary options.
- Required fields use ``Field(...)`` — no implicit defaults for semantic
  parameters.
- ``model_config = {"extra": "forbid"}`` on every model to catch typos
  at construction time.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field, model_validator


class Side(StrEnum):
    """Order side for binary option contracts."""

    BUY = "buy"
    SELL = "sell"


class PositionDirection(StrEnum):
    """Direction of a binary option position, aligned with Kalshi terminology.

    On Kalshi, each binary event has YES and NO contracts:

    - **YES** pays $1 if the event occurs (nominee wins), $0 otherwise.
    - **NO** pays $1 if the event does NOT occur (nominee loses), $0 otherwise.
    """

    YES = "yes"
    NO = "no"

    @property
    def opposite(self) -> "PositionDirection":
        """Return the opposite direction: YES <-> NO."""
        return PositionDirection.NO if self == PositionDirection.YES else PositionDirection.YES


class FeeType(StrEnum):
    """Kalshi fee schedule type.

    - TAKER (7%): market orders that immediately match existing orders.
    - MAKER (1.75%): resting/limit orders that provide liquidity.
    """

    TAKER = "taker"
    MAKER = "maker"


class TradeAction(StrEnum):
    """Signal action recommendation."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class KellyMode(StrEnum):
    """Kelly criterion sizing mode.

    - INDEPENDENT: sizes each outcome separately, ignoring correlations.
      Simple and fast, but oversizes the total portfolio when outcomes are
      mutually exclusive (buying A implicitly hedges against B).
    - MULTI_OUTCOME: jointly optimizes across mutually exclusive outcomes
      by maximizing expected log-wealth.  Required when both YES and NO
      positions are allowed, because independent Kelly can double-count
      the bankroll (sizing YES-A and NO-B as if they were uncorrelated,
      when in fact exactly one outcome wins).
    """

    INDEPENDENT = "independent"
    MULTI_OUTCOME = "multi_outcome"


class BankrollMode(StrEnum):
    """How the backtest engine determines bankroll for Kelly sizing.

    - FIXED: Kelly sizes against initial bankroll (path-independent).
    - DYNAMIC: sizes against current mark-to-market wealth (compounding).
    """

    FIXED = "fixed"
    DYNAMIC = "dynamic"


# Sentinel value for sell_edge_threshold that effectively disables selling.
NEVER_SELL_THRESHOLD: float = -1.0


# ============================================================================
# Core data models
# ============================================================================


class Position(BaseModel):
    """Current position in one outcome.

    Self-describing: includes the outcome name and ticker alongside the
    holding details, so a Position can be passed around without external
    context.
    """

    model_config = {"extra": "forbid"}

    outcome: str = Field(..., description="Outcome name this position is for (e.g. 'Sinners')")
    ticker: str = Field(default="", description="Kalshi market ticker (e.g. 'KXOSCARPIC-26-SIN')")
    direction: PositionDirection = Field(
        ..., description="Whether this is a YES or NO contract position"
    )
    contracts: int = Field(..., ge=0, description="Number of contracts currently held")
    avg_cost: float = Field(
        ..., ge=0, le=1.0, description="Volume-weighted average purchase price in dollars"
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def outlay_dollars(self) -> float:
        """Total cost basis in dollars = contracts * avg_cost."""
        return round(self.contracts * self.avg_cost, 2)


# ============================================================================
# Configuration
# ============================================================================


class KellyConfig(BaseModel):
    """Configuration for Kelly criterion sizing.

    Includes the sizing mode (independent vs. multi-outcome) alongside all
    Kelly parameters, so a single object fully describes position sizing.

    Reasonable starting values::

        KellyConfig(
            kelly_mode="multi_outcome",
            bankroll=1000, kelly_fraction=0.25,
            buy_edge_threshold=0.05, max_position_per_outcome=250,
            max_total_exposure=500,
        )
    """

    model_config = {"extra": "forbid"}

    kelly_mode: KellyMode = Field(
        ...,
        description=(
            "Sizing mode: 'independent' treats each outcome as a standalone bet; "
            "'multi_outcome' jointly optimizes across mutually exclusive outcomes"
        ),
    )
    bankroll: float = Field(
        ...,
        gt=0,
        description=(
            "Total capital available for Kelly sizing in USD. In fixed bankroll mode "
            "this stays constant; in dynamic mode it updates to current wealth each snapshot."
        ),
    )
    kelly_fraction: float = Field(
        ...,
        gt=0,
        le=1,
        description=(
            "Fraction of full Kelly to bet (1.0 = full Kelly, 0.25 = quarter Kelly). "
            "Lower values reduce variance at the cost of slower growth."
        ),
    )
    buy_edge_threshold: float = Field(
        ...,
        ge=0,
        description="Minimum net edge (after fees) required to open a new position (fraction)",
    )
    max_position_per_outcome: float = Field(
        ..., gt=0, description="Maximum USD outlay allowed for any single outcome"
    )
    max_total_exposure: float = Field(
        ..., gt=0, description="Maximum total USD outlay across all outcomes combined"
    )


class TradingConfig(BaseModel):
    """Strategy parameters for both live trading and backtesting.

    All fields are required — callers must explicitly specify every parameter.
    The same model is used by ``generate_signals()``, ``BacktestEngine``, and
    live trading.

    Reasonable starting values for experimentation::

        TradingConfig(
            kelly=KellyConfig(
                kelly_mode="multi_outcome",
                bankroll=1000, kelly_fraction=0.25,
                buy_edge_threshold=0.05, max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            fee_type="taker",
            limit_price_offset=0.0,
            sell_edge_threshold=-0.03,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
        )
    """

    model_config = {"extra": "forbid"}

    kelly: KellyConfig = Field(..., description="Kelly criterion sizing parameters")
    fee_type: FeeType = Field(
        ...,
        description="Fee schedule: 'taker' (7%) for market orders, 'maker' (1.75%) for limit orders",
    )
    limit_price_offset: float = Field(
        ...,
        ge=0,
        description=(
            "Offset from bid for limit order pricing, in dollars. "
            "For buys: execution_price = bid + offset. "
            "0.0 = use ask (taker behavior). 0.01 = bid+1¢ (aggressive limit). "
            "Non-zero offset requires fee_type=maker."
        ),
    )
    sell_edge_threshold: float = Field(
        ...,
        description=(
            "Sell when net edge drops below this (should be negative, "
            "accounting for round-trip transaction costs)"
        ),
    )
    min_price: float = Field(
        ...,
        ge=0,
        description=(
            "Skip contracts priced below this level (dollars). Low-price contracts "
            "have disproportionate fee drag."
        ),
    )
    allowed_directions: frozenset[PositionDirection] = Field(
        ...,
        description=(
            "Which contract directions to consider. "
            "Use frozenset({PositionDirection.YES}) for YES-only, "
            "frozenset({PositionDirection.YES, PositionDirection.NO}) for both."
        ),
    )

    @model_validator(mode="after")
    def _validate_offset_requires_maker(self) -> "TradingConfig":
        if self.limit_price_offset > 0 and self.fee_type != FeeType.MAKER:
            msg = (
                f"limit_price_offset={self.limit_price_offset} requires "
                f"fee_type='maker', got '{self.fee_type.value}'"
            )
            raise ValueError(msg)
        return self


# ============================================================================
# Market data
# ============================================================================


class MarketQuotes(BaseModel):
    """Spread-adjusted bid/ask prices for YES and NO contracts across all outcomes.

    Follows the Kalshi orderbook convention where the primary data is
    **YES bids** and **NO bids**. Ask prices are derived from the opposite
    side's bids:

    - ``yes_ask = 1.0 - no_bid``  (buying YES = matching a NO bidder)
    - ``no_ask  = 1.0 - yes_bid`` (buying NO  = matching a YES bidder)

    This is the execution-price representation used by the signal pipeline.
    Raw close prices (``dict[str, float]``) are used for mark-to-market
    valuation; ``MarketQuotes`` adds spread adjustment for realistic
    execution simulation.

    Construction:

    - **Backtest**: use ``from_close_prices(prices, spread)`` to derive
      bids from daily close prices adjusted by a uniform half-spread.
    - **Live**: construct directly from orderbook best-bid levels.
    """

    model_config = {"extra": "forbid"}

    yes_bid: dict[str, float] = Field(
        ...,
        description="Mapping of outcome name to YES bid price in dollars",
    )
    no_bid: dict[str, float] = Field(
        ...,
        description="Mapping of outcome name to NO bid price in dollars",
    )

    @property
    def yes_ask(self) -> dict[str, float]:
        """Cost to buy YES = 1.0 - no_bid (match a NO bidder)."""
        return {o: min(1.0, 1.0 - p) for o, p in self.no_bid.items()}

    @property
    def no_ask(self) -> dict[str, float]:
        """Cost to buy NO = 1.0 - yes_bid (match a YES bidder)."""
        return {o: min(1.0, 1.0 - p) for o, p in self.yes_bid.items()}

    @classmethod
    def from_close_prices(cls, prices: dict[str, float], spread: float = 0.0) -> "MarketQuotes":
        """Construct from YES close prices and a uniform one-way spread penalty.

        Close prices approximate the midpoint. ``spread`` is the
        half-spread (one-way distance from mid to bid/ask) in dollars.

        Args:
            prices: {outcome: yes_close_price} in dollars [0.0, 1.0].
            spread: One-way spread in dollars (half of full bid-ask width).
        """
        return cls(
            yes_bid={o: max(0.0, p - spread) for o, p in prices.items()},
            no_bid={o: max(0.0, (1.0 - p) - spread) for o, p in prices.items()},
        )


# ============================================================================
# Kelly allocation
# ============================================================================


class KellyAllocation(BaseModel):
    """Position sizing result for a single outcome from Kelly criterion.

    Example (buy 10 YES contracts at $0.25 each)::

        KellyAllocation(
            outcome="Sinners",
            direction=PositionDirection.YES,
            model_prob=0.35,
            execution_price=0.25,
            net_edge=0.09,
            recommended_contracts=10,
        )
        # .outlay_dollars = 10 * 0.25 = 2.50
        # .max_profit_dollars = 10 * 0.75 = 7.50
    """

    model_config = {"extra": "forbid"}

    outcome: str = Field(..., description="Outcome name (e.g. 'Sinners')")
    direction: PositionDirection = Field(
        ..., description="YES or NO — which contract side this allocation is for"
    )
    model_prob: float = Field(
        ..., ge=0, le=1, description="Model's estimated probability for this direction"
    )
    execution_price: float = Field(
        ..., ge=0, le=1.0, description="Expected execution price in dollars (ask for buys)"
    )
    net_edge: float = Field(..., description="Edge after fees: model_prob - implied_prob - fee")
    recommended_contracts: int = Field(
        ..., ge=0, description="Number of contracts Kelly recommends buying"
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def fee(self) -> float:
        """Per-contract fee derived from edge components.

        From the edge definition: net_edge = model_prob - execution_price - fee,
        therefore: fee = model_prob - execution_price - net_edge.
        """
        return round(self.model_prob - self.execution_price - self.net_edge, 6)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def outlay_dollars(self) -> float:
        """Total cost to buy recommended_contracts at execution_price."""
        return round(self.recommended_contracts * self.execution_price, 2)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def max_profit_dollars(self) -> float:
        """Profit if this bet is correct (payout - outlay). Binary pays $1."""
        return round(self.recommended_contracts * (1.0 - self.execution_price), 2)


# ============================================================================
# Trade records
# ============================================================================


class Fill(BaseModel):
    """Record of a single trade execution.

    One Fill per signal that results in a trade (non-zero delta_contracts).
    In backtests, each signal fills completely at the listed price.
    In live trading, a single order may produce multiple partial fills.
    """

    model_config = {"extra": "forbid"}

    timestamp: datetime = Field(..., description="UTC datetime when this trade executed")
    outcome: str = Field(..., description="Outcome name (e.g. 'Sinners')")
    direction: PositionDirection = Field(..., description="YES or NO — which contract was traded")
    action: TradeAction = Field(..., description="BUY or SELL action that produced this fill")
    contracts: int = Field(..., gt=0, description="Number of contracts traded")
    price: float = Field(..., ge=0, le=1.0, description="Execution price per contract in dollars")
    fee_dollars: float = Field(..., ge=0, description="Total fee paid for this trade in USD")
    cash_delta: float = Field(
        ..., description="Net impact on cash in USD (negative=spend, positive=receive)"
    )
    reason: str = Field(..., description="Human-readable reason for this trade")


class SettlementResult(BaseModel):
    """Outcome of settling all positions assuming a specific winner.

    In a prediction market with mutually exclusive outcomes (one winner),
    all YES contracts on the winner pay $1, all others expire at $0.
    """

    model_config = {"extra": "forbid"}

    winner: str = Field(..., description="The outcome assumed to have won")
    initial_bankroll: float = Field(..., gt=0, description="Starting capital for return_pct")
    final_cash: float = Field(..., description="Cash after all positions settle in USD")
    total_pnl: float = Field(..., description="Net profit/loss in USD (final_cash - bankroll)")
    pnl_by_outcome: dict[str, float] = Field(
        default_factory=dict,
        description="P&L breakdown per outcome name in USD",
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def return_pct(self) -> float:
        """Return on initial bankroll as a percentage."""
        if self.initial_bankroll == 0:
            return 0.0
        return round(self.total_pnl / self.initial_bankroll * 100, 1)
