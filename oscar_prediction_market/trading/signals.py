"""Trade signal generation: edge + Kelly -> actionable trade recommendations.

This is the top-level orchestrator of the trading pipeline. Given model
predictions and current market prices, it:

1. Computes edge for each outcome (via ``edge.py``)
2. Sizes positions via Kelly criterion (via ``kelly.py``)
3. Compares target positions to current holdings
4. Outputs BUY/SELL/HOLD signals with exact contract deltas

The main entry point is :func:`generate_signals`, which returns a flat
``list[TradeSignal]``. Callers (backtest engine, live trading) iterate
over the list directly.

**Stateless by design**: all state (current positions, bankroll, prices) is
passed in as arguments. The function is pure -- no side effects, no API calls,
no mutation. This makes it easy to test, backtest, and reason about.

Key design decisions:

- **Sell signals** are generated when edge flips negative below a threshold,
  accounting for round-trip transaction costs (buying + selling fees).
- **Position deltas** (not absolute targets) are output, so the caller knows
  exactly how many contracts to buy or sell.
- Both independent and multi-outcome Kelly can run simultaneously for
  comparison, with multi-outcome preferred for actual signals.

Usage::

    from oscar_prediction_market.trading.signals import generate_signals
    from oscar_prediction_market.trading.schema import (
        KellyConfig, MarketQuotes, TradingConfig, PositionDirection,
    )

    quotes = MarketQuotes.from_close_prices(
        {"One Battle after Another": 0.25, "Sinners": 0.12}, spread=0.02,
    )
    signals = generate_signals(
        model_predictions={"One Battle after Another": 0.30, "Sinners": 0.15},
        execution_prices=quotes,
        current_positions=[],
        config=TradingConfig(
            kelly=KellyConfig(
                kelly_mode="multi_outcome",
                bankroll=1000, kelly_fraction=0.25,
                buy_edge_threshold=0.05, max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            fee_type="taker",
            sell_edge_threshold=-0.03,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
        ),
    )
"""

from typing import NamedTuple

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.trading.edge import Edge
from oscar_prediction_market.trading.kelly import (
    independent_kelly,
    multi_outcome_kelly,
)
from oscar_prediction_market.trading.schema import (
    KellyMode,
    MarketQuotes,
    Position,
    PositionDirection,
    TradeAction,
    TradingConfig,
)


class TradeSignal(BaseModel):
    """Trade recommendation for a single outcome.

    Represents a BUY, SELL, or HOLD recommendation with full context:
    the edge that motivated it, the current and target positions, and
    the execution price. ``delta_contracts`` and ``outlay_dollars`` are
    computed from stored fields.
    """

    model_config = {"extra": "forbid"}

    outcome: str = Field(..., description="Outcome name (e.g. 'Sinners')")
    ticker: str = Field(default="", description="Kalshi market ticker for this outcome")
    direction: PositionDirection = Field(
        ..., description="YES or NO -- which contract this signal is for"
    )
    action: TradeAction = Field(..., description="BUY, SELL, or HOLD recommendation")
    model_prob: float = Field(
        ..., ge=0, le=1, description="Model's probability estimate for this direction"
    )
    execution_price: float = Field(
        ..., ge=0, le=1.0, description="Price per contract in dollars for this trade"
    )
    net_edge: float = Field(
        ..., description="Net edge after fees (model_prob - implied_prob - fee)"
    )
    current_contracts: int = Field(
        ..., ge=0, description="Number of contracts currently held before this signal"
    )
    target_contracts: int = Field(
        ..., ge=0, description="Kelly-optimal number of contracts to hold after this signal"
    )
    reason: str = Field(..., description="Human-readable explanation for this signal")

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def delta_contracts(self) -> int:
        """Change in contracts: target - current (positive=buy, negative=sell)."""
        return self.target_contracts - self.current_contracts

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def outlay_dollars(self) -> float:
        """Cost of this delta in USD (positive=spend, negative=receive)."""
        return round(self.delta_contracts * self.execution_price, 2)


# ============================================================================
# Helpers
# ============================================================================


def _get_sell_price(
    outcome: str,
    direction: PositionDirection,
    execution_prices: MarketQuotes,
) -> float:
    """Get the execution price for selling/closing a position.

    Selling means hitting the bid on the respective side:

    - Selling YES -> hit YES bids -> ``execution_prices.yes_bid[outcome]``
    - Selling NO  -> hit NO bids  -> ``execution_prices.no_bid[outcome]``
    """
    if direction == PositionDirection.YES:
        return execution_prices.yes_bid.get(outcome, 0.0)
    else:
        return execution_prices.no_bid.get(outcome, 0.0)


def _compute_edges(
    model_predictions: dict[str, float],
    execution_prices: MarketQuotes,
    config: TradingConfig,
) -> list[Edge]:
    """Compute buy-side edges for outcomes that pass the price filter.

    When ``config.allowed_directions`` includes NO, also computes BUY NO edges
    for each outcome. The NO ask price is derived from the YES bid:
    ``no_ask = 100 - yes_bid`` (on Kalshi's dual-sided orderbook, buying NO
    means matching a YES bidder).

    When ``config.limit_price_offset > 0`` (maker/limit-order mode), the
    execution price is computed as ``bid + offset`` instead of the ask.
    This gives a better fill price at the cost of uncertain fill timing.

    Outcomes with an execution price below ``config.min_price`` are
    excluded: the Kalshi fee schedule charges a fixed minimum per contract,
    making low-priced contracts prohibitively expensive as a fraction of
    notional (e.g. a 1c fee on a 3c contract is 33% overhead).

    Args:
        model_predictions: {outcome: model_probability} (YES-side, i.e.
            probability the nominee wins).
        execution_prices: Spread-adjusted bid prices for all outcomes.
        config: Trading config providing fee_type, limit_price_offset,
            min_price, allowed_directions.

    Returns:
        List of Edge. When allowed_directions includes NO, may contain both
        YES and NO edges for the same outcome (at most one will have positive
        edge after fees).
    """
    edges: list[Edge] = []
    for outcome, model_prob in model_predictions.items():
        # --- YES edge ---
        if PositionDirection.YES in config.allowed_directions:
            if config.limit_price_offset > 0:
                # Limit order: price relative to bid
                yes_bid = execution_prices.yes_bid.get(outcome)
                buy_price = yes_bid + config.limit_price_offset if yes_bid is not None else None
            else:
                # Market order: price at ask (current behavior)
                buy_price = execution_prices.yes_ask.get(outcome)
            if buy_price is not None and buy_price >= config.min_price:
                edges.append(
                    Edge(
                        outcome=outcome,
                        direction=PositionDirection.YES,
                        model_prob=model_prob,
                        execution_price=max(0, min(1.0, buy_price)),
                        fee_type=config.fee_type,
                    )
                )

        # --- NO edge ---
        if PositionDirection.NO in config.allowed_directions:
            if config.limit_price_offset > 0:
                # Limit order: price relative to bid
                no_bid = execution_prices.no_bid.get(outcome)
                no_buy_price = no_bid + config.limit_price_offset if no_bid is not None else None
            else:
                # Market order: price at ask
                no_buy_price = execution_prices.no_ask.get(outcome)
            if no_buy_price is not None and no_buy_price >= config.min_price:
                no_model_prob = 1.0 - model_prob
                edges.append(
                    Edge(
                        outcome=outcome,
                        direction=PositionDirection.NO,
                        model_prob=no_model_prob,
                        execution_price=max(0, min(1.0, no_buy_price)),
                        fee_type=config.fee_type,
                    )
                )

    return edges


def _decide_action(
    edge: Edge,
    pos: Position,
    target_contracts: int,
    config: TradingConfig,
) -> tuple[TradeAction, int, str]:
    """Map edge + position to (action, final_target, reason).

    Applies a three-zone decision tree based on net edge::

        SELL zone         HOLD zone          BUY zone
          <--|------------|----------------------|-->
         sell_edge       0.00            buy_edge
         threshold                       threshold

    The gap between thresholds provides hysteresis: a position is only
    opened once edge clears ``buy_edge_threshold``, and only closed when
    edge drops below ``sell_edge_threshold`` (a negative number). The zone
    between prevents unnecessary churn from small edge fluctuations.

    Args:
        edge: Computed Edge for the outcome.
        pos: Current held position.
        target_contracts: Kelly-recommended contract count for this outcome.
        config: Provides buy_edge_threshold and sell_edge_threshold.

    Returns:
        Tuple of (action, final_target_contracts, reason).
    """
    current = pos.contracts
    delta = target_contracts - current

    if delta > 0 and edge.net_edge >= config.kelly.buy_edge_threshold:
        return TradeAction.BUY, target_contracts, f"positive edge ({edge.net_edge:.1%})"

    if current > 0 and edge.net_edge < config.sell_edge_threshold:
        return (
            TradeAction.SELL,
            0,
            f"edge flipped negative ({edge.net_edge:.1%})",
        )

    if current > 0 and delta < 0:
        return (
            TradeAction.SELL,
            target_contracts,
            f"reducing position (edge={edge.net_edge:.1%})",
        )

    reason = "hold" if current > 0 else "no edge"
    return TradeAction.HOLD, current, reason


# ============================================================================
# Signal Generation
# ============================================================================


class _SignalContext(NamedTuple):
    """Shared per-call state threaded through signal processing helpers.

    Bundles the computed maps and config that are constant across all outcomes
    in a single ``generate_signals()`` call, avoiding long argument lists in
    each helper.
    """

    target_by_key: dict[tuple[str, PositionDirection], int]
    edge_map: dict[tuple[str, PositionDirection], Edge]
    execution_prices: MarketQuotes
    config: TradingConfig
    ticker_map: dict[str, str]


def _empty_pos(outcome: str, direction: PositionDirection = PositionDirection.YES) -> Position:
    """Sentinel for 'no position held'."""
    return Position(outcome=outcome, direction=direction, contracts=0, avg_cost=0)


def _process_orphan(
    outcome: str,
    pos: Position,
    ctx: _SignalContext,
) -> TradeSignal | None:
    """Handle an outcome that has a held position but no model prediction.

    An orphan occurs when the model drops an outcome (e.g. a nominee was
    withdrawn after the initial bet was placed). Liquidate immediately at
    the best available bid.
    """
    if pos.contracts <= 0:
        return None
    ticker = ctx.ticker_map.get(outcome, "")
    sell_price = _get_sell_price(outcome, pos.direction, ctx.execution_prices)
    return TradeSignal(
        outcome=outcome,
        ticker=ticker,
        direction=pos.direction,
        action=TradeAction.SELL,
        model_prob=0,
        execution_price=sell_price,
        net_edge=0,
        current_contracts=pos.contracts,
        target_contracts=0,
        reason="no model prediction available",
    )


def _process_existing(
    outcome: str,
    pos: Position,
    ctx: _SignalContext,
) -> list[TradeSignal]:
    """Handle an outcome where we currently hold a position.

    Two sub-cases:

    1. **Direction flip**: Kelly targets the opposite direction (e.g. we hold
       YES but the model now strongly favors NO). Emit SELL of the current
       position followed by BUY in the new direction -- two separate trades
       with two sets of fees, modeling the realistic cost of changing conviction.

    2. **Same direction**: Normal BUY/SELL/HOLD decision via ``_decide_action``.
    """
    ticker = ctx.ticker_map.get(outcome, "")
    opposite_dir = pos.direction.opposite
    target_same = ctx.target_by_key.get((outcome, pos.direction), 0)
    target_opp = ctx.target_by_key.get((outcome, opposite_dir), 0)

    if target_opp > 0:
        # --- Direction flip: close current, open opposite ---
        sell_price = _get_sell_price(outcome, pos.direction, ctx.execution_prices)
        result: list[TradeSignal] = [
            TradeSignal(
                outcome=outcome,
                ticker=ticker,
                direction=pos.direction,
                action=TradeAction.SELL,
                model_prob=0,
                execution_price=sell_price,
                net_edge=0,
                current_contracts=pos.contracts,
                target_contracts=0,
                reason=f"direction flip to {opposite_dir.value}",
            )
        ]
        opp_edge = ctx.edge_map.get((outcome, opposite_dir))
        if opp_edge is not None:
            result.append(
                TradeSignal(
                    outcome=outcome,
                    ticker=ticker,
                    direction=opposite_dir,
                    action=TradeAction.BUY,
                    model_prob=opp_edge.model_prob,
                    execution_price=opp_edge.execution_price,
                    net_edge=opp_edge.net_edge,
                    current_contracts=0,
                    target_contracts=target_opp,
                    reason=f"positive {opposite_dir.value} edge ({opp_edge.net_edge:.1%})",
                )
            )
        return result

    # --- Same direction: normal BUY/SELL/HOLD ---
    same_edge = ctx.edge_map.get((outcome, pos.direction))
    if same_edge is None:
        sell_price = _get_sell_price(outcome, pos.direction, ctx.execution_prices)
        return [
            TradeSignal(
                outcome=outcome,
                ticker=ticker,
                direction=pos.direction,
                action=TradeAction.SELL,
                model_prob=0,
                execution_price=sell_price,
                net_edge=0,
                current_contracts=pos.contracts,
                target_contracts=0,
                reason="no edge data for position direction",
            )
        ]

    action, final_target, reason = _decide_action(same_edge, pos, target_same, ctx.config)
    exec_price = (
        _get_sell_price(outcome, pos.direction, ctx.execution_prices)
        if action == TradeAction.SELL
        else same_edge.execution_price
    )
    return [
        TradeSignal(
            outcome=outcome,
            ticker=ticker,
            direction=pos.direction,
            action=action,
            model_prob=same_edge.model_prob,
            execution_price=exec_price,
            net_edge=same_edge.net_edge,
            current_contracts=pos.contracts,
            target_contracts=final_target,
            reason=reason,
        )
    ]


def _process_fresh(
    outcome: str,
    ctx: _SignalContext,
) -> TradeSignal | None:
    """Handle an outcome where we hold no position.

    Selects the best direction Kelly targets (at most one will have target > 0
    due to position netting in the optimizer). Falls through to a HOLD signal
    if neither direction clears the buy-edge threshold.
    """
    ticker = ctx.ticker_map.get(outcome, "")
    target_yes = ctx.target_by_key.get((outcome, PositionDirection.YES), 0)
    target_no = ctx.target_by_key.get((outcome, PositionDirection.NO), 0)

    for direction, target in [
        (PositionDirection.YES, target_yes),
        (PositionDirection.NO, target_no),
    ]:
        if target > 0:
            edge = ctx.edge_map.get((outcome, direction))
            if edge is not None:
                return TradeSignal(
                    outcome=outcome,
                    ticker=ticker,
                    direction=direction,
                    action=TradeAction.BUY,
                    model_prob=edge.model_prob,
                    execution_price=edge.execution_price,
                    net_edge=edge.net_edge,
                    current_contracts=0,
                    target_contracts=target,
                    reason=f"positive {direction.value} edge ({edge.net_edge:.1%})",
                )

    # No edge on either side -- emit a HOLD for the YES side (if it was computed)
    yes_edge = ctx.edge_map.get((outcome, PositionDirection.YES))
    if yes_edge is not None:
        return TradeSignal(
            outcome=outcome,
            ticker=ticker,
            direction=PositionDirection.YES,
            action=TradeAction.HOLD,
            model_prob=yes_edge.model_prob,
            execution_price=yes_edge.execution_price,
            net_edge=yes_edge.net_edge,
            current_contracts=0,
            target_contracts=0,
            reason="no edge",
        )
    return None


def generate_signals(
    model_predictions: dict[str, float],
    execution_prices: MarketQuotes,
    current_positions: list[Position],
    config: TradingConfig,
    ticker_map: dict[str, str] | None = None,
) -> list[TradeSignal]:
    """Generate trade signals from model predictions and market prices.

    This is the main entry point for the signal pipeline. The core logic:

    1. For each outcome, compute edge = model_prob - implied_prob - fees.
       Which directions are computed depends on ``config.allowed_directions``.
    2. Run Kelly sizing to determine optimal contract counts (jointly
       across YES and NO when both are enabled).
    3. Compare optimal ("target") positions to what we already hold.
    4. Output the delta: how many contracts to buy or sell.

    **Direction flips**: If we hold YES but Kelly wants NO (or vice versa),
    the pipeline emits a SELL signal for the entire current position followed
    by a BUY signal for the new direction. This models two separate trades
    with two sets of fees -- the realistic cost of changing conviction.

    **Sell logic**: If we hold contracts and the edge drops below
    ``sell_edge_threshold`` (typically a small negative number like -3%),
    we generate a SELL signal for the entire position. The threshold is
    negative (not zero) because selling incurs fees too -- we only exit if
    the edge is bad enough to justify paying round-trip costs.

    **Hold logic**: If we hold contracts and the edge is still positive but
    below ``buy_edge_threshold``, we hold (don't buy more, don't sell). This
    avoids unnecessary churn.

    Args:
        model_predictions: {outcome: model_probability} mapping (YES-side).
        execution_prices: Spread-adjusted buy/sell prices. Use
            ``MarketQuotes.from_close_prices(prices, spread)`` for backtesting
            or construct from live orderbook VWAP for live trading.
        current_positions: Currently held positions. Each Position includes
            ``outcome`` and ``direction`` fields.
        config: Trading strategy configuration (see ``TradingConfig``).
        ticker_map: {outcome: kalshi_ticker} for labeling (optional).

    Returns:
        List of TradeSignal with all trade recommendations, sorted by
        action priority (SELL first, then BUY, then HOLD).
    """
    if ticker_map is None:
        ticker_map = {}

    # Build position lookup by outcome
    if len({p.outcome for p in current_positions}) != len(current_positions):
        raise ValueError("Duplicate outcome in current_positions")
    pos_by_outcome: dict[str, Position] = {p.outcome: p for p in current_positions}

    # 1. Compute buy-side edges (YES and optionally NO) for eligible outcomes
    edges = _compute_edges(model_predictions, execution_prices, config)

    # 2. Run Kelly sizing
    if config.kelly.kelly_mode == KellyMode.INDEPENDENT:
        allocations = independent_kelly(edges, config.kelly)
    elif config.kelly.kelly_mode == KellyMode.MULTI_OUTCOME:
        allocations = multi_outcome_kelly(edges, config.kelly)
    else:
        raise ValueError(f"Unknown kelly_mode: {config.kelly.kelly_mode}")

    # 3. Build lookup maps from Kelly results
    target_by_key: dict[tuple[str, PositionDirection], int] = {
        (alloc.outcome, alloc.direction): alloc.recommended_contracts for alloc in allocations
    }
    edge_map: dict[tuple[str, PositionDirection], Edge] = {
        (e.outcome, e.direction): e for e in edges
    }
    ctx = _SignalContext(
        target_by_key=target_by_key,
        edge_map=edge_map,
        execution_prices=execution_prices,
        config=config,
        ticker_map=ticker_map,
    )

    # 4. Generate signals with deltas
    signals: list[TradeSignal] = []
    all_outcomes = set(model_predictions.keys()) | set(pos_by_outcome.keys())

    for outcome in sorted(all_outcomes):
        pos = pos_by_outcome.get(outcome, _empty_pos(outcome))

        if outcome not in model_predictions:
            signal = _process_orphan(outcome, pos, ctx)
            if signal is not None:
                signals.append(signal)
            continue

        if pos.contracts > 0:
            signals.extend(_process_existing(outcome, pos, ctx))
        else:
            signal = _process_fresh(outcome, ctx)
            if signal is not None:
                signals.append(signal)

    # Sort by action priority: SELL first (to free cash + clear positions for
    # direction flips), then BUY, then HOLD.
    action_order = {TradeAction.SELL: 0, TradeAction.BUY: 1, TradeAction.HOLD: 2}
    signals.sort(key=lambda s: (action_order.get(s.action, 3), -abs(s.net_edge)))

    return signals
