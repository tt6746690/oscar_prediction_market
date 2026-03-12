"""Fetch live Kalshi portfolio and compute position adjustments.

Connects to the Kalshi API to retrieve:
1. Account balance (wallet cash + portfolio MTM value)
2. Current open positions across all Oscar markets

Then computes the delta between current holdings and model-recommended
target positions, producing actionable trade instructions.

Used by ``generate_report.py`` when ``--live-portfolio`` is passed.

The target portfolio is derived from the backtest position_summary CSV,
scaled to the actual Kalshi account equity and weighted by the maxedge_100
allocation strategy.

Example::

    portfolio = fetch_live_portfolio(ceremony_year=2026)
    targets = compute_target_positions(...)
    adjustments = compute_adjustments(portfolio, targets, market_prices)
"""

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.trading.kalshi_client import (
    KalshiClient,
    estimate_fee,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.schema import FeeType, PositionDirection

logger = logging.getLogger(__name__)

#: Categories covered by the model (same 9 as in the report).
MODELED_CATEGORY_SLUGS: set[str] = {
    "best_picture",
    "directing",
    "actor_leading",
    "actress_leading",
    "actor_supporting",
    "actress_supporting",
    "original_screenplay",
    "animated_feature",
    "cinematography",
}


# ============================================================================
# Models
# ============================================================================


class LivePosition(BaseModel):
    """A single open position from Kalshi.

    Represents a real holding on the exchange with cost basis computed
    from ``market_exposure / contracts``.

    Example::

        LivePosition(
            category_slug="actress_supporting",
            nominee="Amy Madigan",
            ticker="KXOSCARSUPACTR-26-AMY",
            direction="yes",
            contracts=202,
            avg_cost=0.43,
            fees_paid=3.47,
            realized_pnl=0.0,
        )
    """

    model_config = {"extra": "forbid"}

    category_slug: str = Field(..., description="Category slug (e.g. 'actor_leading')")
    nominee: str = Field(..., description="Nominee name")
    ticker: str = Field(..., description="Kalshi market ticker")
    direction: PositionDirection = Field(...)
    contracts: int = Field(..., ge=0)
    avg_cost: float = Field(..., ge=0, le=1.0, description="Average cost per contract in dollars")
    fees_paid: float = Field(..., ge=0, description="Total fees paid on this position in dollars")
    realized_pnl: float = Field(..., description="Realized P&L in dollars")

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def cost_basis(self) -> float:
        """Total cost basis in dollars."""
        return round(self.contracts * self.avg_cost, 2)


class LivePortfolio(BaseModel):
    """Complete live portfolio state from Kalshi.

    Combines account-level data (cash, MTM) with per-position details.
    ``total_equity`` is the bankroll used for target position sizing.
    """

    model_config = {"extra": "forbid"}

    wallet_balance: float = Field(..., description="Available cash in dollars")
    portfolio_value: float = Field(..., description="Current MTM value of positions in dollars")
    positions: list[LivePosition] = Field(default_factory=list)
    total_fees_paid: float = Field(default=0.0, description="Lifetime fees paid in dollars")
    total_realized_pnl: float = Field(default=0.0, description="Lifetime realized P&L in dollars")

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def total_equity(self) -> float:
        """Total account value: cash + position value."""
        return round(self.wallet_balance + self.portfolio_value, 2)


class TargetPosition(BaseModel):
    """A single position recommended by the model, sized to the actual bankroll."""

    model_config = {"extra": "forbid"}

    category_slug: str
    nominee: str
    ticker: str
    direction: PositionDirection
    contracts: int = Field(..., ge=0)
    execution_price: float = Field(
        ..., ge=0, le=1.0, description="Estimated execution price per contract"
    )


class PositionAdjustment(BaseModel):
    """A single trade action to adjust from current to target portfolio.

    Each instance is directly actionable: go to ``kalshi_url``, place the
    order described by ``action`` for ``delta_contracts`` contracts.
    """

    model_config = {"extra": "forbid"}

    category_slug: str
    nominee: str
    ticker: str
    kalshi_url: str
    action: str = Field(..., description="e.g. BUY YES, SELL NO, HOLD")
    current_desc: str = Field(..., description="Human-readable current (e.g. '154 NO')")
    target_desc: str = Field(..., description="Human-readable target (e.g. '92 YES')")
    delta_contracts: int = Field(..., ge=0, description="Number of contracts to trade")
    est_price: float = Field(..., ge=0, le=1.0, description="Estimated execution price")
    est_cost: float = Field(..., ge=0, description="Estimated cost/proceeds in dollars")
    est_fee: float = Field(..., ge=0, description="Estimated taker fee in dollars")


# ============================================================================
# Helpers
# ============================================================================


def _build_ticker_map(ceremony_year: int) -> dict[str, tuple[str, str]]:
    """Build ticker → (category_slug, nominee) for all markets in a ceremony year.

    Iterates over all OscarCategory values and looks up the registry.
    Returns a dict like ``{"KXOSCARACTO-26-TIM": ("actor_leading", "Timothée Chalamet")}``.
    """
    result: dict[str, tuple[str, str]] = {}
    for cat in OscarCategory:
        try:
            data = OSCAR_MARKETS.get_category_data(cat, ceremony_year)
        except KeyError:
            continue
        for nominee, ticker in data.nominee_tickers.items():
            result[ticker] = (cat.slug, nominee)
    return result


def _category_series_url(category_slug: str) -> str:
    """Get Kalshi series-level URL for a category.

    e.g. "actor_leading" → "https://kalshi.com/markets/kxoscaracto"
    """
    try:
        cat = OscarCategory.from_slug(category_slug)
        series = OSCAR_MARKETS.get_series_ticker(cat)
        return f"https://kalshi.com/markets/{series.lower()}"
    except (KeyError, ValueError):
        return ""


def _nominee_ticker(category_slug: str, nominee: str, ceremony_year: int) -> str:
    """Look up Kalshi market ticker for a nominee (case-insensitive)."""
    try:
        cat = OscarCategory.from_slug(category_slug)
        data = OSCAR_MARKETS.get_category_data(cat, ceremony_year)
        # Case-insensitive lookup
        nominee_lower = nominee.lower()
        for reg_name, ticker in data.nominee_tickers.items():
            if reg_name.lower() == nominee_lower:
                return ticker
        return ""
    except (KeyError, ValueError):
        return ""


# ============================================================================
# Portfolio fetching
# ============================================================================


def fetch_live_portfolio(ceremony_year: int) -> LivePortfolio:
    """Fetch current portfolio state from Kalshi.

    Returns balance, portfolio value, and all open positions across
    Oscar markets for the given ceremony year.

    Positions are mapped to categories and nominees using the market
    registry.  Unknown tickers (non-Oscar or different year) are
    logged and skipped.
    """
    client = KalshiClient()

    # Account balance
    balance_data = client.get_balance()
    wallet_balance = balance_data["balance"] / 100.0  # cents → dollars
    portfolio_value = balance_data["portfolio_value"] / 100.0

    # Positions
    positions_data = client.get_positions()
    market_positions: list[dict[str, Any]] = positions_data.get("market_positions", [])
    event_positions: list[dict[str, Any]] = positions_data.get("event_positions", [])

    # Sum fees and realized P&L across all events
    total_fees = sum(float(ep.get("fees_paid_dollars", "0")) for ep in event_positions)
    total_realized = sum(float(ep.get("realized_pnl_dollars", "0")) for ep in event_positions)

    # Build ticker → (category, nominee) mapping
    ticker_map = _build_ticker_map(ceremony_year)

    positions: list[LivePosition] = []
    for mp in market_positions:
        ticker = mp["ticker"]
        pos_count = mp["position"]  # signed: positive=YES, negative=NO

        if pos_count == 0:
            continue  # Skip closed positions

        # Map ticker to category and nominee
        if ticker not in ticker_map:
            logger.warning("Unknown ticker %s (position=%d), skipping", ticker, pos_count)
            continue

        cat_slug, nominee = ticker_map[ticker]
        direction = PositionDirection.YES if pos_count > 0 else PositionDirection.NO
        contracts = abs(pos_count)

        # avg_cost = market_exposure (cents) / contracts → dollars
        market_exposure_cents = mp.get("market_exposure", 0)
        avg_cost = (market_exposure_cents / contracts / 100.0) if contracts > 0 else 0.0

        fees_paid = float(mp.get("fees_paid_dollars", "0"))
        realized_pnl = float(mp.get("realized_pnl_dollars", "0"))

        positions.append(
            LivePosition(
                category_slug=cat_slug,
                nominee=nominee,
                ticker=ticker,
                direction=direction,
                contracts=contracts,
                avg_cost=round(avg_cost, 4),
                fees_paid=fees_paid,
                realized_pnl=realized_pnl,
            )
        )

    return LivePortfolio(
        wallet_balance=wallet_balance,
        portfolio_value=portfolio_value,
        positions=positions,
        total_fees_paid=round(total_fees, 2),
        total_realized_pnl=round(total_realized, 2),
    )


# ============================================================================
# Target position computation
# ============================================================================


def compute_target_positions(
    position_summary: pd.DataFrame,
    config_label: str,
    model_type: str,
    entry_snapshot: str,
    bankroll: float,
    backtest_bankroll_per_category: float,
    n_categories: int,
    allocation_weights: dict[str, float],
    execution_prices: dict[tuple[str, str], float],
    ceremony_year: int,
) -> list[TargetPosition]:
    """Compute target positions sized to the actual bankroll.

    Takes the backtest position_summary (at $1000/category) and scales
    to the actual bankroll, applying maxedge_100 allocation weights.

    Target contracts = raw_contracts × (bankroll / total_backtest_bankroll) × cat_weight

    Args:
        position_summary: Position summary DataFrame from the backtest.
        config_label: Config label to filter (e.g. "fee=taker_kf=0.05_bet=0.2...").
        model_type: Model type to filter (e.g. "avg_ensemble").
        entry_snapshot: Entry snapshot to filter (e.g. "live_2026-03-08T05:02Z").
        bankroll: Actual bankroll (total equity) in dollars.
        backtest_bankroll_per_category: Bankroll per category in the backtest ($1000).
        n_categories: Number of categories (9).
        allocation_weights: {category_slug: weight} from maxedge_100.
        execution_prices: {(category, nominee): yes_execution_price_dollars}.
            Pre-computed by caller (ask price for taker, bid+offset for maker).
        ceremony_year: For ticker lookup.

    Returns:
        List of TargetPosition objects with contract counts sized to real bankroll.
    """
    scale_factor = bankroll / (backtest_bankroll_per_category * n_categories)

    pos_slice = position_summary[
        (position_summary["config_label"] == config_label)
        & (position_summary["model_type"] == model_type)
        & (position_summary["entry_snapshot"] == entry_snapshot)
    ]

    targets: list[TargetPosition] = []
    for _, row in pos_slice.iterrows():
        cat = row["category"]
        nominee = row["outcome"]
        direction = PositionDirection(row["direction"])
        raw_contracts = row["contracts"]
        cat_weight = allocation_weights.get(cat, 1.0)

        # Scale: raw × global_scale × per-category weight
        target_contracts = int(round(raw_contracts * scale_factor * cat_weight))
        if target_contracts <= 0:
            continue

        # Execution price from caller-provided prices
        yes_price = execution_prices.get((cat, nominee), 0.5)
        exec_price = yes_price if direction == PositionDirection.YES else (1.0 - yes_price)

        # Look up ticker
        ticker = _nominee_ticker(cat, nominee, ceremony_year)

        targets.append(
            TargetPosition(
                category_slug=cat,
                nominee=nominee,
                ticker=ticker,
                direction=direction,
                contracts=target_contracts,
                execution_price=round(max(0.01, min(0.99, exec_price)), 4),
            )
        )

    return targets


# ============================================================================
# Adjustment computation
# ============================================================================


def _normalize_key(category: str, nominee: str) -> tuple[str, str]:
    """Normalize (category, nominee) key for case-insensitive matching."""
    return (category, nominee.lower())


def compute_adjustments(
    current_portfolio: LivePortfolio,
    target_positions: list[TargetPosition],
    market_prices: dict[tuple[str, str], float],
    buy_fee_type: FeeType,
) -> list[PositionAdjustment]:
    """Compute trade actions to move from current holdings to target portfolio.

    Handles five cases per (category, nominee) pair:

    1. **New position**: target exists, no current → BUY
    2. **Increase**: same direction, target > current → BUY more
    3. **Decrease**: same direction, target < current → SELL some
    4. **Hold**: same direction, same quantity → no action
    5. **Direction reversal**: opposite directions → SELL current + BUY target
    6. **Close**: current exists, no target → SELL all

    Each resulting ``PositionAdjustment`` is a single tradeable action.
    Direction reversals produce two rows (sell then buy).

    Nominee matching is case-insensitive to handle mismatches between
    the backtest data and the Kalshi ticker registry.

    Args:
        current_portfolio: Current Kalshi holdings.
        target_positions: Model-recommended positions sized to actual bankroll.
        market_prices: {(category, nominee): yes_mid_price} for cost estimation.
        buy_fee_type: Fee schedule to use for BUY trades (TAKER or MAKER).

    Returns:
        List of PositionAdjustment objects, sorted by category then nominee.
    """
    # Build lookup maps keyed by normalized (category_slug, nominee_lower)
    current_map: dict[tuple[str, str], LivePosition] = {
        _normalize_key(p.category_slug, p.nominee): p for p in current_portfolio.positions
    }
    target_map: dict[tuple[str, str], TargetPosition] = {
        _normalize_key(t.category_slug, t.nominee): t for t in target_positions
    }

    # Market prices also need normalized keys for lookup
    price_map: dict[tuple[str, str], float] = {
        _normalize_key(k[0], k[1]): v for k, v in market_prices.items()
    }

    adjustments: list[PositionAdjustment] = []
    processed_keys: set[tuple[str, str]] = set()

    # --- Process all target positions ---
    for key in sorted(target_map.keys()):
        target = target_map[key]
        processed_keys.add(key)
        cat, nominee = key
        current = current_map.get(key)
        url = _category_series_url(cat)
        ticker = target.ticker

        if current is None:
            # Case 1: New position
            _append_buy(
                adjustments,
                cat,
                target.nominee,
                ticker,
                url,
                target,
                current_desc="—",
                buy_fee_type=buy_fee_type,
            )

        elif current.direction == target.direction:
            # Cases 2, 3, 4: Same direction
            delta = target.contracts - current.contracts
            dir_str = _dir_str(target.direction)
            # Use the display name from current (it came from the Kalshi registry)
            display_nominee = current.nominee
            cur_desc = f"{current.contracts} {dir_str}"
            tgt_desc = f"{target.contracts} {dir_str}"

            if delta > 0:
                # Buy more
                price = target.execution_price
                cost = delta * price
                fee = estimate_fee(price, buy_fee_type, delta)
                adjustments.append(
                    PositionAdjustment(
                        category_slug=cat,
                        nominee=display_nominee,
                        ticker=ticker,
                        kalshi_url=url,
                        action=f"BUY {dir_str}",
                        current_desc=cur_desc,
                        target_desc=tgt_desc,
                        delta_contracts=delta,
                        est_price=round(price, 4),
                        est_cost=round(cost, 2),
                        est_fee=round(fee, 2),
                    )
                )
            elif delta < 0:
                # Sell some
                sell_count = abs(delta)
                price = target.execution_price
                cost = sell_count * price
                fee = estimate_fee(price, FeeType.TAKER, sell_count)
                adjustments.append(
                    PositionAdjustment(
                        category_slug=cat,
                        nominee=display_nominee,
                        ticker=ticker,
                        kalshi_url=url,
                        action=f"SELL {dir_str}",
                        current_desc=cur_desc,
                        target_desc=tgt_desc,
                        delta_contracts=sell_count,
                        est_price=round(price, 4),
                        est_cost=round(cost, 2),
                        est_fee=round(fee, 2),
                    )
                )
            else:
                # Hold
                adjustments.append(
                    PositionAdjustment(
                        category_slug=cat,
                        nominee=display_nominee,
                        ticker=ticker,
                        kalshi_url=url,
                        action="HOLD",
                        current_desc=cur_desc,
                        target_desc=tgt_desc,
                        delta_contracts=0,
                        est_price=0.0,
                        est_cost=0.0,
                        est_fee=0.0,
                    )
                )

        else:
            # Case 5: Direction reversal — unwind + reverse
            # Step 1: Sell current
            display_nominee = current.nominee
            cur_dir = _dir_str(current.direction)
            tgt_dir = _dir_str(target.direction)
            yes_price = price_map.get(key, 0.5)
            cur_price = (
                yes_price if current.direction == PositionDirection.YES else (1.0 - yes_price)
            )
            cur_price = max(0.01, min(0.99, cur_price))
            cur_cost = current.contracts * cur_price
            cur_fee = estimate_fee(cur_price, FeeType.TAKER, current.contracts)

            adjustments.append(
                PositionAdjustment(
                    category_slug=cat,
                    nominee=display_nominee,
                    ticker=ticker,
                    kalshi_url=url,
                    action=f"SELL {cur_dir}",
                    current_desc=f"{current.contracts} {cur_dir}",
                    target_desc=f"{target.contracts} {tgt_dir}",
                    delta_contracts=current.contracts,
                    est_price=round(cur_price, 4),
                    est_cost=round(cur_cost, 2),
                    est_fee=round(cur_fee, 2),
                )
            )

            # Step 2: Buy target
            _append_buy(
                adjustments,
                cat,
                nominee,
                ticker,
                url,
                target,
                current_desc="(after unwind)",
                buy_fee_type=buy_fee_type,
            )

    # --- Close positions that have no target ---
    for key in sorted(current_map.keys()):
        if key in processed_keys:
            continue

        current = current_map[key]
        cat, _ = key  # nominee is normalized (lowercase) — use current.nominee for display

        # Only process modeled categories
        if cat not in MODELED_CATEGORY_SLUGS:
            continue

        ticker = current.ticker
        url = _category_series_url(cat)
        dir_str = _dir_str(current.direction)

        yes_price = price_map.get(key, 0.5)
        price = yes_price if current.direction == PositionDirection.YES else (1.0 - yes_price)
        price = max(0.01, min(0.99, price))
        cost = current.contracts * price
        fee = estimate_fee(price, FeeType.TAKER, current.contracts)

        adjustments.append(
            PositionAdjustment(
                category_slug=cat,
                nominee=current.nominee,
                ticker=ticker,
                kalshi_url=url,
                action=f"SELL {dir_str}",
                current_desc=f"{current.contracts} {dir_str}",
                target_desc="—",
                delta_contracts=current.contracts,
                est_price=round(price, 4),
                est_cost=round(cost, 2),
                est_fee=round(fee, 2),
            )
        )

    return adjustments


# ============================================================================
# Internal helpers
# ============================================================================


def _dir_str(direction: PositionDirection) -> str:
    """Convert PositionDirection to uppercase string for display."""
    return "YES" if direction == PositionDirection.YES else "NO"


def _append_buy(
    adjustments: list[PositionAdjustment],
    cat: str,
    nominee: str,
    ticker: str,
    url: str,
    target: TargetPosition,
    current_desc: str,
    buy_fee_type: FeeType,
) -> None:
    """Append a BUY adjustment for a target position."""
    dir_str = _dir_str(target.direction)
    price = target.execution_price
    cost = target.contracts * price
    fee = estimate_fee(price, buy_fee_type, target.contracts)

    adjustments.append(
        PositionAdjustment(
            category_slug=cat,
            nominee=nominee,
            ticker=ticker,
            kalshi_url=url,
            action=f"BUY {dir_str}",
            current_desc=current_desc,
            target_desc=f"{target.contracts} {dir_str}",
            delta_contracts=target.contracts,
            est_price=round(price, 4),
            est_cost=round(cost, 2),
            est_fee=round(fee, 2),
        )
    )
