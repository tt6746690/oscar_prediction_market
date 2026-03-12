"""Shared fixtures for trading tests."""

import pytest

from oscar_prediction_market.trading.schema import (
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
    TradingConfig,
)


@pytest.fixture()
def default_trading_config() -> TradingConfig:
    """Standard TradingConfig for tests that don't care about specific values.

    Uses taker fees, multi-outcome Kelly, YES-side trading, and moderate
    edge thresholds (5% buy, -3% sell).  Bankroll $1000.
    """
    return TradingConfig(
        kelly=KellyConfig(
            bankroll=1000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.MULTI_OUTCOME,
            buy_edge_threshold=0.05,
            max_position_per_outcome=250,
            max_total_exposure=500,
        ),
        sell_edge_threshold=-0.03,
        fee_type=FeeType.TAKER,
        limit_price_offset=0.0,
        min_price=0,
        allowed_directions=frozenset({PositionDirection.YES}),
    )
