"""Recommended buy-hold configs for live 2026 analysis.

Two configs (edge_20_taker, edge_20_maker) based on the avg_ensemble model,
which showed the best risk-adjusted performance in the config selection sweep
(d20260305_config_selection_sweep). Both use a 0.20 buy_edge_threshold and 5%
Kelly fraction with multi-outcome Kelly sizing and fixed bankroll. They differ
only in fee model: taker (market orders, offset=0.0) vs maker (limit orders,
offset=0.01).
"""

from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestSimulationConfig,
)
from oscar_prediction_market.trading.schema import (
    NEVER_SELL_THRESHOLD,
    BankrollMode,
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
    TradingConfig,
)

DEFAULT_BANKROLL = 1000.0
DEFAULT_SPREAD_PENALTY = 0.03
_DEFAULT_MAX_POSITION_FRACTION = 0.5
_DEFAULT_MAX_EXPOSURE_FRACTION = 1.0

RECOMMENDED_MODEL = "avg_ensemble"

_CONFIG_SPECS: dict[str, tuple[FeeType, float]] = {
    "edge_20_taker": (FeeType.TAKER, 0.0),
    "edge_20_maker": (FeeType.MAKER, 0.01),
}

OPTION_MODELS: dict[str, str] = dict.fromkeys(_CONFIG_SPECS, "avg_ensemble")


def get_recommended_configs(
    bankroll: float,
    spread_penalty: float,
) -> dict[str, BacktestConfig]:
    """Build recommended configs for taker and maker fee models.

    Parameters were selected from the d20260305_config_selection_sweep and
    d20260305_portfolio_kelly analyses: avg_ensemble with 5% Kelly fraction,
    multi-outcome Kelly sizing, and 0.20 buy_edge_threshold. The two configs
    differ in fee model: taker (market orders) vs maker (limit orders with
    a 0.01 price offset).
    """
    max_position_per_outcome = bankroll * _DEFAULT_MAX_POSITION_FRACTION
    max_total_exposure = bankroll * _DEFAULT_MAX_EXPOSURE_FRACTION

    return {
        name: BacktestConfig(
            trading=TradingConfig(
                kelly=KellyConfig(
                    bankroll=bankroll,
                    kelly_fraction=0.05,
                    kelly_mode=KellyMode.MULTI_OUTCOME,
                    buy_edge_threshold=0.20,
                    max_position_per_outcome=max_position_per_outcome,
                    max_total_exposure=max_total_exposure,
                ),
                fee_type=fee_type,
                limit_price_offset=limit_price_offset,
                sell_edge_threshold=NEVER_SELL_THRESHOLD,
                min_price=0.0,
                allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
            ),
            simulation=BacktestSimulationConfig(
                spread_penalty=spread_penalty,
                bankroll_mode=BankrollMode.FIXED,
            ),
        )
        for name, (fee_type, limit_price_offset) in _CONFIG_SPECS.items()
    }


RECOMMENDED_CONFIGS = get_recommended_configs(
    bankroll=DEFAULT_BANKROLL,
    spread_penalty=DEFAULT_SPREAD_PENALTY,
)
