"""Shared trading config grids for Oscar backtest parameter sweeps.

The grid definition stays shared because multiple Oscar backtest one-offs
use the same sweep over Kelly fractions, entry thresholds, fee type, and
directional constraints. Runtime-dependent values such as bankroll and
spread penalty are provided explicitly so the returned configs are ready to
run with no placeholder binding step.
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

_DEFAULT_MAX_POSITION_FRACTION = 0.5
_DEFAULT_MAX_EXPOSURE_FRACTION = 1.0


def generate_full_trading_configs(
    bankroll: float,
    spread_penalty: float,
) -> list[BacktestConfig]:
    """Generate the full buy-hold backtest sweep.

    This is the research grid used when we want to understand the shape of
    the strategy family rather than just test one opinionated setup. The
    sweep is intentionally broad across sizing, entry threshold, fee model,
    and directional constraints so cross-year analysis can reveal which
    behaviors are robust and which only look good in one regime.

    Runtime inputs such as bankroll and spread penalty are passed in
    explicitly so the returned configs are immediately runnable. That keeps
    this function responsible for "what policy variants do we want to test"
    rather than mixing in a second runtime-binding step.
    """
    configs: list[BacktestConfig] = []
    max_position_per_outcome = bankroll * _DEFAULT_MAX_POSITION_FRACTION
    max_total_exposure = bankroll * _DEFAULT_MAX_EXPOSURE_FRACTION

    for kelly_fraction in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
        for buy_edge_threshold in [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
            for kelly_mode in [KellyMode.INDEPENDENT, KellyMode.MULTI_OUTCOME]:
                for fee_type in [FeeType.MAKER, FeeType.TAKER]:
                    for allowed_directions in [
                        frozenset({PositionDirection.YES}),
                        frozenset({PositionDirection.NO}),
                        frozenset({PositionDirection.YES, PositionDirection.NO}),
                    ]:
                        configs.append(
                            BacktestConfig(
                                trading=TradingConfig(
                                    kelly=KellyConfig(
                                        bankroll=bankroll,
                                        kelly_fraction=kelly_fraction,
                                        kelly_mode=kelly_mode,
                                        buy_edge_threshold=buy_edge_threshold,
                                        max_position_per_outcome=max_position_per_outcome,
                                        max_total_exposure=max_total_exposure,
                                    ),
                                    fee_type=fee_type,
                                    limit_price_offset=0.01 if fee_type == FeeType.MAKER else 0.0,
                                    sell_edge_threshold=NEVER_SELL_THRESHOLD,
                                    min_price=0.0,
                                    allowed_directions=allowed_directions,
                                ),
                                simulation=BacktestSimulationConfig(
                                    spread_penalty=spread_penalty,
                                    bankroll_mode=BankrollMode.FIXED,
                                ),
                            )
                        )

    return configs


def generate_targeted_trading_configs(
    bankroll: float,
    spread_penalty: float,
) -> list[BacktestConfig]:
    """Generate the targeted config sweep for config selection analysis.

    Fixes the three parameters whose best values are clear from d20260225
    analysis (taker fees, multi_outcome Kelly, all directions) and sweeps
    the two parameters that actually drive risk-return tradeoffs:

    - **edge_threshold** (9 values, 0.02–0.25): finer resolution than the
      full grid, especially at the low end where many configs cluster, and
      extended to 0.25 to explore very conservative entry.
    - **kelly_fraction** (3 values): mainly a robustness check since
      multi_outcome Kelly is insensitive to the fraction (it only sets the
      optimizer's initial guess). Three values confirm this.

    Total: 9 × 3 = 27 configs per model.  With 6 models, 162 total — much
    faster than the full 588 × 6 = 3,528 grid while providing the resolution
    needed for config selection.

    Fixed parameters (justified by d20260225 cross-year findings):

    - **fee_type = taker**: conservative (worst-case fees).
    - **kelly_mode = multi_outcome**: consistently outperforms independent.
    - **allowed_directions = all**: consistently outperforms yes-only/no-only.
    """
    configs: list[BacktestConfig] = []
    max_position_per_outcome = bankroll * _DEFAULT_MAX_POSITION_FRACTION
    max_total_exposure = bankroll * _DEFAULT_MAX_EXPOSURE_FRACTION

    for kelly_fraction in [0.05, 0.15, 0.25]:
        for buy_edge_threshold in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
            configs.append(
                BacktestConfig(
                    trading=TradingConfig(
                        kelly=KellyConfig(
                            bankroll=bankroll,
                            kelly_fraction=kelly_fraction,
                            kelly_mode=KellyMode.MULTI_OUTCOME,
                            buy_edge_threshold=buy_edge_threshold,
                            max_position_per_outcome=max_position_per_outcome,
                            max_total_exposure=max_total_exposure,
                        ),
                        fee_type=FeeType.TAKER,
                        limit_price_offset=0.0,
                        sell_edge_threshold=NEVER_SELL_THRESHOLD,
                        min_price=0.0,
                        allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
                    ),
                    simulation=BacktestSimulationConfig(
                        spread_penalty=spread_penalty,
                        bankroll_mode=BankrollMode.FIXED,
                    ),
                )
            )

    return configs


def generate_fast_trading_configs(
    bankroll: float,
    spread_penalty: float,
) -> list[BacktestConfig]:
    """Generate the small fast-mode grid for quick iteration.

    Fast mode preserves the main sizing/threshold tradeoff while fixing the
    less important axes to the most realistic default path: maker fees,
    YES-only trading, and the two Kelly sizing modes. The goal is not to be
    statistically exhaustive; it is to keep development loops short while
    still exercising the same code paths and the same high-level strategy
    choices as the full grid.
    """
    configs: list[BacktestConfig] = []
    max_position_per_outcome = bankroll * _DEFAULT_MAX_POSITION_FRACTION
    max_total_exposure = bankroll * _DEFAULT_MAX_EXPOSURE_FRACTION

    for kelly_fraction in [0.10, 0.20, 0.35]:
        for buy_edge_threshold in [0.05, 0.08, 0.12]:
            for kelly_mode in [KellyMode.INDEPENDENT, KellyMode.MULTI_OUTCOME]:
                configs.append(
                    BacktestConfig(
                        trading=TradingConfig(
                            kelly=KellyConfig(
                                bankroll=bankroll,
                                kelly_fraction=kelly_fraction,
                                kelly_mode=kelly_mode,
                                buy_edge_threshold=buy_edge_threshold,
                                max_position_per_outcome=max_position_per_outcome,
                                max_total_exposure=max_total_exposure,
                            ),
                            fee_type=FeeType.MAKER,
                            limit_price_offset=0.01,
                            sell_edge_threshold=NEVER_SELL_THRESHOLD,
                            min_price=0.0,
                            allowed_directions=frozenset({PositionDirection.YES}),
                        ),
                        simulation=BacktestSimulationConfig(
                            spread_penalty=spread_penalty,
                            bankroll_mode=BankrollMode.FIXED,
                        ),
                    )
                )

    return configs


def generate_trading_configs(
    bankroll: float,
    spread_penalty: float,
    fast: bool = False,
    grid: str = "full",
) -> list[BacktestConfig]:
    """Generate the shared Oscar buy-hold config grid.

    Args:
        bankroll: Starting capital per category per entry point.
        spread_penalty: Estimated bid-ask spread cost.
        fast: If True, use the fast grid (overrides ``grid``).
        grid: Which config grid to use. One of:
            - ``"full"``: broad 588-config sweep (default)
            - ``"targeted"``: 27-config focused sweep from d20260305
              config selection analysis (fixes fee_type=taker,
              kelly_mode=multi_outcome, allowed_directions=all)
    """
    if fast:
        return generate_fast_trading_configs(bankroll=bankroll, spread_penalty=spread_penalty)
    if grid == "targeted":
        return generate_targeted_trading_configs(bankroll=bankroll, spread_penalty=spread_penalty)
    return generate_full_trading_configs(bankroll=bankroll, spread_penalty=spread_penalty)
