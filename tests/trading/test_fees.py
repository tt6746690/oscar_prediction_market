"""Tests for fee estimation.

Kalshi fees use a variance-based formula:
  fee = ceil(rate × C × P × (1 - P))
where P is price in dollars, C is contract count, and rate is:
  - 0.07  for taker orders (immediately matched)
  - 0.0175 for maker orders (resting, filled later)

The P*(1-P) term is the Bernoulli variance: fees are highest when the
outcome is most uncertain (P ≈ $0.50) and zero at the extremes (P=$0, P=$1).
"""

from oscar_prediction_market.trading.kalshi_client import (
    estimate_fee,
)
from oscar_prediction_market.trading.schema import (
    FeeType,
)

# ============================================================================
# Section 1: Fee Estimation
#
# Kalshi fees use a variance-based formula:
#   fee = ceil(rate × C × P × (1 - P))
# where P is price in dollars, C is contract count, and rate is:
#   - 0.07  for taker orders (immediately matched)
#   - 0.0175 for maker orders (resting, filled later)
#
# The P*(1-P) term is the Bernoulli variance: fees are highest when the
# outcome is most uncertain (P ≈ $0.50) and zero at the extremes (P=$0, P=$1).
# ============================================================================


class TestEstimateFee:
    """Kalshi fee = ceil(rate × C × P × (1 - P) × 100) / 100 in dollars."""

    def test_low_price_variance_based(self) -> None:
        """At low prices, the P*(1-P) term keeps fees small.

        Unlike the old formula (7c flat minimum), the variance-based
        formula charges proportionally less for extreme-priced contracts.
        This makes low-priced longshots cheaper to trade.

        $0.05:  ceil(0.07 × 0.05 × 0.95 × 100) / 100 = $0.01
        $0.10:  ceil(0.07 × 0.10 × 0.90 × 100) / 100 = $0.01
        $0.25:  ceil(0.07 × 0.25 × 0.75 × 100) / 100 = $0.02
        """
        assert estimate_fee(0.05, FeeType.TAKER) == 0.01
        assert estimate_fee(0.10, FeeType.TAKER) == 0.01
        assert estimate_fee(0.25, FeeType.TAKER) == 0.02

    def test_mid_price_peak_at_50c(self) -> None:
        """Fee is maximized at P=$0.50 where uncertainty is highest.

        P*(1-P) peaks at P=0.5: 0.5 * 0.5 = 0.25.
        Fee = ceil(0.07 × 0.25 × 100) / 100 = $0.02 per contract.

        At $0.40 and $0.60 (symmetric around $0.50): same fee.
        $0.40: ceil(0.07 × 0.40 × 0.60 × 100) / 100 = $0.02
        """
        assert estimate_fee(0.50, FeeType.TAKER) == 0.02
        assert estimate_fee(0.40, FeeType.TAKER) == 0.02
        assert estimate_fee(0.60, FeeType.TAKER) == 0.02

    def test_extreme_prices_zero_fee(self) -> None:
        """At P=$0 or P=$1, the outcome is certain → no fee.

        P*(1-P) = 0 at both extremes. No uncertainty, no fee.
        """
        assert estimate_fee(0.0, FeeType.TAKER) == 0.0
        assert estimate_fee(1.0, FeeType.TAKER) == 0.0

    def test_multi_contract_total_fee(self) -> None:
        """Fee scales with contract count, rounding applied to total.

        10 contracts at $0.25:
          fee = ceil(0.07 × 10 × 0.25 × 0.75 × 100) / 100 = $0.14

        Note: rounding is on the TOTAL, not per-contract.
        Per-contract fee is $0.02, so 10 × $0.02 = $0.20. But the actual total
        is $0.14 — cheaper because ceil on the aggregate is less lossy.
        """
        assert estimate_fee(0.25, FeeType.TAKER, n_contracts=10) == 0.14
        # Verify it's different from 10 × single-contract fee
        assert estimate_fee(0.25, FeeType.TAKER, n_contracts=1) * 10 == 0.20  # naïve
        assert estimate_fee(0.25, FeeType.TAKER, n_contracts=10) < 0.20  # actual

    def test_maker_fee_is_quarter_of_taker(self) -> None:
        """Maker fee rate (1.75%) is exactly 1/4 of taker rate (7%).

        At $0.50, taker = $0.02.
        At $0.50, maker = ceil(0.0175 × 0.25 × 100) / 100 = $0.01.

        Maker fees incentivize providing liquidity (resting orders).
        """
        assert estimate_fee(0.50, fee_type=FeeType.TAKER) == 0.02
        assert estimate_fee(0.50, fee_type=FeeType.MAKER) == 0.01

        # At $0.25, maker is even cheaper
        # ceil(0.0175 × 0.25 × 0.75 × 100) / 100 = $0.01
        assert estimate_fee(0.25, fee_type=FeeType.MAKER) == 0.01

    def test_symmetry_around_50c(self) -> None:
        """P*(1-P) is symmetric: fee at P = fee at (1-P).

        $0.20 and $0.80 have the same variance: 0.20 * 0.80 = 0.16.
        Both cost ceil(0.07 × 0.16 × 100) / 100 = $0.02.
        """
        assert estimate_fee(0.20, FeeType.TAKER) == estimate_fee(0.80, FeeType.TAKER)
        assert estimate_fee(0.10, FeeType.TAKER) == estimate_fee(0.90, FeeType.TAKER)
        assert estimate_fee(0.30, FeeType.TAKER) == estimate_fee(0.70, FeeType.TAKER)
