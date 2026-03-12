"""Tests for build_trading_moments from the backtest strategies one-off.

build_trading_moments is the single entry point that converts a TemporalModel
+ market prices into a list[MarketSnapshot] for BacktestEngine.

These tests use synthetic data to verify the core logic:
  1. Only trading days where a snapshot is available produce moments
  2. Daily close prices used for non-hourly mode
  3. Moments are chronologically sorted

The TemporalModel (SnapshotModel) is responsible for snapshot resolution;
build_trading_moments queries it with a timestamp for each trading day.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from oscar_prediction_market.data.awards_calendar import (
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)
from oscar_prediction_market.trading.oscar_moments import (
    build_trading_moments,
)
from oscar_prediction_market.trading.temporal_model import (
    SnapshotModel,
    get_snapshot_sequence,
)


def _make_calendar() -> AwardsCalendar:
    """Minimal calendar for build_trading_moments testing."""
    return AwardsCalendar(
        ceremony_year=2025,
        events={
            (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 10, 13, 30, tzinfo=UTC),
            (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 2, 6, 0, tzinfo=UTC),
            (AwardOrg.DGA, EventPhase.NOMINATION): datetime(2025, 1, 5, 16, 0, tzinfo=UTC),
            (AwardOrg.DGA, EventPhase.WINNER): datetime(2025, 2, 2, 6, 30, tzinfo=UTC),
        },
    )


class TestBuildTradingMoments:
    """Integration test for build_trading_moments with daily-close mode.

    Setup:
      - 1 snapshot on Jan 10 (nominations), delay=0 → available Jan 10 13:30 UTC
      - Trading days: Jan 10, 11, 12
      - 2 matched model nominees: A, B (pre-renormalized to sum to 1.0)
      - Prices for all 3 trading days

    Expected: 3 moments (Jan 10–12), each using the Jan 10 snapshot.
    """

    def test_basic_daily_close_moments(self) -> None:
        """Daily-close mode: 3 trading days after snapshot → 3 moments."""
        cal = _make_calendar()
        snapshots = get_snapshot_sequence(cal)
        # Only the oscar_noms snapshot (DGA winner is after nominations but
        # before ceremony, so it's also included — but we only need oscar_noms
        # for this test)
        oscar_noms_snap = [s for s in snapshots if "oscar_noms" in s.dir_name][0]
        trading_dates = [date(2025, 1, 10), date(2025, 1, 11), date(2025, 1, 12)]

        availability = {
            s.dir_name: s.event_datetime_utc + timedelta(hours=0) for s in [oscar_noms_snap]
        }

        # Model predictions — already renormalized to matched set {A, B}
        # (SnapshotModel returns these as-is; renormalization happens upstream)
        preds_by_key = {
            oscar_noms_snap.dir_name: {"A": 0.5, "B": 0.5},
        }

        source = SnapshotModel(
            predictions_by_key=preds_by_key,
            snapshot_availability=availability,
            name="test_model",
        )

        # Market prices (daily close, in dollars)
        market_prices = {
            "2025-01-10": {"A": 0.50, "B": 0.50},
            "2025-01-11": {"A": 0.55, "B": 0.45},
            "2025-01-12": {"A": 0.60, "B": 0.40},
        }

        moments = build_trading_moments(
            trading_dates=trading_dates,
            source=source,
            market_prices_by_date=market_prices,
            hourly_candles=None,
            use_hourly=False,
        )

        assert len(moments) == 3
        # All 3 trading days produced moments (same snapshot backing them)
        assert [m.timestamp.date() for m in moments] == trading_dates
        # Predictions come from the SnapshotModel (already renormalized)
        assert moments[0].predictions["A"] == pytest.approx(0.5)
        assert moments[0].predictions["B"] == pytest.approx(0.5)
        # Prices from market_prices_by_date
        assert moments[0].prices == {"A": 0.50, "B": 0.50}
        assert moments[2].prices == {"A": 0.60, "B": 0.40}

    def test_no_snapshot_before_trading_day(self) -> None:
        """Trading day before any snapshot is available → skipped."""
        cal = _make_calendar()
        snapshots = get_snapshot_sequence(cal)
        oscar_noms_snap = [s for s in snapshots if "oscar_noms" in s.dir_name][0]
        # Trading on Jan 9, before the snapshot is available
        trading_dates = [date(2025, 1, 9)]

        availability = {
            s.dir_name: s.event_datetime_utc + timedelta(hours=0) for s in [oscar_noms_snap]
        }

        source = SnapshotModel(
            predictions_by_key={oscar_noms_snap.dir_name: {"A": 0.5}},
            snapshot_availability=availability,
            name="test_model",
        )

        moments = build_trading_moments(
            trading_dates=trading_dates,
            source=source,
            market_prices_by_date={"2025-01-09": {"A": 0.30}},
            hourly_candles=None,
            use_hourly=False,
        )

        assert moments == []
