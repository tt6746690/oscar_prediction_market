"""Tests for snapshot availability and active-snapshot resolution.

Tests use a minimal AwardsCalendar fixture with 3 snapshot keys:
  - Nominations (Jan 22): oscar nominations datetime = 13:30 UTC
  - DGA winner (Feb 8 local / Feb 9 06:30 UTC)
  - BAFTA winner (Feb 22 local / Feb 22 21:30 UTC)

The two functions under test:

  get_snapshot_sequence(calendar)
    → list[SnapshotInfo] with dir_name and event_datetime_utc

  _get_active_snapshot(timestamp, snapshot_keys, availability)
    → most recent snapshot dir_name available at that time, or None

Snapshot availability (dir_name → available_at_utc) is now computed inline
as {s.dir_name: s.event_datetime_utc + timedelta(hours=lag) for s in snapshots}.
"""

from datetime import UTC, datetime, timedelta

import pytest

from oscar_prediction_market.data.awards_calendar import (
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)
from oscar_prediction_market.trading.temporal_model import (
    SnapshotInfo,
    _get_active_snapshot,
    get_snapshot_sequence,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def calendar() -> AwardsCalendar:
    """Minimal 2025 calendar with 3 events.

    Timeline:
      Jan 22: Oscar nominations announced at 13:30 UTC
      DGA ceremony Feb 8 PST → winner datetime = Feb 9 06:30 UTC
      BAFTA ceremony Feb 22 → winner datetime = Feb 22 21:30 UTC
    """
    return AwardsCalendar(
        ceremony_year=2025,
        events={
            (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 22, 13, 30, tzinfo=UTC),
            (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 2, 6, 0, tzinfo=UTC),
            (AwardOrg.DGA, EventPhase.NOMINATION): datetime(2025, 1, 8, 16, 0, tzinfo=UTC),
            (AwardOrg.DGA, EventPhase.WINNER): datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
            (AwardOrg.BAFTA, EventPhase.NOMINATION): datetime(2025, 1, 15, 8, 0, tzinfo=UTC),
            (AwardOrg.BAFTA, EventPhase.WINNER): datetime(2025, 2, 22, 21, 30, tzinfo=UTC),
        },
    )


@pytest.fixture()
def snapshots(calendar: AwardsCalendar) -> list[SnapshotInfo]:
    return get_snapshot_sequence(calendar)


# ============================================================================
# get_snapshot_sequence + availability tests
# ============================================================================


class TestSnapshotAvailability:
    """Test snapshot availability derived from SnapshotInfo.event_datetime_utc.

    Uses actual event UTC datetimes from the calendar:
      Nominations (Jan 22): 13:30 UTC + lag
      DGA (Feb 8 local, winner datetime Feb 9 06:30 UTC): + lag
      BAFTA (Feb 22 local, winner datetime Feb 22 21:30 UTC): + lag
    """

    def test_lag_6h(self, snapshots: list[SnapshotInfo]) -> None:
        """6-hour lag after event end.

        Jan 22: 13:30 + 6h = 19:30 UTC Jan 22
        DGA:    06:30 UTC Feb 9 + 6h = 12:30 UTC Feb 9
        BAFTA:  21:30 UTC Feb 22 + 6h = 03:30 UTC Feb 23
        """
        avail = {s.dir_name: s.event_datetime_utc + timedelta(hours=6) for s in snapshots}
        assert avail["2025-01-22_oscar_noms"] == datetime(2025, 1, 22, 19, 30, tzinfo=UTC)
        assert avail["2025-02-08_dga"] == datetime(2025, 2, 9, 12, 30, tzinfo=UTC)
        assert avail["2025-02-22_bafta"] == datetime(2025, 2, 23, 3, 30, tzinfo=UTC)

    def test_lag_0h(self, snapshots: list[SnapshotInfo]) -> None:
        """Zero lag: available immediately when event ends."""
        avail = {s.dir_name: s.event_datetime_utc + timedelta(hours=0) for s in snapshots}
        assert avail["2025-01-22_oscar_noms"] == datetime(2025, 1, 22, 13, 30, tzinfo=UTC)
        assert avail["2025-02-08_dga"] == datetime(2025, 2, 9, 6, 30, tzinfo=UTC)


# ============================================================================
# _get_active_snapshot tests
# ============================================================================


class TestGetActiveSnapshot:
    """Test which snapshot is active at a given timestamp.

    Uses lag_hours=0 for simplicity:
      "2025-01-22_oscar_noms" available at 13:30 UTC (nominations announcement)
      "2025-02-08_dga" available at 06:30 UTC Feb 9 (DGA winner datetime)
      "2025-02-22_bafta" available at 21:30 UTC Feb 22 (BAFTA)
    """

    @pytest.fixture()
    def availability(self, snapshots: list[SnapshotInfo]) -> dict[str, datetime]:
        return {s.dir_name: s.event_datetime_utc for s in snapshots}

    @pytest.fixture()
    def key_names(self, snapshots: list[SnapshotInfo]) -> list[str]:
        return [s.dir_name for s in snapshots]

    def test_before_any_snapshot(
        self, key_names: list[str], availability: dict[str, datetime]
    ) -> None:
        """Timestamp before any snapshot is available → None."""
        ts = datetime(2025, 1, 22, 10, 0, tzinfo=UTC)  # before 13:30
        assert _get_active_snapshot(ts, key_names, availability) is None

    def test_after_first_snapshot(
        self, key_names: list[str], availability: dict[str, datetime]
    ) -> None:
        """Timestamp after first snapshot but before second → first snapshot."""
        ts = datetime(2025, 1, 25, 12, 0, tzinfo=UTC)
        assert _get_active_snapshot(ts, key_names, availability) == "2025-01-22_oscar_noms"

    def test_between_second_and_third(
        self, key_names: list[str], availability: dict[str, datetime]
    ) -> None:
        """Timestamp in between → latest available."""
        ts = datetime(2025, 2, 15, 0, 0, tzinfo=UTC)
        assert _get_active_snapshot(ts, key_names, availability) == "2025-02-08_dga"

    def test_after_all_snapshots(
        self, key_names: list[str], availability: dict[str, datetime]
    ) -> None:
        """Timestamp after all snapshots → last one."""
        ts = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
        assert _get_active_snapshot(ts, key_names, availability) == "2025-02-22_bafta"

    def test_exactly_at_available_time(
        self, key_names: list[str], availability: dict[str, datetime]
    ) -> None:
        """Timestamp exactly at available_at → that snapshot is active (<=)."""
        ts = availability["2025-02-08_dga"]
        assert _get_active_snapshot(ts, key_names, availability) == "2025-02-08_dga"
