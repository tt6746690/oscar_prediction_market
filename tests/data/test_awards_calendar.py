"""Regression tests for AwardsCalendar data integrity and UTC→local correctness.

Validates the core invariants of the calendar refactor:
1. UTC datetimes produce correct local dates via timezone conversion
2. Each calendar constant has the right number of events, orgs, and phases
3. Structural properties hold: nominations before winners, precursors before Oscar
4. Ceremony numbers match known values

These tests catch data-entry bugs in the hand-written calendar constants
(wrong year, month, org, or UTC offset) that would silently corrupt feature
availability, snapshot ordering, and reporting.
"""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardOrg,
    AwardsCalendar,
    EventPhase,
    _local_tz,
)

# ============================================================================
# Category 1: UTC→Local Date Correctness
# ============================================================================


class TestUtcToLocalCorrectness:
    """Verify that UTC datetimes in calendar constants produce correct local dates.

    The old system stored local dates directly. The new system derives them
    from UTC datetimes via timezone conversion. A wrong UTC value silently
    produces wrong local dates — these tests catch that.
    """

    KNOWN_CEREMONY_DATES: dict[int, date] = {
        # Ground truth from Oscars.org / Wikipedia
        2022: date(2022, 3, 27),  # 94th — Sunday
        2023: date(2023, 3, 12),  # 95th — Sunday
        2024: date(2024, 3, 10),  # 96th — Sunday
        2025: date(2025, 3, 2),  # 97th — Sunday
        2026: date(2026, 3, 15),  # 98th — Sunday
    }

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_oscar_ceremony_is_sunday(self, year: int) -> None:
        """Oscar ceremony always falls on a Sunday in LA."""
        cal = CALENDARS[year]
        assert cal.oscar_ceremony_date_local.weekday() == 6, (
            f"{year}: ceremony date {cal.oscar_ceremony_date_local} is not Sunday "
            f"(weekday={cal.oscar_ceremony_date_local.weekday()})"
        )

    @pytest.mark.parametrize(
        "year,expected_date",
        list(KNOWN_CEREMONY_DATES.items()),
        ids=[str(y) for y in KNOWN_CEREMONY_DATES],
    )
    def test_oscar_ceremony_local_dates_match_known_values(
        self, year: int, expected_date: date
    ) -> None:
        """Oscar ceremony local dates match Wikipedia/Oscars.org ground truth.

        This is the most important correctness check: a wrong UTC datetime
        (e.g., off by a day) would make the ceremony land on the wrong date.
        """
        assert CALENDARS[year].oscar_ceremony_date_local == expected_date

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_bafta_dates_use_london_timezone(self, year: int) -> None:
        """BAFTA events convert via Europe/London, not America/Los_Angeles.

        All BAFTA events (Jan-Feb) are in GMT (UTC+0). If we accidentally
        used LA timezone (UTC-8), the local date could shift backward by 1 day.
        """
        cal = CALENDARS[year]
        for phase in EventPhase:
            utc_dt = cal.events[(AwardOrg.BAFTA, phase)]
            london_date = utc_dt.astimezone(ZoneInfo("Europe/London")).date()
            la_date = utc_dt.astimezone(ZoneInfo("America/Los_Angeles")).date()
            resolved = cal.get_local_date(AwardOrg.BAFTA, phase)
            assert resolved == london_date, (
                f"{year} BAFTA {phase}: get_local_date={resolved}, "
                f"london={london_date}, la={la_date}"
            )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_us_events_use_la_timezone(self, year: int) -> None:
        """US org events convert via America/Los_Angeles.

        US ceremonies happen in the evening. If we accidentally used UTC
        for the date, it would shift forward by 1 day (since 10 PM PST
        is 06:00 UTC next day).
        """
        cal = CALENDARS[year]
        us_org = AwardOrg.DGA
        for phase in EventPhase:
            utc_dt = cal.events[(us_org, phase)]
            la_date = utc_dt.astimezone(ZoneInfo("America/Los_Angeles")).date()
            resolved = cal.get_local_date(us_org, phase)
            assert resolved == la_date

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_local_tz_returns_london_only_for_bafta(self, year: int) -> None:
        """Verify _local_tz routing: London for BAFTA, LA for everyone else."""
        assert _local_tz(AwardOrg.BAFTA) == ZoneInfo("Europe/London")
        for org in AwardOrg:
            if org != AwardOrg.BAFTA:
                assert _local_tz(org) == ZoneInfo("America/Los_Angeles"), f"{org}"

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_all_datetimes_are_utc(self, year: int) -> None:
        """Every datetime in every calendar must have UTC tzinfo.

        A naive datetime (tzinfo=None) would silently produce wrong local
        dates from astimezone().
        """
        for (org, phase), dt in CALENDARS[year].events.items():
            assert dt.tzinfo == UTC, f"{year} {org} {phase}: tzinfo={dt.tzinfo}, expected UTC"


# ============================================================================
# Category 2: Calendar Data Integrity
# ============================================================================


class TestCalendarDataIntegrity:
    """Structural invariants of calendar constants.

    Catches hand-written data errors: missing org, swapped phase dates,
    wrong ceremony_year, etc.
    """

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_all_calendars_have_20_events(self, year: int) -> None:
        """Each calendar should have exactly 10 orgs × 2 phases = 20 events."""
        assert len(CALENDARS[year].events) == 20, (
            f"{year}: got {len(CALENDARS[year].events)} events, expected 20"
        )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_all_calendars_have_all_orgs(self, year: int) -> None:
        """Every AwardOrg should appear in every calendar."""
        orgs = {org for org, _ in CALENDARS[year].events.keys()}
        assert orgs == set(AwardOrg), f"{year}: missing orgs {set(AwardOrg) - orgs}"

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_all_calendars_have_both_phases(self, year: int) -> None:
        """Every AwardOrg should have both NOMINATION and WINNER."""
        cal = CALENDARS[year]
        for org in AwardOrg:
            for phase in EventPhase:
                assert (org, phase) in cal.events, f"{year} {org} {phase} missing"

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_nominations_before_winners_for_each_org(self, year: int) -> None:
        """For every org, nomination UTC datetime must precede winner UTC datetime.

        A swapped pair would mean winners are announced before nominations —
        impossible in real life, and would break feature availability logic.
        """
        cal = CALENDARS[year]
        for org in AwardOrg:
            nom_dt = cal.events[(org, EventPhase.NOMINATION)]
            win_dt = cal.events[(org, EventPhase.WINNER)]
            assert nom_dt < win_dt, (
                f"{year} {org}: nomination ({nom_dt}) not before winner ({win_dt})"
            )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_precursor_winners_before_oscar_ceremony(self, year: int) -> None:
        """All precursor winners should be announced before the Oscar ceremony.

        Known exception: WGA 2024 winners (Apr 14) were after Oscars (Mar 10)
        due to the writer's strike.
        """
        from oscar_prediction_market.data.awards_calendar import PRECURSOR_ORGS

        cal = CALENDARS[year]
        oscar_dt = cal.events[(AwardOrg.OSCAR, EventPhase.WINNER)]
        for org in PRECURSOR_ORGS:
            win_dt = cal.events[(org, EventPhase.WINNER)]
            # WGA 2024 is the known exception
            if year == 2024 and org == AwardOrg.WGA:
                assert win_dt > oscar_dt, "WGA 2024 should be AFTER Oscar ceremony (strike delay)"
                continue
            assert win_dt < oscar_dt, (
                f"{year} {org}: winner ({win_dt}) not before Oscar ceremony ({oscar_dt})"
            )

    KNOWN_CEREMONY_NUMBERS: dict[int, int] = {
        2022: 94,
        2023: 95,
        2024: 96,
        2025: 97,
        2026: 98,
    }

    @pytest.mark.parametrize(
        "year,expected_number",
        list(KNOWN_CEREMONY_NUMBERS.items()),
        ids=[str(y) for y in KNOWN_CEREMONY_NUMBERS],
    )
    def test_ceremony_number_correct(self, year: int, expected_number: int) -> None:
        """Ceremony number = ceremony_year - 1928.

        94th ceremony was 2022, 97th was 2025, etc.
        """
        assert CALENDARS[year].ceremony_number == expected_number

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_oscar_noms_before_ceremony(self, year: int) -> None:
        """Oscar nominations are always announced before the ceremony."""
        cal = CALENDARS[year]
        nom_local = cal.oscar_nominations_date_local
        ceremony_local = cal.oscar_ceremony_date_local
        assert nom_local < ceremony_local, (
            f"{year}: nominations ({nom_local}) not before ceremony ({ceremony_local})"
        )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_ceremony_year_in_winner_datetime(self, year: int) -> None:
        """Oscar ceremony UTC datetime should fall in the ceremony year.

        Guards against copy-paste errors where the year in _utc() doesn't
        match ceremony_year. Oscar ceremony is always Jan-Mar of ceremony_year
        (the UTC datetime may be +1 day from LA date, but still same year).
        """
        cal = CALENDARS[year]
        oscar_dt = cal.events[(AwardOrg.OSCAR, EventPhase.WINNER)]
        assert oscar_dt.year == year, (
            f"Oscar ceremony UTC datetime year ({oscar_dt.year}) != ceremony_year ({year})"
        )


# ============================================================================
# Edge case: UTC→LA date shift
# ============================================================================


class TestUtcToLaDateShift:
    """Verify the critical UTC→LA date-shift behavior.

    A UTC datetime of ~06:00 on day N corresponds to ~10 PM PST on day N-1.
    This is the most common pattern in our calendar (evening LA galas ending
    at ~10 PM PST become early-morning next-day UTC). Getting this wrong
    would shift all computed local dates forward by 1 day.
    """

    def test_la_gala_date_shift(self) -> None:
        """10 PM PST = 06:00 UTC next day. Local date should be the gala date, not UTC date.

        Example: DGA 2025 gala on Feb 8 PST. UTC datetime = Feb 9 06:30.
        get_local_date should return Feb 8, not Feb 9.
        """
        cal = AwardsCalendar(
            ceremony_year=2025,
            events={
                (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 23, 13, 30, tzinfo=UTC),
                (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 3, 6, 0, tzinfo=UTC),
                (AwardOrg.DGA, EventPhase.NOMINATION): datetime(2025, 1, 8, 16, 0, tzinfo=UTC),
                # Feb 9 06:30 UTC = Feb 8 22:30 PST
                (AwardOrg.DGA, EventPhase.WINNER): datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
            },
        )
        assert cal.get_local_date(AwardOrg.DGA, EventPhase.WINNER) == date(2025, 2, 8)

    def test_bafta_no_date_shift(self) -> None:
        """BAFTA in GMT (winter): UTC datetime = London datetime. No date shift.

        BAFTA 2025 ceremony evening: Feb 16 21:30 UTC = Feb 16 21:30 GMT.
        Local date = Feb 16 (same as UTC date).
        """
        cal = AwardsCalendar(
            ceremony_year=2025,
            events={
                (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 23, 13, 30, tzinfo=UTC),
                (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 3, 6, 0, tzinfo=UTC),
                (AwardOrg.BAFTA, EventPhase.NOMINATION): datetime(2025, 1, 15, 8, 0, tzinfo=UTC),
                # Feb 16 21:30 UTC = Feb 16 21:30 GMT
                (AwardOrg.BAFTA, EventPhase.WINNER): datetime(2025, 2, 16, 21, 30, tzinfo=UTC),
            },
        )
        assert cal.get_local_date(AwardOrg.BAFTA, EventPhase.WINNER) == date(2025, 2, 16)

    def test_nominations_morning_no_date_shift(self) -> None:
        """Oscar noms at 5:30 AM PST = 13:30 UTC same day. No date shift.

        Unlike evening galas, morning announcements don't cross the date boundary.
        """
        cal = CALENDARS[2025]
        # Jan 23 13:30 UTC = Jan 23 5:30 AM PST
        assert cal.oscar_nominations_date_local == date(2025, 1, 23)
