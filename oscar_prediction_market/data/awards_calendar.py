"""Awards season calendar types and constants.

Defines the calendar of awards-season events (precursor nominations, precursor
winners, Oscar nominations, Oscar ceremony) via the ``AwardsCalendar`` model,
which maps ``(AwardOrg, EventPhase)`` pairs to UTC datetimes.

Timezone conventions & sources
------------------------------
All datetimes are UTC.  Field names use *no* ``_utc`` suffix — UTC is the
project-wide convention.

Sources & methodology:
  - Broadcast shows (Golden Globe, Critics Choice, SAG): end time of live
    broadcast.  Typically ~3 hours, key film categories in final hour.
  - Non-broadcast galas (DGA, PGA, Annie, WGA, ASC): results trickle out
    on social media during the dinner.  We use approximate ceremony end.
  - BAFTA (London): broadcast on BBC, use end time in GMT/BST.
  - Oscar nominations: early-morning LA announcement, live-streamed.
  - Oscar ceremony: live broadcast, use approximate end time.

Time-zone notes (winter = standard time for all Jan/Feb events):
  - LA galas: PST = UTC−8.  10 PM PST ≈ 06:00 UTC next day.
  - NY/broadcast shows: EST = UTC−5.  11 PM EST ≈ 04:00 UTC next day.
  - London: GMT = UTC+0.  9:30 PM GMT = 21:30 UTC same day.
  - Events after US DST start (2nd Sunday March):
    PDT = UTC−7, EDT = UTC−4.

Nomination datetimes are estimated as ~8 AM local time (16:00 UTC for US
orgs, 08:00 UTC for BAFTA) since exact times are not tracked.

These are best-effort estimates rounded to the nearest 30 min.
Refine with trade-volume inference when available.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, computed_field

# ============================================================================
# Enums
# ============================================================================


class AwardOrg(StrEnum):
    """Award organization in the awards season timeline.

    Includes Oscar plus all precursor organizations. Related to but distinct
    from PrecursorAward in schema.py — AwardOrg includes OSCAR while
    PrecursorAward is precursor-only. Values match for shared members.
    """

    OSCAR = "oscar"
    PGA = "pga"
    DGA = "dga"
    SAG = "sag"
    BAFTA = "bafta"
    GOLDEN_GLOBE = "golden_globe"
    CRITICS_CHOICE = "critics_choice"
    WGA = "wga"
    ASC = "asc"
    ANNIE = "annie"


class EventPhase(StrEnum):
    """Phase of an awards event."""

    NOMINATION = "nomination"
    WINNER = "winner"


PRECURSOR_ORGS: frozenset[AwardOrg] = frozenset(o for o in AwardOrg if o != AwardOrg.OSCAR)


# ============================================================================
# Timezone helpers
# ============================================================================

_TZ_LONDON = ZoneInfo("Europe/London")
_TZ_LA = ZoneInfo("America/Los_Angeles")


def _local_tz(org: AwardOrg) -> ZoneInfo:
    """Return the local timezone for an award organization.

    BAFTA ceremonies are in London (Europe/London).
    All other organizations hold their events in the US (America/Los_Angeles).
    For US broadcast shows (SAG, Critics Choice, Golden Globe) that use EST,
    using LA timezone still gives the correct local date because all US
    ceremonies occur in the evening.
    """
    return _TZ_LONDON if org == AwardOrg.BAFTA else _TZ_LA


# ============================================================================
# Helper
# ============================================================================


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Shorthand for UTC datetime construction in calendar constant definitions."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ============================================================================
# Awards Calendar
# ============================================================================


class AwardsCalendar(BaseModel):
    """Awards season calendar for a specific ceremony year.

    Maps (organization, phase) pairs to UTC datetimes for every event in the
    awards season: precursor nominations, precursor winners, Oscar nominations,
    and the Oscar ceremony.

    The ceremony_year is the year the ceremony is held (e.g., 2026).
    The ceremony number is derived as ceremony_year - 1928
    (1st Academy Awards was in 1929).

    Event datetimes are best-effort estimates rounded to nearest 30 min.
    Sources: broadcast end times, social media trickle, or trade-volume
    inference. See module docstring for timezone conventions.

    For local dates, use get_local_date() which converts UTC datetimes
    to the organization's local timezone (Los Angeles for US, London for BAFTA).
    Convention: date fields are local, datetime fields are UTC.
    """

    model_config = {"extra": "forbid"}

    ceremony_year: int = Field(...)
    events: dict[tuple[AwardOrg, EventPhase], datetime] = Field(
        ..., description="Mapping of (organization, phase) to UTC datetime."
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def ceremony_number(self) -> int:
        """Derive ceremony number from ceremony year (1st ceremony was 1929)."""
        return self.ceremony_year - 1928

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def oscar_nominations_date_local(self) -> date:
        """Local calendar date of Oscar nominations announcement."""
        return self.get_local_date(AwardOrg.OSCAR, EventPhase.NOMINATION)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def oscar_ceremony_date_local(self) -> date:
        """Local calendar date of the Oscar ceremony."""
        return self.get_local_date(AwardOrg.OSCAR, EventPhase.WINNER)

    def get_event_datetime(self, org: AwardOrg, phase: EventPhase) -> datetime:
        """Look up the UTC datetime for an (organization, phase) pair."""
        return self.events[(org, phase)]

    def get_local_date(self, org: AwardOrg, phase: EventPhase) -> date:
        """Convert a UTC event datetime to the organization's local calendar date.

        Uses America/Los_Angeles for all US organizations,
        Europe/London for BAFTA.
        """
        utc_dt = self.events[(org, phase)]
        return utc_dt.astimezone(_local_tz(org)).date()


# ============================================================================
# Calendar Constants
# ============================================================================

CALENDAR_2022 = AwardsCalendar(
    ceremony_year=2022,
    events={
        # Critics Choice nominations — Dec 13, 2021 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.NOMINATION): _utc(2021, 12, 13, 16, 0),  # ~8 AM PST
        # Golden Globe nominations — Dec 13, 2021 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.NOMINATION): _utc(2021, 12, 13, 16, 0),  # ~8 AM PST
        # Annie nominations — Dec 21, 2021 local
        (AwardOrg.ANNIE, EventPhase.NOMINATION): _utc(2021, 12, 21, 16, 0),  # ~8 AM PST
        # Golden Globe winners — Jan 9, 2022 local
        # 2022 Golden Globes: non-broadcast (HFPA controversy), results via social media
        (AwardOrg.GOLDEN_GLOBE, EventPhase.WINNER): _utc(
            2022, 1, 10, 2, 0
        ),  # results ~6 PM PST Jan 9
        # SAG nominations — Jan 12, 2022 local
        (AwardOrg.SAG, EventPhase.NOMINATION): _utc(2022, 1, 12, 16, 0),  # ~8 AM PST
        # ASC nominations — Jan 25, 2022 local
        (AwardOrg.ASC, EventPhase.NOMINATION): _utc(2022, 1, 25, 16, 0),  # ~8 AM PST
        # DGA nominations — Jan 27, 2022 local
        (AwardOrg.DGA, EventPhase.NOMINATION): _utc(2022, 1, 27, 16, 0),  # ~8 AM PST
        # PGA nominations — Jan 27, 2022 local
        (AwardOrg.PGA, EventPhase.NOMINATION): _utc(2022, 1, 27, 16, 0),  # ~8 AM PST
        # WGA nominations — Jan 27, 2022 local
        (AwardOrg.WGA, EventPhase.NOMINATION): _utc(2022, 1, 27, 16, 0),  # ~8 AM PST
        # BAFTA nominations — Feb 3, 2022 local
        (AwardOrg.BAFTA, EventPhase.NOMINATION): _utc(2022, 2, 3, 8, 0),  # ~8 AM GMT
        # Oscar nominations — Feb 8, 2022 local
        (AwardOrg.OSCAR, EventPhase.NOMINATION): _utc(2022, 2, 8, 13, 30),  # ~5:30 AM PST
        # SAG winners — Feb 27, 2022 local
        (AwardOrg.SAG, EventPhase.WINNER): _utc(2022, 2, 28, 3, 0),  # broadcast ~10 PM EST Feb 27
        # Annie winners — Mar 12, 2022 local
        (AwardOrg.ANNIE, EventPhase.WINNER): _utc(
            2022, 3, 13, 5, 30
        ),  # ceremony ~9:30 PM PST Mar 12
        # DGA winners — Mar 12, 2022 local
        (AwardOrg.DGA, EventPhase.WINNER): _utc(
            2022, 3, 13, 6, 30
        ),  # dinner gala ~10:30 PM PST Mar 12
        # BAFTA winners — Mar 13, 2022 local
        (AwardOrg.BAFTA, EventPhase.WINNER): _utc(
            2022, 3, 13, 21, 30
        ),  # BBC broadcast ~9:30 PM GMT Mar 13
        # Critics Choice winners — Mar 13, 2022 local
        # Mar 13 is DST start day (2 AM → 3 AM). CCA was evening EDT.
        (AwardOrg.CRITICS_CHOICE, EventPhase.WINNER): _utc(
            2022, 3, 14, 3, 0
        ),  # broadcast ~11 PM EDT Mar 13
        # PGA winners — Mar 19, 2022 local
        (AwardOrg.PGA, EventPhase.WINNER): _utc(2022, 3, 20, 5, 0),  # dinner gala ~10 PM PDT Mar 19
        # ASC winners — Mar 20, 2022 local
        (AwardOrg.ASC, EventPhase.WINNER): _utc(2022, 3, 21, 5, 0),  # ceremony ~10 PM PDT Mar 20
        # WGA winners — Mar 20, 2022 local
        (AwardOrg.WGA, EventPhase.WINNER): _utc(2022, 3, 21, 5, 0),  # ceremony ~10 PM PDT Mar 20
        # Oscar ceremony — Mar 27, 2022 local
        (AwardOrg.OSCAR, EventPhase.WINNER): _utc(2022, 3, 28, 5, 0),  # 10 PM PDT Mar 27
    },
)

CALENDAR_2023 = AwardsCalendar(
    ceremony_year=2023,
    events={
        # Golden Globe nominations — Dec 12, 2022 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.NOMINATION): _utc(2022, 12, 12, 16, 0),  # ~8 AM PST
        # Critics Choice nominations — Dec 14, 2022 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.NOMINATION): _utc(2022, 12, 14, 16, 0),  # ~8 AM PST
        # ASC nominations — Jan 9, 2023 local
        (AwardOrg.ASC, EventPhase.NOMINATION): _utc(2023, 1, 9, 16, 0),  # ~8 AM PST
        # Golden Globe winners — Jan 10, 2023 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.WINNER): _utc(
            2023, 1, 11, 4, 0
        ),  # broadcast ~11 PM EST Jan 10
        # DGA nominations — Jan 11, 2023 local
        (AwardOrg.DGA, EventPhase.NOMINATION): _utc(2023, 1, 11, 16, 0),  # ~8 AM PST
        # SAG nominations — Jan 11, 2023 local
        (AwardOrg.SAG, EventPhase.NOMINATION): _utc(2023, 1, 11, 16, 0),  # ~8 AM PST
        # PGA nominations — Jan 12, 2023 local
        (AwardOrg.PGA, EventPhase.NOMINATION): _utc(2023, 1, 12, 16, 0),  # ~8 AM PST
        # Critics Choice winners — Jan 15, 2023 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.WINNER): _utc(
            2023, 1, 16, 4, 0
        ),  # broadcast ~11 PM EST Jan 15
        # Annie nominations — Jan 17, 2023 local
        (AwardOrg.ANNIE, EventPhase.NOMINATION): _utc(2023, 1, 17, 16, 0),  # ~8 AM PST
        # BAFTA nominations — Jan 19, 2023 local
        (AwardOrg.BAFTA, EventPhase.NOMINATION): _utc(2023, 1, 19, 8, 0),  # ~8 AM GMT
        # Oscar nominations — Jan 24, 2023 local
        (AwardOrg.OSCAR, EventPhase.NOMINATION): _utc(2023, 1, 24, 13, 30),  # ~5:30 AM PST
        # WGA nominations — Jan 25, 2023 local
        (AwardOrg.WGA, EventPhase.NOMINATION): _utc(2023, 1, 25, 16, 0),  # ~8 AM PST
        # DGA winners — Feb 18, 2023 local
        (AwardOrg.DGA, EventPhase.WINNER): _utc(
            2023, 2, 19, 6, 30
        ),  # dinner gala ~10:30 PM PST Feb 18
        # BAFTA winners — Feb 19, 2023 local
        (AwardOrg.BAFTA, EventPhase.WINNER): _utc(
            2023, 2, 19, 21, 30
        ),  # BBC broadcast ~9:30 PM GMT Feb 19
        # Annie winners — Feb 25, 2023 local
        (AwardOrg.ANNIE, EventPhase.WINNER): _utc(
            2023, 2, 26, 5, 30
        ),  # ceremony ~9:30 PM PST Feb 25
        # PGA winners — Feb 25, 2023 local
        (AwardOrg.PGA, EventPhase.WINNER): _utc(2023, 2, 26, 6, 0),  # dinner gala ~10 PM PST Feb 25
        # SAG winners — Feb 26, 2023 local
        (AwardOrg.SAG, EventPhase.WINNER): _utc(2023, 2, 27, 3, 0),  # broadcast ~10 PM EST Feb 26
        # ASC winners — Mar 5, 2023 local
        (AwardOrg.ASC, EventPhase.WINNER): _utc(2023, 3, 6, 6, 0),  # ceremony ~10 PM PST Mar 5
        # WGA winners — Mar 5, 2023 local
        (AwardOrg.WGA, EventPhase.WINNER): _utc(2023, 3, 6, 6, 0),  # ceremony ~10 PM PST Mar 5
        # Oscar ceremony — Mar 12, 2023 local
        (AwardOrg.OSCAR, EventPhase.WINNER): _utc(2023, 3, 13, 5, 0),  # 10 PM PDT Mar 12
    },
)

CALENDAR_2024 = AwardsCalendar(
    ceremony_year=2024,
    events={
        # Golden Globe nominations — Dec 11, 2023 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.NOMINATION): _utc(2023, 12, 11, 16, 0),  # ~8 AM PST
        # Critics Choice nominations — Dec 13, 2023 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.NOMINATION): _utc(2023, 12, 13, 16, 0),  # ~8 AM PST
        # Golden Globe winners — Jan 7, 2024 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.WINNER): _utc(
            2024, 1, 8, 4, 0
        ),  # broadcast ~11 PM EST Jan 7
        # DGA nominations — Jan 10, 2024 local
        (AwardOrg.DGA, EventPhase.NOMINATION): _utc(2024, 1, 10, 16, 0),  # ~8 AM PST
        # SAG nominations — Jan 10, 2024 local
        (AwardOrg.SAG, EventPhase.NOMINATION): _utc(2024, 1, 10, 16, 0),  # ~8 AM PST
        # Annie nominations — Jan 11, 2024 local
        (AwardOrg.ANNIE, EventPhase.NOMINATION): _utc(2024, 1, 11, 16, 0),  # ~8 AM PST
        # ASC nominations — Jan 11, 2024 local
        (AwardOrg.ASC, EventPhase.NOMINATION): _utc(2024, 1, 11, 16, 0),  # ~8 AM PST
        # PGA nominations — Jan 12, 2024 local
        (AwardOrg.PGA, EventPhase.NOMINATION): _utc(2024, 1, 12, 16, 0),  # ~8 AM PST
        # Critics Choice winners — Jan 14, 2024 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.WINNER): _utc(
            2024, 1, 15, 4, 0
        ),  # broadcast ~11 PM EST Jan 14
        # BAFTA nominations — Jan 18, 2024 local
        (AwardOrg.BAFTA, EventPhase.NOMINATION): _utc(2024, 1, 18, 8, 0),  # ~8 AM GMT
        # Oscar nominations — Jan 23, 2024 local
        (AwardOrg.OSCAR, EventPhase.NOMINATION): _utc(2024, 1, 23, 13, 30),  # ~5:30 AM PST
        # DGA winners — Feb 10, 2024 local
        (AwardOrg.DGA, EventPhase.WINNER): _utc(
            2024, 2, 11, 6, 30
        ),  # dinner gala ~10:30 PM PST Feb 10
        # Annie winners — Feb 17, 2024 local
        (AwardOrg.ANNIE, EventPhase.WINNER): _utc(
            2024, 2, 18, 5, 30
        ),  # ceremony ~9:30 PM PST Feb 17
        # BAFTA winners — Feb 18, 2024 local
        (AwardOrg.BAFTA, EventPhase.WINNER): _utc(
            2024, 2, 18, 21, 30
        ),  # BBC broadcast ~9:30 PM GMT Feb 18
        # WGA nominations — Feb 21, 2024 local
        (AwardOrg.WGA, EventPhase.NOMINATION): _utc(2024, 2, 21, 16, 0),  # ~8 AM PST
        # SAG winners — Feb 24, 2024 local
        (AwardOrg.SAG, EventPhase.WINNER): _utc(2024, 2, 25, 3, 0),  # broadcast ~10 PM EST Feb 24
        # PGA winners — Feb 25, 2024 local
        (AwardOrg.PGA, EventPhase.WINNER): _utc(2024, 2, 26, 6, 0),  # dinner gala ~10 PM PST Feb 25
        # ASC winners — Mar 3, 2024 local
        (AwardOrg.ASC, EventPhase.WINNER): _utc(2024, 3, 4, 6, 0),  # ceremony ~10 PM PST Mar 3
        # Oscar ceremony — Mar 10, 2024 local
        (AwardOrg.OSCAR, EventPhase.WINNER): _utc(2024, 3, 11, 5, 0),  # 10 PM PDT Mar 10
        # WGA winners — Apr 14, 2024 local
        (AwardOrg.WGA, EventPhase.WINNER): _utc(
            2024, 4, 15, 5, 0
        ),  # ceremony ~10 PM PDT Apr 14 (post-DST)
    },
)

CALENDAR_2025 = AwardsCalendar(
    ceremony_year=2025,
    events={
        # Golden Globe nominations — Dec 9, 2024 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.NOMINATION): _utc(2024, 12, 9, 16, 0),  # ~8 AM PST
        # Critics Choice nominations — Dec 12, 2024 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.NOMINATION): _utc(2024, 12, 12, 16, 0),  # ~8 AM PST
        # Annie nominations — Dec 20, 2024 local
        (AwardOrg.ANNIE, EventPhase.NOMINATION): _utc(2024, 12, 20, 16, 0),  # ~8 AM PST
        # Golden Globe winners — Jan 5, 2025 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.WINNER): _utc(
            2025, 1, 6, 4, 0
        ),  # broadcast ~11 PM EST Jan 5
        # SAG nominations — Jan 8, 2025 local
        (AwardOrg.SAG, EventPhase.NOMINATION): _utc(2025, 1, 8, 16, 0),  # ~8 AM PST
        # DGA nominations — Jan 8, 2025 local
        (AwardOrg.DGA, EventPhase.NOMINATION): _utc(2025, 1, 8, 16, 0),  # ~8 AM PST
        # BAFTA nominations — Jan 15, 2025 local
        (AwardOrg.BAFTA, EventPhase.NOMINATION): _utc(2025, 1, 15, 8, 0),  # ~8 AM GMT
        # WGA nominations — Jan 15, 2025 local
        (AwardOrg.WGA, EventPhase.NOMINATION): _utc(2025, 1, 15, 16, 0),  # ~8 AM PST
        # ASC nominations — Jan 16, 2025 local
        (AwardOrg.ASC, EventPhase.NOMINATION): _utc(2025, 1, 16, 16, 0),  # ~8 AM PST
        # PGA nominations — Jan 16, 2025 local
        (AwardOrg.PGA, EventPhase.NOMINATION): _utc(2025, 1, 16, 16, 0),  # ~8 AM PST
        # Oscar nominations — Jan 23, 2025 local
        (AwardOrg.OSCAR, EventPhase.NOMINATION): _utc(2025, 1, 23, 13, 30),  # ~5:30 AM PST
        # Critics Choice winners — Feb 7, 2025 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.WINNER): _utc(
            2025, 2, 8, 3, 0
        ),  # broadcast ~10 PM EST Feb 7
        # Annie winners — Feb 8, 2025 local
        (AwardOrg.ANNIE, EventPhase.WINNER): _utc(2025, 2, 9, 5, 30),  # ceremony ~9:30 PM PST Feb 8
        # PGA winners — Feb 8, 2025 local
        (AwardOrg.PGA, EventPhase.WINNER): _utc(2025, 2, 9, 6, 0),  # dinner gala ~10 PM PST Feb 8
        # DGA winners — Feb 8, 2025 local
        (AwardOrg.DGA, EventPhase.WINNER): _utc(
            2025, 2, 9, 6, 30
        ),  # dinner gala ~10:30 PM PST Feb 8
        # WGA winners — Feb 15, 2025 local
        (AwardOrg.WGA, EventPhase.WINNER): _utc(2025, 2, 16, 6, 0),  # ceremony ~10 PM PST Feb 15
        # BAFTA winners — Feb 16, 2025 local
        (AwardOrg.BAFTA, EventPhase.WINNER): _utc(
            2025, 2, 16, 21, 30
        ),  # BBC broadcast ~9:30 PM GMT Feb 16
        # SAG winners — Feb 23, 2025 local
        (AwardOrg.SAG, EventPhase.WINNER): _utc(2025, 2, 24, 3, 0),  # broadcast ~10 PM EST Feb 23
        # ASC winners — Feb 23, 2025 local
        (AwardOrg.ASC, EventPhase.WINNER): _utc(2025, 2, 24, 6, 0),  # ceremony ~10 PM PST Feb 23
        # Oscar ceremony — Mar 2, 2025 local
        (AwardOrg.OSCAR, EventPhase.WINNER): _utc(2025, 3, 3, 6, 0),  # 10 PM PST Mar 2
    },
)

CALENDAR_2026 = AwardsCalendar(
    ceremony_year=2026,
    events={
        # Critics Choice nominations — Dec 5, 2025 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.NOMINATION): _utc(2025, 12, 5, 16, 0),  # ~8 AM PST
        # Golden Globe nominations — Dec 8, 2025 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.NOMINATION): _utc(2025, 12, 8, 16, 0),  # ~8 AM PST
        # Critics Choice winners — Jan 4, 2026 local
        (AwardOrg.CRITICS_CHOICE, EventPhase.WINNER): _utc(
            2026, 1, 5, 4, 0
        ),  # broadcast ~11 PM EST
        # Annie nominations — Jan 5, 2026 local
        (AwardOrg.ANNIE, EventPhase.NOMINATION): _utc(2026, 1, 5, 16, 0),  # ~8 AM PST
        # SAG nominations — Jan 7, 2026 local
        (AwardOrg.SAG, EventPhase.NOMINATION): _utc(2026, 1, 7, 16, 0),  # ~8 AM PST
        # DGA nominations — Jan 8, 2026 local
        (AwardOrg.DGA, EventPhase.NOMINATION): _utc(2026, 1, 8, 16, 0),  # ~8 AM PST
        # ASC nominations — Jan 8, 2026 local
        (AwardOrg.ASC, EventPhase.NOMINATION): _utc(2026, 1, 8, 16, 0),  # ~8 AM PST
        # PGA nominations — Jan 9, 2026 local
        (AwardOrg.PGA, EventPhase.NOMINATION): _utc(2026, 1, 9, 16, 0),  # ~8 AM PST
        # Golden Globe winners — Jan 11, 2026 local
        (AwardOrg.GOLDEN_GLOBE, EventPhase.WINNER): _utc(2026, 1, 12, 4, 0),  # broadcast ~11 PM EST
        # Oscar nominations — Jan 22, 2026 local
        (AwardOrg.OSCAR, EventPhase.NOMINATION): _utc(2026, 1, 22, 13, 30),  # ~5:30 AM PST
        # BAFTA nominations — Jan 27, 2026 local
        (AwardOrg.BAFTA, EventPhase.NOMINATION): _utc(2026, 1, 27, 8, 0),  # ~8 AM GMT
        # WGA nominations — Jan 27, 2026 local
        (AwardOrg.WGA, EventPhase.NOMINATION): _utc(2026, 1, 27, 16, 0),  # ~8 AM PST
        # DGA winners — Feb 7, 2026 local
        (AwardOrg.DGA, EventPhase.WINNER): _utc(2026, 2, 8, 6, 30),  # dinner gala ~10:30 PM PST
        # Annie winners — Feb 21, 2026 local
        (AwardOrg.ANNIE, EventPhase.WINNER): _utc(2026, 2, 22, 5, 30),  # ceremony ~9:30 PM PST
        # BAFTA winners — Feb 22, 2026 local
        (AwardOrg.BAFTA, EventPhase.WINNER): _utc(
            2026, 2, 22, 21, 30
        ),  # BBC broadcast ~9:30 PM GMT
        # PGA winners — Feb 28, 2026 local
        (AwardOrg.PGA, EventPhase.WINNER): _utc(2026, 3, 1, 6, 0),  # dinner gala ~10 PM PST
        # SAG winners — Mar 1, 2026 local
        (AwardOrg.SAG, EventPhase.WINNER): _utc(2026, 3, 2, 3, 0),  # broadcast ~10 PM EST
        # ASC winners — Mar 8, 2026 local
        (AwardOrg.ASC, EventPhase.WINNER): _utc(
            2026, 3, 9, 5, 0
        ),  # ceremony ~10 PM PDT (after DST)
        # WGA winners — Mar 8, 2026 local
        (AwardOrg.WGA, EventPhase.WINNER): _utc(
            2026, 3, 9, 5, 0
        ),  # ceremony ~10 PM PDT (after DST)
        # Oscar ceremony — Mar 15, 2026 local
        (AwardOrg.OSCAR, EventPhase.WINNER): _utc(2026, 3, 16, 5, 0),  # 10 PM PDT Mar 15
    },
)

CALENDARS: dict[int, AwardsCalendar] = {
    2022: CALENDAR_2022,
    2023: CALENDAR_2023,
    2024: CALENDAR_2024,
    2025: CALENDAR_2025,
    2026: CALENDAR_2026,
}
