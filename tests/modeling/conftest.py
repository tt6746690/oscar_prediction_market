"""Fixtures for modeling tests.

Provides synthetic NominationRecord / NominationDataset fixtures that exercise
all feature engineering code paths. Values are chosen so expected feature outputs
are hand-computable.
"""

from datetime import UTC, date, datetime

import pytest

from oscar_prediction_market.data.awards_calendar import (
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)
from oscar_prediction_market.data.schema import (
    AwardResult,
    FilmData,
    FilmMetadata,
    NominationDataset,
    NominationRecord,
    OscarCategory,
    OscarNominationInfo,
    PersonData,
)

# ---------------------------------------------------------------------------
# Minimal AwardsCalendar for ceremony year 2025 (test use)
# ---------------------------------------------------------------------------

TEST_CALENDAR = AwardsCalendar(
    ceremony_year=2025,
    events={
        # Oscar nominations — Jan 23, 2025 local
        (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 23, 13, 30, tzinfo=UTC),
        # Oscar ceremony — Mar 2, 2025 local (Mar 3 UTC)
        (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 2, 6, 0, tzinfo=UTC),
        # Golden Globe
        (AwardOrg.GOLDEN_GLOBE, EventPhase.NOMINATION): datetime(2024, 12, 9, 16, 0, tzinfo=UTC),
        (AwardOrg.GOLDEN_GLOBE, EventPhase.WINNER): datetime(2025, 1, 6, 4, 0, tzinfo=UTC),
        # Critics Choice
        (AwardOrg.CRITICS_CHOICE, EventPhase.NOMINATION): datetime(2024, 12, 12, 16, 0, tzinfo=UTC),
        (AwardOrg.CRITICS_CHOICE, EventPhase.WINNER): datetime(2025, 2, 8, 3, 0, tzinfo=UTC),
        # DGA
        (AwardOrg.DGA, EventPhase.NOMINATION): datetime(2025, 1, 8, 16, 0, tzinfo=UTC),
        (AwardOrg.DGA, EventPhase.WINNER): datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        # Annie
        (AwardOrg.ANNIE, EventPhase.NOMINATION): datetime(2024, 12, 20, 16, 0, tzinfo=UTC),
        (AwardOrg.ANNIE, EventPhase.WINNER): datetime(2025, 2, 9, 5, 30, tzinfo=UTC),
        # PGA
        (AwardOrg.PGA, EventPhase.NOMINATION): datetime(2025, 1, 16, 16, 0, tzinfo=UTC),
        (AwardOrg.PGA, EventPhase.WINNER): datetime(2025, 2, 9, 6, 0, tzinfo=UTC),
        # WGA
        (AwardOrg.WGA, EventPhase.NOMINATION): datetime(2025, 1, 15, 16, 0, tzinfo=UTC),
        (AwardOrg.WGA, EventPhase.WINNER): datetime(2025, 2, 16, 6, 0, tzinfo=UTC),
        # BAFTA
        (AwardOrg.BAFTA, EventPhase.NOMINATION): datetime(2025, 1, 15, 8, 0, tzinfo=UTC),
        (AwardOrg.BAFTA, EventPhase.WINNER): datetime(2025, 2, 16, 21, 30, tzinfo=UTC),
        # SAG
        (AwardOrg.SAG, EventPhase.NOMINATION): datetime(2025, 1, 8, 16, 0, tzinfo=UTC),
        (AwardOrg.SAG, EventPhase.WINNER): datetime(2025, 2, 24, 3, 0, tzinfo=UTC),
        # ASC
        (AwardOrg.ASC, EventPhase.NOMINATION): datetime(2025, 1, 16, 16, 0, tzinfo=UTC),
        (AwardOrg.ASC, EventPhase.WINNER): datetime(2025, 2, 24, 6, 0, tzinfo=UTC),
    },
)


# ---------------------------------------------------------------------------
# Helper to build a NominationRecord with sensible defaults
# ---------------------------------------------------------------------------


def _make_bp_record(
    *,
    title: str,
    film_id: str,
    ceremony: int,
    year_film: int,
    winner: bool,
    total_noms: int,
    noms_by_category: dict[str, int],
    wins_list: list[str] | None = None,
    metacritic: int | None = None,
    rotten_tomatoes: int | None = None,
    imdb_rating: float | None = None,
    box_office_worldwide: int | None = None,
    box_office_domestic: int | None = None,
    budget: int | None = None,
    runtime_minutes: int | None = None,
    released: date | None = None,
    genres: list[str] | None = None,
    rated: str | None = None,
    production_companies: list[str] | None = None,
    precursors: dict[str, AwardResult] | None = None,
) -> NominationRecord:
    """Build a Best Picture NominationRecord with explicit values."""
    metadata = FilmMetadata(
        film_id=film_id,
        title=title,
        metacritic=metacritic,
        rotten_tomatoes=rotten_tomatoes,
        imdb_rating=imdb_rating,
        imdb_votes=None,
        box_office_worldwide=box_office_worldwide,
        box_office_domestic=box_office_domestic,
        budget=budget,
        runtime_minutes=runtime_minutes,
        released=released,
        genres=genres or [],
        rated=rated,
        director=None,
        language=None,
        country=None,
        total_awards_wins=None,
        total_awards_nominations=None,
        production_companies=production_companies or [],
    )
    film = FilmData(
        film_id=film_id,
        title=title,
        metadata=metadata,
        oscar_noms=OscarNominationInfo(
            oscar_total_nominations=total_noms,
            oscar_total_wins=len(wins_list) if wins_list else 0,
            oscar_nominations_by_category=noms_by_category,
            oscar_wins_list=wins_list or [],
        ),
    )
    return NominationRecord(
        category=OscarCategory.BEST_PICTURE,
        ceremony=ceremony,
        year_film=year_film,
        category_winner=winner,
        nominee_name=None,
        film=film,
        person=None,
        precursors=precursors or {},
    )


def _make_actor_record(
    *,
    title: str,
    film_id: str,
    ceremony: int,
    year_film: int,
    winner: bool,
    total_noms: int,
    noms_by_category: dict[str, int],
    person_name: str,
    prev_noms_same: int = 0,
    prev_noms_any: int = 0,
    prev_wins_same: int = 0,
    prev_wins_any: int = 0,
    birth_date: date | None = None,
    tmdb_popularity: float | None = None,
    total_film_credits: int | None = None,
    metacritic: int | None = None,
    rotten_tomatoes: int | None = None,
    imdb_rating: float | None = None,
    precursors: dict[str, AwardResult] | None = None,
) -> NominationRecord:
    """Build an Actor Leading NominationRecord with person data."""
    metadata = FilmMetadata(
        film_id=film_id,
        title=title,
        rated=None,
        released=None,
        runtime_minutes=None,
        metacritic=metacritic,
        rotten_tomatoes=rotten_tomatoes,
        imdb_rating=imdb_rating,
        imdb_votes=None,
        box_office_domestic=None,
        box_office_worldwide=None,
        budget=None,
        director=None,
        language=None,
        country=None,
        total_awards_wins=None,
        total_awards_nominations=None,
        genres=["Drama"],
        production_companies=[],
    )
    film = FilmData(
        film_id=film_id,
        title=title,
        metadata=metadata,
        oscar_noms=OscarNominationInfo(
            oscar_total_nominations=total_noms,
            oscar_total_wins=0,
            oscar_nominations_by_category=noms_by_category,
            oscar_wins_list=[],
        ),
    )
    person = PersonData(
        name=person_name,
        prev_noms_same_category=prev_noms_same,
        prev_noms_any_category=prev_noms_any,
        prev_wins_same_category=prev_wins_same,
        prev_wins_any_category=prev_wins_any,
        birth_date=birth_date,
        tmdb_popularity=tmdb_popularity,
        total_film_credits=total_film_credits,
    )
    return NominationRecord(
        category=OscarCategory.ACTOR_LEADING,
        ceremony=ceremony,
        year_film=year_film,
        category_winner=winner,
        nominee_name=person_name,
        film=film,
        person=person,
        precursors=precursors or {},
    )


# ---------------------------------------------------------------------------
# Fixtures: Best Picture dataset (3 nominees, ceremony 97 = 2025)
# ---------------------------------------------------------------------------


@pytest.fixture()
def bp_dataset_3_nominees() -> NominationDataset:
    """3 BP nominees for ceremony 97 (2025) with varied metadata.

    Film A: Frontrunner — 10 noms, high scores, won PGA+DGA+GG drama
    Film B: Contender  —  6 noms, mid scores, won Critics Choice only
    Film C: Dark horse  —  3 noms, lower scores, no precursor wins

    Metacritic values: 90, 80, 70 → percentiles in year: 2/3, 1/3, 0/3
    """
    film_a = _make_bp_record(
        title="Film A",
        film_id="tt0000001",
        ceremony=97,
        year_film=2024,
        winner=True,
        total_noms=10,
        noms_by_category={
            OscarCategory.BEST_PICTURE.value: 1,
            OscarCategory.DIRECTING.value: 1,
            OscarCategory.FILM_EDITING.value: 1,
            OscarCategory.ACTOR_LEADING.value: 1,
            OscarCategory.ADAPTED_SCREENPLAY.value: 1,
        },
        metacritic=90,
        rotten_tomatoes=95,
        imdb_rating=8.5,
        box_office_worldwide=300_000_000,
        box_office_domestic=150_000_000,
        budget=80_000_000,
        runtime_minutes=150,
        released=date(2024, 11, 15),
        genres=["Drama", "Biography"],
        rated="R",
        precursors={
            "pga_bp": AwardResult(nominee=True, winner=True),
            "dga_directing": AwardResult(nominee=True, winner=True),
            "sag_ensemble": AwardResult(nominee=True, winner=False),
            "bafta_film": AwardResult(nominee=True, winner=True),
            "golden_globe_drama": AwardResult(nominee=True, winner=True),
            "golden_globe_musical": AwardResult(nominee=False, winner=False),
            "critics_choice_picture": AwardResult(nominee=True, winner=False),
        },
    )
    film_b = _make_bp_record(
        title="Film B",
        film_id="tt0000002",
        ceremony=97,
        year_film=2024,
        winner=False,
        total_noms=6,
        noms_by_category={
            OscarCategory.BEST_PICTURE.value: 1,
            OscarCategory.DIRECTING.value: 1,
            OscarCategory.FILM_EDITING.value: 1,
        },
        metacritic=80,
        rotten_tomatoes=85,
        imdb_rating=7.8,
        box_office_worldwide=150_000_000,
        box_office_domestic=80_000_000,
        budget=50_000_000,
        runtime_minutes=130,
        released=date(2024, 12, 20),
        genres=["Drama"],
        rated="PG-13",
        precursors={
            "pga_bp": AwardResult(nominee=True, winner=False),
            "dga_directing": AwardResult(nominee=True, winner=False),
            "sag_ensemble": AwardResult(nominee=True, winner=False),
            "bafta_film": AwardResult(nominee=True, winner=False),
            "golden_globe_drama": AwardResult(nominee=False, winner=False),
            "golden_globe_musical": AwardResult(nominee=True, winner=False),
            "critics_choice_picture": AwardResult(nominee=True, winner=True),
        },
    )
    film_c = _make_bp_record(
        title="Film C",
        film_id="tt0000003",
        ceremony=97,
        year_film=2024,
        winner=False,
        total_noms=3,
        noms_by_category={
            OscarCategory.BEST_PICTURE.value: 1,
            OscarCategory.ORIGINAL_SCREENPLAY.value: 1,
        },
        metacritic=70,
        rotten_tomatoes=75,
        imdb_rating=7.0,
        box_office_worldwide=50_000_000,
        box_office_domestic=25_000_000,
        budget=20_000_000,
        runtime_minutes=110,
        released=date(2024, 10, 5),
        genres=["Drama", "War"],
        rated="R",
        precursors={
            "pga_bp": AwardResult(nominee=False, winner=False),
            "dga_directing": AwardResult(nominee=False, winner=False),
            "critics_choice_picture": AwardResult(nominee=True, winner=False),
        },
    )
    return NominationDataset(
        category=OscarCategory.BEST_PICTURE,
        year_start=97,
        year_end=97,
        record_count=3,
        records=[film_a, film_b, film_c],
    )


@pytest.fixture()
def actor_dataset_3_nominees() -> NominationDataset:
    """3 Actor Leading nominees for ceremony 97 (2025) with person data.

    Actor X: Veteran — 4 prior noms, 0 wins (overdue), won SAG
    Actor Y: Rising star — 0 prior noms, won Critics Choice
    Actor Z: Legend — 2 prior noms, 1 prior win
    """
    actor_x = _make_actor_record(
        title="Film A",
        film_id="tt0000001",
        ceremony=97,
        year_film=2024,
        winner=True,
        total_noms=10,
        noms_by_category={
            OscarCategory.BEST_PICTURE.value: 1,
            OscarCategory.ACTOR_LEADING.value: 1,
            OscarCategory.DIRECTING.value: 1,
        },
        person_name="Actor X",
        prev_noms_same=4,
        prev_noms_any=5,
        prev_wins_same=0,
        prev_wins_any=0,
        birth_date=date(1970, 5, 15),
        tmdb_popularity=85.5,
        total_film_credits=45,
        metacritic=90,
        rotten_tomatoes=95,
        imdb_rating=8.5,
        precursors={
            "sag_lead_actor": AwardResult(nominee=True, winner=True),
            "bafta_lead_actor": AwardResult(nominee=True, winner=False),
            "golden_globe_actor_drama": AwardResult(nominee=True, winner=True),
            "golden_globe_actor_musical": AwardResult(nominee=False, winner=False),
            "critics_choice_actor": AwardResult(nominee=True, winner=False),
        },
    )
    actor_y = _make_actor_record(
        title="Film B",
        film_id="tt0000002",
        ceremony=97,
        year_film=2024,
        winner=False,
        total_noms=6,
        noms_by_category={
            OscarCategory.BEST_PICTURE.value: 1,
            OscarCategory.ACTOR_LEADING.value: 1,
        },
        person_name="Actor Y",
        prev_noms_same=0,
        prev_noms_any=0,
        prev_wins_same=0,
        prev_wins_any=0,
        birth_date=date(1995, 8, 20),
        tmdb_popularity=45.2,
        total_film_credits=12,
        metacritic=80,
        rotten_tomatoes=85,
        imdb_rating=7.8,
        precursors={
            "sag_lead_actor": AwardResult(nominee=True, winner=False),
            "bafta_lead_actor": AwardResult(nominee=True, winner=True),
            "golden_globe_actor_drama": AwardResult(nominee=False, winner=False),
            "golden_globe_actor_musical": AwardResult(nominee=True, winner=False),
            "critics_choice_actor": AwardResult(nominee=True, winner=True),
        },
    )
    actor_z = _make_actor_record(
        title="Film D",
        film_id="tt0000004",
        ceremony=97,
        year_film=2024,
        winner=False,
        total_noms=4,
        noms_by_category={
            OscarCategory.ACTOR_LEADING.value: 1,
        },
        person_name="Actor Z",
        prev_noms_same=2,
        prev_noms_any=3,
        prev_wins_same=1,
        prev_wins_any=1,
        birth_date=date(1960, 1, 10),
        tmdb_popularity=120.0,
        total_film_credits=80,
        metacritic=70,
        rotten_tomatoes=72,
        imdb_rating=7.0,
        precursors={
            "sag_lead_actor": AwardResult(nominee=False, winner=False),
            "golden_globe_actor_drama": AwardResult(nominee=True, winner=False),
            "golden_globe_actor_musical": AwardResult(nominee=False, winner=False),
            "critics_choice_actor": AwardResult(nominee=False, winner=False),
        },
    )
    return NominationDataset(
        category=OscarCategory.ACTOR_LEADING,
        year_start=97,
        year_end=97,
        record_count=3,
        records=[actor_x, actor_y, actor_z],
    )


@pytest.fixture()
def test_calendar() -> AwardsCalendar:
    """The 2025 awards calendar for test use."""
    return TEST_CALENDAR
