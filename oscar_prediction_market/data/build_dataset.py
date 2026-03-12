"""Build consolidated Oscar raw dataset for any category.

This module builds the RAW dataset (Layer 2) from intermediate sources.
Feature engineering is done separately in feature_engineering.py.

Pipeline stages:
1. oscar: Extract Oscar nominations from oscars.csv -> oscar_nominations.json
2. metadata: Fetch film metadata from OMDb + TMDb -> film_metadata.json
3. precursors: Fetch precursor awards from Wikipedia -> precursor_awards.json
4. person: Build person career data from oscars.csv (for person-level categories)
5. merge: Merge all intermediate files -> oscar_{category}_raw.json

Usage:
    # Build Best Picture dataset
    uv run python -m ...build_dataset --category BEST_PICTURE --year-start 2000 --year-end 2026

    # Build Best Actor dataset
    uv run python -m ...build_dataset --category ACTOR_LEADING --year-start 2000 --year-end 2026

    # Run individual stages
    uv run python -m ...build_dataset --mode oscar --category BEST_PICTURE
    uv run python -m ...build_dataset --mode merge --category BEST_PICTURE --as-of-date 2026-02-04
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from oscar_prediction_market.constants import TMDB_API_KEY
from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)
from oscar_prediction_market.data.fetch_omdb import OMDbFetcher
from oscar_prediction_market.data.fetch_precursor_awards import (
    PrecursorAwardsFetcher,
)
from oscar_prediction_market.data.fetch_tmdb import TMDbFetcher
from oscar_prediction_market.data.precursor_mappings import (
    PrecursorSpec,
    get_precursor_specs,
)
from oscar_prediction_market.data.schema import (
    PERSON_CATEGORIES,
    AwardResult,
    FilmData,
    FilmMetadata,
    FilmMetadataFile,
    NominationDataset,
    NominationRecord,
    OscarCategory,
    OscarNominationInfo,
    OscarNominationsFile,
    OscarNominee,
    PersonData,
    PrecursorAward,
    PrecursorAwardRecord,
    PrecursorAwardsFile,
)
from oscar_prediction_market.data.utils import (
    clean_screenplay_names,
    normalize_person_name,
)
from oscar_prediction_market.modeling.utils import (
    year_to_ceremony,
)

# Paths
BASE_DIR = Path(__file__).parent
OSCAR_DATA_PATH = BASE_DIR / "oscars.csv"


def load_oscar_data(min_ceremony: int, max_ceremony: int) -> pd.DataFrame:
    """Load Oscar nomination data from DLu oscars.csv."""
    df = pd.read_csv(OSCAR_DATA_PATH, sep="\t")
    df = df[(df["Ceremony"] >= min_ceremony) & (df["Ceremony"] <= max_ceremony)]
    df["Winner"] = df["Winner"].apply(lambda x: x is True or x == "True")
    return df


def get_all_nominations_for_film(df: pd.DataFrame, film_id: str) -> dict:
    """Get all Oscar nominations for a film across all categories.

    Returns raw nomination data only. Derived features (has_director_nom, etc.)
    are computed in the feature engineering layer.
    """
    film_df = df[df["FilmId"].str.contains(film_id, na=False)]

    category_counts = film_df["CanonicalCategory"].value_counts().to_dict()
    wins = film_df[film_df["Winner"]]["CanonicalCategory"].tolist()

    return {
        "total_nominations": len(film_df),
        "total_wins": len(wins),
        "nominations_by_category": category_counts,
        "wins": wins,
    }


def build_oscar_nominations(
    category: OscarCategory,
    year_start: int,
    year_end: int,
) -> OscarNominationsFile:
    """Stage 1: Extract Oscar nominations from oscars.csv for a given category."""
    min_ceremony = year_to_ceremony(year_start)
    max_ceremony = year_to_ceremony(year_end)

    print(f"Loading Oscar data (ceremonies {min_ceremony}-{max_ceremony})...")
    df = load_oscar_data(min_ceremony, max_ceremony)
    print(f"Total nominations loaded: {len(df)}")

    # Filter to the target category
    cat_df = df[df["CanonicalCategory"] == category.value].copy()
    cat_df = cat_df[
        ["Ceremony", "Year", "Film", "FilmId", "Winner", "Name", "Nominees", "NomineeIds"]
    ]
    unique_nominees = cat_df.drop_duplicates(subset=["FilmId", "Ceremony", "Name"])
    print(f"{category.value} nominees: {len(unique_nominees)}")

    records = []
    total_nominees = len(unique_nominees)

    for i, (_, row) in enumerate(unique_nominees.iterrows()):
        film_id = row["FilmId"]
        if pd.isna(film_id):
            print(f"  Skipping {row['Film']} - no FilmId")
            continue

        primary_film_id = film_id.split("|")[0] if "|" in str(film_id) else film_id
        print(f"[{i + 1}/{total_nominees}] {row['Film']}")

        nom_data = get_all_nominations_for_film(df, primary_film_id)

        # For person-level categories, extract the person name
        nominee_name: str | None = None
        if category in PERSON_CATEGORIES:
            raw_name = row["Name"] if pd.notna(row["Name"]) else None
            # Clean screenplay credits: strip "Written by", "Screenplay by" prefixes
            # and normalize multi-writer credits to comma-separated
            if raw_name is not None and category == OscarCategory.ORIGINAL_SCREENPLAY:
                nominee_name = clean_screenplay_names(raw_name)
            else:
                nominee_name = raw_name

        record = OscarNominee(
            film_id=primary_film_id,
            title=row["Film"],
            ceremony=int(row["Ceremony"]),
            year_film=int(row["Year"]),
            category=category,
            category_winner=bool(row["Winner"]),
            nominee_name=nominee_name,
            oscar_total_nominations=nom_data["total_nominations"],
            oscar_total_wins=nom_data["total_wins"],
            oscar_nominations_by_category=nom_data["nominations_by_category"],
            oscar_wins_list=nom_data["wins"],
        )
        records.append(record)

    return OscarNominationsFile(
        category=category,
        year_start=year_start,
        year_end=year_end,
        record_count=len(records),
        records=records,
    )


# ============================================================================
# Stage 2: Film Metadata (OMDb + TMDb)
# ============================================================================


def build_film_metadata(
    year_start: int, year_end: int, input_dirs: Path | list[Path]
) -> FilmMetadataFile:
    """Stage 2: Fetch film metadata from OMDb + TMDb.

    Args:
        year_start: Start ceremony year.
        year_end: End ceremony year.
        input_dirs: One or more directories containing oscar_nominations.json files.
            When multiple directories are provided, film IDs are collected from all
            of them and deduplicated before fetching metadata. This enables building
            a single shared metadata file covering all categories at once.
    """
    # Normalize to list
    dirs = [input_dirs] if isinstance(input_dirs, Path) else input_dirs

    # Collect unique film IDs across all input dirs
    film_ids: set[str] = set()
    for d in dirs:
        oscar_path = d / "oscar_nominations.json"
        if not oscar_path.exists():
            raise FileNotFoundError(f"Run --mode oscar first: {oscar_path}")
        with open(oscar_path) as f:
            oscar_data = OscarNominationsFile(**json.load(f))
        film_ids.update(r.film_id for r in oscar_data.records)
    print(f"Fetching metadata for {len(film_ids)} unique films (across {len(dirs)} dir(s))...")

    omdb_fetcher = OMDbFetcher()
    tmdb_fetcher = TMDbFetcher(api_key=TMDB_API_KEY)

    # Sort for deterministic ordering
    film_ids_sorted = sorted(film_ids)
    total = len(film_ids_sorted)
    records: dict[str, FilmMetadata] = {}

    for i, film_id in enumerate(film_ids_sorted):
        print(f"[{i + 1}/{total}] {film_id}")

        # OMDb
        omdb_data = omdb_fetcher.fetch(film_id)

        # TMDb
        tmdb_data = tmdb_fetcher.fetch_by_imdb_id(film_id)

        # Combine
        metadata = FilmMetadata(
            film_id=film_id,
            title=omdb_data.title if omdb_data else None,
            rated=omdb_data.rated if omdb_data else None,
            released=omdb_data.released if omdb_data else None,
            runtime_minutes=omdb_data.runtime_minutes if omdb_data else None,
            genres=omdb_data.genres if omdb_data else [],
            director=omdb_data.director if omdb_data else None,
            actors=omdb_data.actors if omdb_data else [],
            language=omdb_data.language if omdb_data else None,
            country=omdb_data.country if omdb_data else None,
            imdb_rating=omdb_data.imdb_rating if omdb_data else None,
            imdb_votes=omdb_data.imdb_votes if omdb_data else None,
            rotten_tomatoes=omdb_data.rotten_tomatoes if omdb_data else None,
            metacritic=omdb_data.metacritic if omdb_data else None,
            box_office_domestic=omdb_data.box_office_domestic if omdb_data else None,
            total_awards_wins=(omdb_data.awards_parsed.total_wins if omdb_data else None),
            total_awards_nominations=(
                omdb_data.awards_parsed.total_nominations if omdb_data else None
            ),
            box_office_worldwide=tmdb_data.revenue if tmdb_data else None,
            budget=tmdb_data.budget if tmdb_data else None,
            production_companies=tmdb_data.production_companies if tmdb_data else [],
        )
        records[film_id] = metadata

    return FilmMetadataFile(
        year_start=year_start,
        year_end=year_end,
        record_count=len(records),
        records=records,
    )


# ============================================================================
# Stage 3: Precursor Awards
# ============================================================================


def build_precursor_awards(year_start: int, year_end: int) -> PrecursorAwardsFile:
    """Stage 3: Fetch precursor awards from Wikipedia.

    Note: Precursor awards use film release year (year_ceremony in schema), not Oscar
    ceremony year. For ceremony year 2000, films are from 1999, so we need to fetch
    precursor awards starting from year_start - 1 to cover all films.
    """
    # Fetch one year earlier to cover films released in year before first ceremony
    # E.g., for ceremony year 2000, we need films from 1999
    precursor_year_start = year_start - 1

    print(f"Fetching precursor awards for {precursor_year_start}-{year_end}...")
    print(
        f"  (Covers film years {precursor_year_start}-{year_end - 1} "
        f"for ceremonies {year_start}-{year_end})"
    )

    fetcher = PrecursorAwardsFetcher()
    raw_awards = fetcher.fetch_all_awards(
        year_range=(precursor_year_start, year_end), progress=True
    )

    awards: dict[str, list[PrecursorAwardRecord]] = {}

    for award_name, df in raw_awards.items():
        records = []
        for _, row in df.iterrows():
            records.append(
                PrecursorAwardRecord(
                    year_ceremony=int(row["year_ceremony"]),
                    film=str(row["film"]),
                    person=str(row["person"])
                    if "person" in row and pd.notna(row["person"])
                    else None,
                    is_winner=bool(row["is_winner"]),
                )
            )
        awards[award_name] = records
        print(f"  {award_name}: {len(records)} records")

    return PrecursorAwardsFile(
        year_start=precursor_year_start,
        year_end=year_end,
        awards=awards,
    )


# ============================================================================
# Name Normalization
# ============================================================================


# ============================================================================
# Stage 4: Person Data (from oscars.csv career history + TMDb enrichment)
# ============================================================================


def build_person_data(
    category: OscarCategory,
    oscar_df: pd.DataFrame,
    nominees: list[OscarNominee],
    tmdb_fetcher: TMDbFetcher | None = None,
) -> dict[tuple[str, str, int], PersonData]:
    """Build person career data for all nominees in a category.

    Uses oscars.csv to compute prior noms/wins for each person.
    Optionally enriches with TMDb person data (birth_date, popularity, credits).

    Name matching uses normalized comparison (strip accents, lowercase) so that
    "Timothée Chalamet" in oscars.csv matches "Timothee Chalamet" elsewhere.

    Args:
        category: Oscar category
        oscar_df: Full oscars.csv DataFrame (all ceremonies)
        nominees: List of nominees from Stage 1
        tmdb_fetcher: Optional TMDb fetcher for person enrichment. If None,
            TMDb fields (birth_date, tmdb_popularity, total_film_credits) will be None.

    Returns:
        Dict mapping (person_name, film_id, ceremony) -> PersonData
    """
    result: dict[tuple[str, str, int], PersonData] = {}

    # Pre-compute normalized names in oscar_df for efficient matching
    oscar_df = oscar_df.copy()
    oscar_df["_name_norm"] = oscar_df["Name"].fillna("").apply(normalize_person_name)

    # Track unique persons to avoid redundant TMDb lookups
    # Key: normalized name, Value: TMDb person result (or None)
    tmdb_cache: dict[str, object] = {}
    _SENTINEL = object()  # Distinguish "not looked up" from "looked up, got None"

    # TMDb lookups are made sequentially to respect rate limits. Since results are
    # cached via diskcache, repeated calls for the same person are free. For fresh
    # builds, this may be slow (~30s per category with ~135 unique persons).
    # TODO: consider async+rate-limit if rebuild time becomes a bottleneck.
    total = len([n for n in nominees if n.nominee_name is not None])
    count = 0

    for nominee in nominees:
        if nominee.nominee_name is None:
            continue

        count += 1
        name = nominee.nominee_name
        name_norm = normalize_person_name(name)
        ceremony = nominee.ceremony

        # Count prior noms/wins using normalized matching (ceremonies < current)
        person_history = oscar_df[
            (oscar_df["_name_norm"] == name_norm) & (oscar_df["Ceremony"] < ceremony)
        ]
        same_cat_history = person_history[person_history["CanonicalCategory"] == category.value]

        # TMDb enrichment
        birth_date = None
        tmdb_popularity = None
        total_film_credits = None

        if tmdb_fetcher is not None:
            if name_norm not in tmdb_cache:
                print(f"  [{count}/{total}] TMDb lookup: {name}")
                tmdb_person = tmdb_fetcher.fetch_person_data(name, known_film_title=nominee.title)
                tmdb_cache[name_norm] = tmdb_person if tmdb_person is not None else _SENTINEL
            else:
                tmdb_person_or_sentinel = tmdb_cache[name_norm]
                tmdb_person = (
                    None if tmdb_person_or_sentinel is _SENTINEL else tmdb_person_or_sentinel  # type: ignore[assignment]  # sentinel pattern: object vs TMDbPersonResult|None
                )

            if tmdb_person is not None:
                birth_date = tmdb_person.birth_date  # type: ignore[union-attr]
                tmdb_popularity = tmdb_person.tmdb_popularity  # type: ignore[union-attr]
                total_film_credits = tmdb_person.total_film_credits  # type: ignore[union-attr]

        result[(name, nominee.film_id, ceremony)] = PersonData(
            name=name,
            prev_noms_same_category=len(same_cat_history),
            prev_noms_any_category=len(person_history),
            prev_wins_same_category=int(same_cat_history["Winner"].sum()),
            prev_wins_any_category=int(person_history["Winner"].sum()),
            birth_date=birth_date,
            tmdb_popularity=tmdb_popularity,
            total_film_credits=total_film_credits,
        )

    return result


# ============================================================================
# Stage 5: Merge
# ============================================================================


def get_award_announcement_status(
    award: PrecursorAward,
    film_year: int,
    as_of_date: date | None,
    calendar: AwardsCalendar | None,
) -> tuple[bool, bool]:
    """Determine if nominees and winner have been announced for an award.

    Args:
        award: The precursor award organization
        film_year: The film's release year (e.g., 2025 for 2026 ceremony)
        as_of_date: Date we're generating dataset for. None = all awards announced.
        calendar: Awards calendar for current ceremony year.

    Returns:
        (nominees_announced, winner_announced) tuple of bools
    """
    # If no date specified or no calendar, treat all awards as announced
    if as_of_date is None or calendar is None:
        return True, True

    # Check if this is current season (film_year matches calendar's ceremony year - 1)
    current_season_film_year = calendar.ceremony_year - 1
    if film_year != current_season_film_year:
        # Historical year - all awards announced
        return True, True

    # Current season - check calendar dates
    nominees_announced = False
    winner_announced = False

    # Check nomination date
    nom_key = (AwardOrg(award.value), EventPhase.NOMINATION)
    if nom_key in calendar.events:
        nom_date = calendar.get_local_date(AwardOrg(award.value), EventPhase.NOMINATION)
        nominees_announced = as_of_date >= nom_date

    # Check winner date
    win_key = (AwardOrg(award.value), EventPhase.WINNER)
    if win_key in calendar.events:
        win_date = calendar.get_local_date(AwardOrg(award.value), EventPhase.WINNER)
        winner_announced = as_of_date >= win_date

    return nominees_announced, winner_announced


def match_precursor_awards(
    film_title: str,
    film_year: int,
    precursor_awards: PrecursorAwardsFile,
    fetcher: PrecursorAwardsFetcher,
    precursor_mapping: list[PrecursorSpec],
    nominee_name: str | None = None,
    as_of_date: date | None = None,
    calendar: AwardsCalendar | None = None,
) -> dict[str, AwardResult]:
    """Match a nominee to precursor awards and return results.

    Supports two matching modes depending on the precursor award data:
    - **Film-level** (e.g., PGA BP, BAFTA Film): Matches by film title only.
    - **Person-level** (e.g., SAG Lead Actor, BAFTA Director): Matches by
      (person_name, film_title) pair. Falls back to film-only if person match fails.

    The mode is auto-detected from the data: if precursor records have a ``person``
    field, person-level matching is used.

    **Duplicate (year, film) handling:** Supporting actor/actress categories often
    have multiple nominees from the same film (e.g., 2019 "The Irishman" has both
    De Niro and Pesci nominated for SAG Supporting Actor). This is intentional —
    the matching finds the correct person-level record. For film-level awards,
    duplicates are rare and the first match is returned.

    **Annie Award bloat:** Annie Awards sometimes nominate many films (23 in 2025).
    No cap is applied — matching only succeeds against actual Oscar nominees, so
    extra entries are naturally filtered out.

    AwardResult semantics:
    - None: Data not yet available (award not announced, or no data for this year)
    - True: Nominee won/was nominated for this precursor
    - False: Nominee did not win/was not nominated (award has been announced)

    Args:
        film_title: Title of the nominee's film
        film_year: Film release year (e.g., 2025)
        precursor_awards: Loaded precursor awards data
        fetcher: PrecursorAwardsFetcher for fuzzy matching
        precursor_mapping: List of PrecursorSpec from CATEGORY_PRECURSORS
        nominee_name: Person name for person-level categories. None for film-level.
        as_of_date: Date for which to generate data. None = all awards available.
        calendar: Awards calendar for date-based filtering.
    """
    result: dict[str, AwardResult] = {}

    for spec in precursor_mapping:
        if spec.key not in precursor_awards.awards:
            continue

        # Get announcement status
        nominees_announced, winner_announced = get_award_announcement_status(
            spec.award, film_year, as_of_date, calendar
        )

        # Filter to ceremony year
        year_records = [
            r for r in precursor_awards.awards[spec.key] if r.year_ceremony == film_year
        ]

        if not year_records:
            # No data for this year - data gap, keep as None
            continue

        # Determine if this is a person-level award (records have person names)
        is_person_award = any(r.person is not None for r in year_records)

        matched_record = None

        if is_person_award and nominee_name:
            # Person-level matching: match by (person_name, film_title)
            matched_record = _match_person_precursor(
                nominee_name, film_title, year_records, fetcher
            )
        else:
            # Film-level matching: match by film title only
            matched_record = _match_film_precursor(film_title, year_records, fetcher)

        award_result = AwardResult()

        if matched_record is not None:
            award_result.nominee = True
            if winner_announced:
                award_result.winner = matched_record.is_winner
        elif nominees_announced:
            award_result.nominee = False
            if winner_announced:
                award_result.winner = False
        # else: nominees not announced, keep both as None (default)

        # Only store keys where at least one field was set
        if award_result.nominee is not None or award_result.winner is not None:
            result[spec.key] = award_result

    return result


def _match_film_precursor(
    film_title: str,
    year_records: list[PrecursorAwardRecord],
    fetcher: PrecursorAwardsFetcher,
) -> PrecursorAwardRecord | None:
    """Match a film to precursor records by title (fuzzy).

    Returns the matched record or None.
    """
    film_list = [r.film for r in year_records]
    match, _score = fetcher.match_film(film_title, film_list)
    if match:
        return next(r for r in year_records if r.film == match)
    return None


def _match_person_precursor(
    nominee_name: str,
    film_title: str,
    year_records: list[PrecursorAwardRecord],
    fetcher: PrecursorAwardsFetcher,
) -> PrecursorAwardRecord | None:
    """Match a person+film to person-level precursor records.

    Strategy:
    1. Normalize nominee_name and all record person names
    2. Find records where normalized person name matches
    3. Among those, verify film title also matches (fuzzy)
    4. Fallback: if no person match, try film-only match (handles name mismatches)

    Returns the matched record or None.
    """
    norm_name = normalize_person_name(nominee_name)

    # Strategy 1: Match by person name first
    person_matches = [
        r
        for r in year_records
        if r.person is not None and normalize_person_name(r.person) == norm_name
    ]

    if person_matches:
        if len(person_matches) == 1:
            return person_matches[0]
        # Multiple records for same person (rare but possible) — verify film
        for r in person_matches:
            film_list = [r.film]
            match, _score = fetcher.match_film(film_title, film_list)
            if match:
                return r
        # Person found but film didn't match — still return first (person match is strong)
        return person_matches[0]

    # Strategy 2: Fallback to film-only matching
    # This handles cases where names differ between Oscar and precursor data
    return _match_film_precursor(film_title, year_records, fetcher)


def build_merged_dataset(
    category: OscarCategory,
    year_start: int,
    year_end: int,
    input_dir: Path,
    as_of_date: date | None = None,
    shared_dir: Path | None = None,
) -> NominationDataset:
    """Stage 5: Merge all intermediate files into raw dataset.

    This produces the RAW dataset with composed records from all sources.
    No derived features are computed here - that happens in feature_engineering.py.

    Args:
        category: Oscar category to build dataset for
        year_start: Start ceremony year (e.g., 2000)
        year_end: End ceremony year (e.g., 2026)
        input_dir: Directory containing oscar_nominations.json (category-specific)
        as_of_date: Date for current season filtering. None = all awards available.
        shared_dir: Directory containing shared film_metadata.json and
            precursor_awards.json. When None, these are read from input_dir
            (single-dir mode). Use shared_dir when building all categories together
            to avoid duplicating large intermediate files per category.
    """
    # Intermediate file sources: oscar_nominations is category-specific;
    # metadata and precursors may come from a shared directory.
    data_dir = shared_dir if shared_dir is not None else input_dir
    oscar_path = input_dir / "oscar_nominations.json"
    metadata_path = data_dir / "film_metadata.json"
    precursor_path = data_dir / "precursor_awards.json"

    for path in [oscar_path, metadata_path, precursor_path]:
        if not path.exists():
            raise FileNotFoundError(f"Run previous stages first: {path}")

    with open(oscar_path) as f:
        oscar_data = OscarNominationsFile(**json.load(f))
    with open(metadata_path) as f:
        metadata_data = FilmMetadataFile(**json.load(f))
    with open(precursor_path) as f:
        precursor_data = PrecursorAwardsFile(**json.load(f))

    # Get calendar for current season if date specified
    calendar = CALENDARS.get(year_end) if as_of_date else None
    if as_of_date:
        print(f"Generating dataset as of {as_of_date.isoformat()}")
        if calendar:
            print(f"Using awards calendar for ceremony year {year_end}")
        else:
            print(f"Warning: No calendar for ceremony year {year_end}, treating all as announced")

    # Build person data if category is person-level
    person_data: dict[tuple[str, str, int], PersonData] = {}
    if category in PERSON_CATEGORIES:
        print("Building person career data from oscars.csv + TMDb enrichment...")
        max_ceremony = year_to_ceremony(year_end)
        full_df = load_oscar_data(min_ceremony=1, max_ceremony=max_ceremony)
        tmdb_fetcher = TMDbFetcher(api_key=TMDB_API_KEY)
        person_data = build_person_data(category, full_df, oscar_data.records, tmdb_fetcher)
        print(f"  Built PersonData for {len(person_data)} nominees")

    # Get precursor mapping for this category
    precursor_mapping = get_precursor_specs(category)

    fetcher = PrecursorAwardsFetcher()
    records = []
    total = len(oscar_data.records)

    for i, nominee in enumerate(oscar_data.records):
        print(f"[{i + 1}/{total}] Merging {nominee.title}")

        film_id = nominee.film_id
        metadata = metadata_data.records.get(film_id)
        film_year = nominee.year_film

        # Build FilmData
        film = FilmData(
            film_id=film_id,
            title=nominee.title,
            metadata=metadata,
            oscar_noms=OscarNominationInfo(
                oscar_total_nominations=nominee.oscar_total_nominations,
                oscar_total_wins=nominee.oscar_total_wins,
                oscar_nominations_by_category=nominee.oscar_nominations_by_category,
                oscar_wins_list=nominee.oscar_wins_list,
            ),
        )

        # Get person data if available
        person: PersonData | None = None
        if nominee.nominee_name:
            person = person_data.get((nominee.nominee_name, nominee.film_id, nominee.ceremony))

        # Precursor awards with date-based filtering
        precursor_result = match_precursor_awards(
            nominee.title,
            film_year,
            precursor_data,
            fetcher,
            precursor_mapping,
            nominee_name=nominee.nominee_name,
            as_of_date=as_of_date,
            calendar=calendar,
        )

        # Create composed record
        record = NominationRecord(
            category=category,
            ceremony=nominee.ceremony,
            year_film=nominee.year_film,
            category_winner=nominee.category_winner,
            nominee_name=nominee.nominee_name,
            film=film,
            person=person,
            precursors=precursor_result,
        )
        records.append(record)

    return NominationDataset(
        category=category,
        year_start=year_start,
        year_end=year_end,
        record_count=len(records),
        records=records,
    )


# ============================================================================
# Save/Load Helpers
# ============================================================================


def save_json(
    data: OscarNominationsFile | FilmMetadataFile | PrecursorAwardsFile | NominationDataset,
    filename: str,
    output_dir: Path,
) -> Path:
    """Save Pydantic model to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        f.write(data.model_dump_json(indent=2))
    print(f"Saved: {path}")
    return path


def get_output_filename(category: OscarCategory) -> str:
    """Get output filename for a category's raw dataset.

    Converts category enum to a snake_case filename.
    E.g., OscarCategory.BEST_PICTURE -> "oscar_best_picture_raw.json"
    """
    cat_slug = category.slug
    return f"oscar_{cat_slug}_raw.json"


# ============================================================================
# Summary Helpers
# ============================================================================


def print_dataset_summary(data: NominationDataset) -> None:
    """Print summary of the raw dataset including data completeness."""
    print("\n" + "=" * 60)
    print(f"Raw Dataset Summary — {data.category.value}")
    print("=" * 60)

    winners = [r for r in data.records if r.category_winner]
    print(f"Total nominees: {len(data.records)}")
    print(f"Winners: {len(winners)}")
    print(f"Non-winners: {len(data.records) - len(winners)}")

    # Data completeness
    print("\nData completeness (showing non-100% fields only):")
    incomplete_found = False

    # Metadata fields
    metadata_fields = [
        "metacritic",
        "rotten_tomatoes",
        "imdb_rating",
        "box_office_domestic",
        "budget",
    ]
    for field in metadata_fields:
        non_null = sum(
            1
            for r in data.records
            if r.film.metadata and getattr(r.film.metadata, field) is not None
        )
        pct = 100 * non_null / len(data.records)
        if pct < 100:
            print(f"  metadata.{field}: {non_null}/{len(data.records)} ({pct:.1f}%)")
            incomplete_found = True

    # Precursor fields (check only the ones relevant to the category's mapping)
    precursor_mapping = get_precursor_specs(data.category)
    for spec in precursor_mapping:
        for suffix in ["nominee", "winner"]:
            non_null = sum(
                1
                for r in data.records
                if getattr(r.precursors.get(spec.key, AwardResult()), suffix) is not None
            )
            pct = 100 * non_null / len(data.records)
            if pct < 100:
                print(
                    f"  precursors.{spec.key}.{suffix}: {non_null}/{len(data.records)} ({pct:.1f}%)"
                )
                incomplete_found = True

    if not incomplete_found:
        print("  All fields are 100% complete.")

    print("\n[INFO] Run feature_engineering.py to generate model-ready features.")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Oscar dataset for any category")
    parser.add_argument(
        "--mode",
        choices=["oscar", "metadata", "precursors", "merge", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="BEST_PICTURE",
        help="Oscar category (enum name, e.g. BEST_PICTURE, ACTOR_LEADING)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2000,
        help="Start ceremony year (default: 2000)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2026,
        help="End ceremony year (default: 2026)",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Generate dataset as of this date (YYYY-MM-DD). For current season filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory with oscar_nominations.json (for merge stage). Defaults to --output-dir.",
    )
    parser.add_argument(
        "--shared-dir",
        type=str,
        default=None,
        help=(
            "Shared directory with film_metadata.json and precursor_awards.json. "
            "When provided, metadata/precursor stages write here instead of --output-dir, "
            "and merge reads from here. Useful when building all categories to avoid "
            "duplicating large intermediate files."
        ),
    )
    parser.add_argument(
        "--extra-input-dirs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Additional directories containing oscar_nominations.json to include in "
            "metadata build (--mode metadata). Allows building a single shared "
            "film_metadata.json covering films from multiple categories."
        ),
    )
    args = parser.parse_args()

    # Parse category
    category = OscarCategory[args.category]

    # Parse as_of_date
    as_of_date: date | None = None
    if args.as_of_date:
        as_of_date = date.fromisoformat(args.as_of_date)

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir) if args.input_dir else output_dir
    shared_dir = Path(args.shared_dir) if args.shared_dir else None
    # Effective directories for metadata/precursors: prefer shared_dir, fall back to output_dir
    data_output_dir = shared_dir if shared_dir is not None else output_dir

    print("=" * 60)
    print(f"Oscar Raw Dataset Builder — {category.value}")
    print(f"Mode: {args.mode} | Years: {args.year_start}-{args.year_end}")
    print(f"Output dir: {output_dir}")
    if input_dir != output_dir:
        print(f"Input dir: {input_dir}")
    if shared_dir:
        print(f"Shared dir: {shared_dir}")
    if as_of_date:
        print(f"As of date: {as_of_date.isoformat()}")
    print("=" * 60)

    if args.mode in ("oscar", "all"):
        print("\n[Stage 1/5] Building Oscar nominations...")
        oscar_data = build_oscar_nominations(category, args.year_start, args.year_end)
        save_json(oscar_data, "oscar_nominations.json", output_dir)

    if args.mode in ("metadata", "all"):
        print("\n[Stage 2/5] Fetching film metadata...")
        extra_dirs = [Path(d) for d in (args.extra_input_dirs or [])]
        all_input_dirs: list[Path] = [input_dir] + extra_dirs
        metadata_data = build_film_metadata(args.year_start, args.year_end, all_input_dirs)
        save_json(metadata_data, "film_metadata.json", data_output_dir)

    if args.mode in ("precursors", "all"):
        print("\n[Stage 3/5] Fetching precursor awards...")
        precursor_data = build_precursor_awards(args.year_start, args.year_end)
        save_json(precursor_data, "precursor_awards.json", data_output_dir)

    if args.mode in ("merge", "all"):
        print(f"\n[Stage 5/5] Merging into raw dataset for {category.value}...")
        merged_data = build_merged_dataset(
            category, args.year_start, args.year_end, input_dir, as_of_date, shared_dir
        )
        save_json(merged_data, get_output_filename(category), output_dir)
        print_dataset_summary(merged_data)


if __name__ == "__main__":
    main()
