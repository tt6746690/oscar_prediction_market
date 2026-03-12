"""Data schemas for Oscar prediction dataset.

Pydantic models ensure type safety and provide validation.

Intermediate file schemas (per-source):
- OscarNominationsFile: Oscar nominations from oscars.csv
- FilmMetadataFile: OMDb + TMDb metadata
- PrecursorAwardsFile: Guild awards data

Final dataset schema (merged, no derived features):
- NominationDataset: Composed of FilmData + PersonData + dict[str, AwardResult]
"""

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field

# ============================================================================
# Enums
# ============================================================================


class OscarCategory(StrEnum):
    """Official Oscar award categories (as they appear in oscars.csv)."""

    ACTOR_LEADING = "ACTOR IN A LEADING ROLE"
    ACTOR_SUPPORTING = "ACTOR IN A SUPPORTING ROLE"
    ACTRESS_LEADING = "ACTRESS IN A LEADING ROLE"
    ACTRESS_SUPPORTING = "ACTRESS IN A SUPPORTING ROLE"
    ANIMATED_FEATURE = "ANIMATED FEATURE FILM"
    ART_DIRECTION = "ART DIRECTION"
    BEST_PICTURE = "BEST PICTURE"
    CASTING = "CASTING"
    CINEMATOGRAPHY = "CINEMATOGRAPHY"
    COSTUME_DESIGN = "COSTUME DESIGN"
    DIRECTING = "DIRECTING"
    FILM_EDITING = "FILM EDITING"
    INTERNATIONAL_FEATURE = "INTERNATIONAL FEATURE FILM"
    MAKEUP = "MAKEUP AND HAIRSTYLING"
    ORIGINAL_SCORE = "MUSIC (Original Score)"
    ORIGINAL_SONG = "MUSIC (Original Song)"
    SOUND = "SOUND MIXING"
    VISUAL_EFFECTS = "VISUAL EFFECTS"
    ADAPTED_SCREENPLAY = "WRITING (Adapted Screenplay)"
    ORIGINAL_SCREENPLAY = "WRITING (Original Screenplay)"

    @property
    def slug(self) -> str:
        """Lowercase name for use in file paths and labels (e.g., 'best_picture')."""
        return self.name.lower()

    @classmethod
    def from_slug(cls, slug: str) -> Self:
        """Look up category from its slug (e.g., 'best_picture' → BEST_PICTURE)."""
        return cls[slug.upper()]


class PrecursorAward(StrEnum):
    """Precursor award organizations.

    Each organization announces nominations and winners on a specific date.
    Used for calendar-based feature availability filtering.
    """

    PGA = "pga"  # Producers Guild of America
    DGA = "dga"  # Directors Guild of America
    SAG = "sag"  # Screen Actors Guild
    BAFTA = "bafta"  # British Academy Film Awards
    GOLDEN_GLOBE = "golden_globe"  # Golden Globe Awards
    CRITICS_CHOICE = "critics_choice"  # Critics Choice Awards
    WGA = "wga"  # Writers Guild of America
    ASC = "asc"  # American Society of Cinematographers
    ANNIE = "annie"  # Annie Awards (animation)


class PrecursorKey(StrEnum):
    """Precursor award keys — universal identifier bridging fetch → storage → features.

    Each key maps to a specific sub-award from a precursor organization. The string
    value is used as:
    - Key in ``PrecursorAwardsFile.awards`` (fetch/cache layer)
    - Key in ``NominationRecord.precursors`` (storage layer)
    - Feature name prefix in feature_engineering.py (``{key}_winner``, ``{key}_nominee``)

    Grouped by organization. Multiple keys per org exist when an organization has
    category-specific or genre-split sub-awards (e.g., Golden Globe Drama vs Musical).
    """

    # --- PGA (Producers Guild of America) ---
    PGA_BP = "pga_bp"
    PGA_ANIMATED = "pga_animated"

    # --- DGA (Directors Guild of America) ---
    DGA_DIRECTING = "dga_directing"

    # --- SAG (Screen Actors Guild) ---
    SAG_ENSEMBLE = "sag_ensemble"
    SAG_LEAD_ACTOR = "sag_lead_actor"
    SAG_LEAD_ACTRESS = "sag_lead_actress"
    SAG_SUPPORTING_ACTOR = "sag_supporting_actor"
    SAG_SUPPORTING_ACTRESS = "sag_supporting_actress"

    # --- BAFTA (British Academy Film Awards) ---
    BAFTA_FILM = "bafta_film"
    BAFTA_DIRECTOR = "bafta_director"
    BAFTA_LEAD_ACTOR = "bafta_lead_actor"
    BAFTA_LEAD_ACTRESS = "bafta_lead_actress"
    BAFTA_SUPPORTING_ACTOR = "bafta_supporting_actor"
    BAFTA_SUPPORTING_ACTRESS = "bafta_supporting_actress"
    BAFTA_ORIGINAL_SCREENPLAY = "bafta_original_screenplay"
    BAFTA_CINEMATOGRAPHY = "bafta_cinematography"
    BAFTA_ANIMATED = "bafta_animated"

    # --- Golden Globe ---
    GOLDEN_GLOBE_DRAMA = "golden_globe_drama"
    GOLDEN_GLOBE_MUSICAL = "golden_globe_musical"
    GOLDEN_GLOBE_DIRECTOR = "golden_globe_director"
    GOLDEN_GLOBE_ACTOR_DRAMA = "golden_globe_actor_drama"
    GOLDEN_GLOBE_ACTOR_MUSICAL = "golden_globe_actor_musical"
    GOLDEN_GLOBE_ACTRESS_DRAMA = "golden_globe_actress_drama"
    GOLDEN_GLOBE_ACTRESS_MUSICAL = "golden_globe_actress_musical"
    GOLDEN_GLOBE_SUPPORTING_ACTOR = "golden_globe_supporting_actor"
    GOLDEN_GLOBE_SUPPORTING_ACTRESS = "golden_globe_supporting_actress"
    GOLDEN_GLOBE_SCREENPLAY = "golden_globe_screenplay"
    GOLDEN_GLOBE_ANIMATED = "golden_globe_animated"

    # --- Critics Choice ---
    CRITICS_CHOICE_PICTURE = "critics_choice_picture"
    CRITICS_CHOICE_DIRECTOR = "critics_choice_director"
    CRITICS_CHOICE_ACTOR = "critics_choice_actor"
    CRITICS_CHOICE_ACTRESS = "critics_choice_actress"
    CRITICS_CHOICE_SUPPORTING_ACTOR = "critics_choice_supporting_actor"
    CRITICS_CHOICE_SUPPORTING_ACTRESS = "critics_choice_supporting_actress"
    CRITICS_CHOICE_ORIGINAL_SCREENPLAY = "critics_choice_original_screenplay"
    CRITICS_CHOICE_CINEMATOGRAPHY = "critics_choice_cinematography"
    CRITICS_CHOICE_ANIMATED = "critics_choice_animated"

    # --- WGA (Writers Guild of America) ---
    WGA_ORIGINAL = "wga_original"

    # --- ASC (American Society of Cinematographers) ---
    ASC_CINEMATOGRAPHY = "asc_cinematography"

    # --- Annie Awards (animation) ---
    ANNIE_FEATURE = "annie_feature"


class Genre(StrEnum):
    """Film genres (from OMDb/TMDb). Ordered roughly by frequency in Best Picture nominees."""

    DRAMA = "Drama"
    BIOGRAPHY = "Biography"
    HISTORY = "History"
    WAR = "War"
    ROMANCE = "Romance"
    THRILLER = "Thriller"
    CRIME = "Crime"
    MUSIC = "Music"
    MUSICAL = "Musical"
    COMEDY = "Comedy"
    ADVENTURE = "Adventure"
    ACTION = "Action"
    MYSTERY = "Mystery"
    FANTASY = "Fantasy"
    SCI_FI = "Sci-Fi"
    FAMILY = "Family"
    HORROR = "Horror"
    WESTERN = "Western"
    ANIMATION = "Animation"
    SPORT = "Sport"
    DOCUMENTARY = "Documentary"
    OTHER = "Other"


# ============================================================================
# Category Classification
# ============================================================================

#: Categories where the nominee is a film (no person data needed)
FILM_CATEGORIES: frozenset[OscarCategory] = frozenset(
    {
        OscarCategory.BEST_PICTURE,
        OscarCategory.ANIMATED_FEATURE,
    }
)

#: Categories where the nominee is a person (need PersonData)
PERSON_CATEGORIES: frozenset[OscarCategory] = frozenset(
    {
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ACTOR_SUPPORTING,
        OscarCategory.ACTRESS_SUPPORTING,
        OscarCategory.DIRECTING,
        OscarCategory.ORIGINAL_SCREENPLAY,
        OscarCategory.CINEMATOGRAPHY,
    }
)


# ============================================================================
# Intermediate: Oscar Nominations (from oscars.csv)
# ============================================================================


class OscarNominee(BaseModel):
    """Oscar nominee extracted from oscars.csv (intermediate, pre-enrichment).

    Generic across all categories. For BP, nominee_name is None.
    For acting/directing, nominee_name is the person's name.
    """

    model_config = {"extra": "forbid"}

    film_id: str = Field(..., description="IMDb ID (tt...)")
    title: str = Field(..., description="Film title")
    ceremony: int = Field(..., description="Oscar ceremony number")
    year_film: int = Field(..., description="Film release year", ge=1900, le=2100)
    category: OscarCategory = Field(..., description="Oscar category")
    category_winner: bool = Field(..., description="Won this category?")
    nominee_name: str | None = Field(None, description="Person name (for person-level categories)")

    # Raw Oscar data (film-level, shared across categories for same film)
    oscar_total_nominations: int = Field(..., description="Total Oscar nominations")
    oscar_total_wins: int = Field(..., description="Total Oscar wins")
    oscar_nominations_by_category: dict[str, int] = Field(
        ..., description="Nominations per category"
    )
    oscar_wins_list: list[str] = Field(..., description="Categories won")


class OscarNominationsFile(BaseModel):
    """Intermediate file: Oscar nominations data for a single category."""

    model_config = {"extra": "forbid"}

    category: OscarCategory = Field(..., description="Oscar category")
    year_start: int = Field(..., description="Start ceremony year")
    year_end: int = Field(..., description="End ceremony year")
    record_count: int = Field(..., description="Number of nominees")
    records: list[OscarNominee] = Field(..., description="Nominees")


# ============================================================================
# Intermediate: Film Metadata (OMDb + TMDb)
# ============================================================================


class FilmMetadata(BaseModel):
    """Combined OMDb + TMDb metadata for a film."""

    model_config = {"extra": "forbid"}

    film_id: str = Field(..., description="IMDb ID (tt...)")
    title: str | None = Field(None, description="Film title")

    # Basic info (OMDb)
    rated: str | None = Field(None, description="MPAA rating")
    released: date | None = Field(None, description="Release date")
    runtime_minutes: int | None = Field(None, description="Runtime in minutes")
    genres: list[str] = Field(default_factory=list, description="Film genres")
    director: str | None = Field(None, description="Director name(s)")
    actors: list[str] = Field(default_factory=list, description="Top billed actors")
    language: str | None = Field(None, description="Languages")
    country: str | None = Field(None, description="Production countries")

    # Ratings (OMDb) - KEY PREDICTORS
    imdb_rating: float | None = Field(None, description="IMDb rating (0-10)", ge=0, le=10)
    imdb_votes: int | None = Field(None, description="IMDb vote count", ge=0)
    rotten_tomatoes: int | None = Field(
        None, description="Rotten Tomatoes score (0-100)", ge=0, le=100
    )
    metacritic: int | None = Field(None, description="Metacritic score (0-100)", ge=0, le=100)

    # Financials (OMDb + TMDb)
    box_office_domestic: int | None = Field(None, description="US box office ($)", ge=0)
    box_office_worldwide: int | None = Field(None, description="Worldwide box office ($)", ge=0)
    budget: int | None = Field(None, description="Production budget ($)", ge=0)

    # Awards summary (OMDb) — None means data unavailable (vs 0 = genuinely zero)
    total_awards_wins: int | None = Field(None, description="Total awards wins", ge=0)
    total_awards_nominations: int | None = Field(None, description="Total awards nominations", ge=0)

    # Production (TMDb)
    production_companies: list[str] = Field(default_factory=list, description="Production cos")


class FilmMetadataFile(BaseModel):
    """Intermediate file: Film metadata from OMDb + TMDb."""

    model_config = {"extra": "forbid"}

    year_start: int = Field(..., description="Start ceremony year")
    year_end: int = Field(..., description="End ceremony year")
    record_count: int = Field(..., description="Number of films")
    records: dict[str, FilmMetadata] = Field(..., description="Film ID -> metadata")


# ============================================================================
# Intermediate: Precursor Awards
# ============================================================================


class PrecursorAwardRecord(BaseModel):
    """Single precursor award record from Wikipedia scraping."""

    model_config = {"extra": "forbid"}

    year_ceremony: int = Field(..., description="Ceremony year (e.g., 2026)")
    film: str = Field(..., description="Film title")
    person: str | None = Field(None, description="Person name (for acting/directing awards)")
    is_winner: bool = Field(..., description="Won this award?")


class PrecursorAwardsFile(BaseModel):
    """Intermediate file: Precursor awards data."""

    model_config = {"extra": "forbid"}

    year_start: int = Field(..., description="Start ceremony year")
    year_end: int = Field(..., description="End ceremony year")
    awards: dict[str, list[PrecursorAwardRecord]] = Field(
        ...,
        description="Award key -> records (pga, dga, sag, bafta, golden_globe_drama, etc.)",
    )


# ============================================================================
# Core: Film's Oscar Profile
# ============================================================================


class OscarNominationInfo(BaseModel):
    """Oscar nomination/win data for a film, extracted from oscars.csv.

    Shared across all categories — a film's total Oscar profile.
    """

    model_config = {"extra": "forbid"}

    oscar_total_nominations: int = Field(..., description="Total Oscar nominations")
    oscar_total_wins: int = Field(..., description="Total Oscar wins")
    oscar_nominations_by_category: dict[str, int] = Field(
        ..., description="Nominations per category"
    )
    oscar_wins_list: list[str] = Field(..., description="Categories won")


# ============================================================================
# Core: Film Data
# ============================================================================


class FilmData(BaseModel):
    """All data about a film. Collected once, shared across categories.

    A film like "Sinners" appears in BP, Director, Actor, Screenplay.
    FilmData is fetched/assembled once per film_id and reused.
    """

    model_config = {"extra": "forbid"}

    film_id: str = Field(..., description="IMDb ID (tt...)")
    title: str = Field(..., description="Film title")

    metadata: FilmMetadata | None = Field(None, description="OMDb + TMDb metadata")
    oscar_noms: OscarNominationInfo = Field(..., description="Cross-category Oscar profile")


# ============================================================================
# Core: Person Data
# ============================================================================


class PersonData(BaseModel):
    """Career data for a nominated person (actor, director, writer, DP).

    Built from oscars.csv (career history) + optionally TMDb person API
    (birthdate, filmography, popularity).

    For BP and Animated, person data is not applicable — NominationRecord.person
    will be None.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(..., description="Person name")
    # Career history (from oscars.csv, computed at dataset build time)
    prev_noms_same_category: int = Field(..., description="Prior noms in the SAME Oscar category")
    prev_noms_any_category: int = Field(..., description="Prior noms in ANY Oscar category")
    prev_wins_same_category: int = Field(..., description="Prior wins in same category")
    prev_wins_any_category: int = Field(..., description="Prior wins in any category")

    # TMDb person API enrichment (Phase 1)
    birth_date: date | None = None  # For computing age at ceremony
    tmdb_popularity: float | None = None  # TMDb popularity score
    total_film_credits: int | None = None  # Filmography size


# ============================================================================
# Core: Precursor Award Result
# ============================================================================


class AwardResult(BaseModel):
    """Result of a single precursor award for a nominee.

    Stored in NominationRecord.precursors as a flat dict[str, AwardResult],
    keyed by PrecursorSpec.key (e.g., "pga_bp", "sag_lead_actor").

    Only populated keys are stored — no null bloat from inapplicable awards.
    Missing key = award not mapped for this category.

    Fields are None when:
    - Award not yet announced (current season, incremental availability)
    - Award didn't exist for that ceremony year

    Fields are True/False when the award has been announced for that year.
    """

    model_config = {"extra": "forbid"}

    nominee: bool | None = Field(default=None, description="Nominated for this award")
    winner: bool | None = Field(default=None, description="Won this award")


# ============================================================================
# Core: Nomination Record (one per nomination)
# ============================================================================


class NominationRecord(BaseModel):
    """One row = one nomination in one category for one ceremony.

    This is the prediction unit. For Best Picture, the nominee is a film. For
    Best Actor, the nominee is a person (performing in a film). Both share film
    data and precursor data; person data is None for film-level categories.

    Precursors are stored as a flat dict keyed by PrecursorSpec.key. Only keys
    relevant to this category are present (e.g., BP has ~7 keys, not ~30).

    Examples:
        NominationRecord for BP: film=FilmData("Sinners"), person=None,
            precursors={"pga_bp": AwardResult(nominee=True, winner=True), ...}
        NominationRecord for Actor: film=FilmData("Sinners"),
            person=PersonData("Michael B. Jordan"),
            precursors={"sag_lead_actor": AwardResult(...), ...}
    """

    model_config = {"extra": "forbid"}

    category: OscarCategory = Field(..., description="Oscar category")
    ceremony: int = Field(..., description="Oscar ceremony number")
    year_film: int = Field(..., description="Film release year")
    category_winner: bool = Field(..., description="Won this category?")
    nominee_name: str | None = Field(None, description="Person name for person-level categories")

    film: FilmData = Field(..., description="Film data (metadata + Oscar profile)")
    person: PersonData | None = Field(None, description="Person data (career + enrichment)")
    precursors: dict[str, AwardResult] = Field(
        default_factory=dict,
        description="Precursor award results, keyed by PrecursorSpec.key",
    )


class NominationDataset(BaseModel):
    """Dataset of nominations for a single category, ready for feature engineering.

    This is the validated in-memory representation used after the one-off
    dataset builders have finished their work. The trading code reuses it
    not because it needs every feature-engineering field, but because it is
    the authoritative Oscar-side namespace for a category/year snapshot.
    That makes it the safest place to derive title/person/title-film remaps
    instead of rebuilding those from ad hoc JSON dicts in trading code.
    """

    model_config = {"extra": "forbid"}

    category: OscarCategory = Field(..., description="Oscar category")
    year_start: int = Field(..., description="Start ceremony year")
    year_end: int = Field(..., description="End ceremony year")
    record_count: int = Field(..., description="Number of nominees")
    records: list[NominationRecord] = Field(..., description="All nominees")

    def records_for_ceremony_year(self, ceremony_year: int) -> list[NominationRecord]:
        """Return records for a specific ceremony year.

        The dataset may span a range of ceremony years for modeling, but the
        trading pipeline always needs one concrete Oscar season when building
        Kalshi remaps. This helper makes that seasonal slice explicit.
        """
        ceremony_number = ceremony_year - 1928
        return [record for record in self.records if record.ceremony == ceremony_number]

    def build_title_to_person_map(self, ceremony_year: int) -> dict[str, str]:
        """Build ``{film_title: person_name}`` for acting/directing-style categories.

        Model outputs for some categories are keyed by film title, while
        Kalshi markets are keyed by person name. Trading needs to reconcile
        those namespaces before matching to market outcomes, and this method
        centralizes that conversion on the validated Oscar dataset.
        """
        result: dict[str, str] = {}
        for record in self.records_for_ceremony_year(ceremony_year):
            title = record.film.title
            person = record.nominee_name
            if title and person:
                result[title] = person
        return result

    def build_nominee_to_film_title_map(self, ceremony_year: int) -> dict[str, str]:
        """Build ``{nominee_name: film_title}`` for screenplay-style remapping.

        Original Screenplay is the awkward category where model-side nominee
        names and Kalshi-side film-title naming differ systematically. This
        helper keeps that category-specific remap close to the source dataset
        instead of spreading screenplay special cases across trading callers.
        """
        result: dict[str, str] = {}
        for record in self.records_for_ceremony_year(ceremony_year):
            nominee = record.nominee_name
            title = record.film.title
            if nominee and title:
                result[nominee] = title
        return result
