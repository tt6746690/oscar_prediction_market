# Oscar Data Pipeline

This directory contains the data pipeline for building Oscar nomination datasets
across 9 categories. The pipeline covers 27 years of awards history (ceremonies
72–98, films 1999–2025).

**Supported categories:** Best Picture, Directing, Actor/Actress Leading,
Actor/Actress Supporting, Original Screenplay, Cinematography, Animated Feature.

## Architecture

The pipeline follows a **3-layer architecture**:

```
Layer 1: Intermediate Files (per-source raw data)
├── oscar_nominations.json      # From oscars.csv
├── film_metadata.json          # From OMDb + TMDb APIs (shared across categories)
└── precursor_awards.json       # From Wikipedia (shared across categories)

Layer 2: Raw Dataset (merged, no derived features)
└── oscar_<category>_raw.json   # Composed from Layer 1

Layer 3: Feature-Engineered Dataset (model-ready)
└── oscar_<category>_features.json  # All transformations applied
```

## Quick Start

```bash
# Build a single category (full pipeline: oscar → metadata → precursors → merge)
uv run python -m oscar_prediction_market.data.build_dataset \
    --category best_picture

# Build all 9 categories at once (shared metadata + precursors fetched once)
zsh oscar_prediction_market/one_offs/d20260218_build_all_datasets/build_all.sh --all

# Run dataset quality report
uv run python -m oscar_prediction_market.data.dataset_report \
    storage/d20260218_build_all_datasets/best_picture/oscar_best_picture_raw.json

# Run individual pipeline stages
uv run python -m oscar_prediction_market.data.build_dataset --mode oscar
uv run python -m oscar_prediction_market.data.build_dataset --mode metadata
uv run python -m oscar_prediction_market.data.build_dataset --mode precursors
uv run python -m oscar_prediction_market.data.build_dataset --mode merge
```

## Pipeline Stages

| Stage | Mode | Output | Description |
|-------|------|--------|-------------|
| 1 | `oscar` | `oscar_nominations.json` | Extract Oscar nominations from oscars.csv |
| 2 | `metadata` | `film_metadata.json` | Fetch OMDb + TMDb film metadata |
| 3 | `precursors` | `precursor_awards.json` | Fetch all guild award tables from Wikipedia |
| 4 | `merge` | `oscar_<category>_raw.json` | Merge into per-record NominationRecords |
| 5 | (separate) | `oscar_<category>_features.json` | Feature engineering |

Stages 2 and 3 fetch from external APIs and are **cached** via `diskcache`. Subsequent
runs skip already-fetched items. Stage 4 is fast and can be rerun freely.

## Directory Structure

```
data/
├── oscars.csv                   # All Oscar nominations/winners (1927–present, from DLu/oscar_data)
├── awards_calendar.py           # Awards season event datetimes (AwardsCalendar model + per-year constants)
├── build_dataset.py             # Pipeline stages 1–4: build raw dataset
├── dataset_report.py            # Dataset quality report (completeness, coverage)
├── fetch_omdb.py                # OMDb API fetcher (ratings, runtime, genre)
├── fetch_tmdb.py                # TMDb API fetcher (budget, box office, person data)
├── fetch_precursor_awards.py    # Wikipedia scraper for all 36 guild award tables
├── oscar_winners.py             # Per-year Oscar winners + Kalshi market conventions
├── precursor_mappings.py        # Oscar category → precursor award mappings
├── schema.py                    # Pydantic data models for all pipeline layers
└── utils.py                     # Text utilities: name normalisation, screenplay parsing
```

## Module Responsibilities

### `schema.py`
Pydantic models for every data layer. Key types:
- `OscarNominationsFile` / `OscarNominee` — Layer 1 Oscar data
- `FilmMetadataFile` / `FilmData` / `FilmMetadata` / `PersonData` — Layer 1 metadata
- `PrecursorAwardsFile` / `PrecursorAwardRecord` — Layer 1 guild award tables
- `NominationDataset` / `NominationRecord` — Layer 2 merged records
- `OscarCategory`, `PrecursorKey`, `PrecursorAward` — enums for categories and awards
- `AwardsCalendar`, `CALENDARS` — per-ceremony award announcement dates

### `awards_calendar.py`
Awards season calendar types and constants. Defines `AwardsCalendar` model mapping
`(AwardOrg, EventPhase)` pairs to UTC datetimes for each ceremony year. Used by
the temporal model and backtesting pipeline to determine when precursor results
become available. All datetimes are UTC (no suffix — project-wide convention).

### `oscar_winners.py`
Per-year Oscar ceremony winners and Kalshi market conventions, consolidated in one
place. Exports `WINNERS_2024`, `WINNERS_2025`, etc. keyed by `OscarCategory`.

### `precursor_mappings.py`
Maps each `OscarCategory` to the precursor awards that predict it:
- `CATEGORY_PRECURSORS: dict[OscarCategory, list[PrecursorSpec]]`
- `PrecursorSpec(key: PrecursorKey, award: PrecursorAward, match_mode: MatchMode)` —
  `film` for film-level categories (Best Picture, Animated), `person` for all others

### `fetch_precursor_awards.py`
Wikipedia scraper with diskcache. Fetches 36 award tables (10+ guild organisations,
multiple categories each). Key exports:
- `fetch_all_awards(as_of_date)` — returns `dict[str, DataFrame]` for all awards
- `AWARD_URLS`, `AWARD_TABLE_CONFIG`, `PRECURSOR_YEAR_INTRODUCED` — all keyed by `PrecursorKey`

### `utils.py`
Text-processing utilities shared across pipeline stages:
- `normalize_person_name(name)` — strips accents, lowercases, collapses whitespace.
  Used for cross-source name matching (oscars.csv ↔ Wikipedia ↔ TMDb).
- `clean_screenplay_names(raw_name)` — parses oscars.csv credit format
  ("Written by A and B; Story by C") into comma-separated individual names.

### `dataset_report.py`
Prints a comprehensive quality report for any raw dataset file:
- Record counts, winner counts, ceremony range
- Metadata completeness per field
- Precursor award coverage per award
- Person data completeness (for person categories)
- Duplicate check (same film/person appearing twice in a ceremony year)
- Optional: per-year breakdown (`--per-year`), sample records (`--samples N`)

## Schema Overview

### Layer 2: Merged Record

```python
class NominationRecord(BaseModel):
    oscar: OscarNominee              # Ceremony, category, film, winner flag
    film: FilmData                   # Metadata + critic scores + box office
    person: PersonData | None        # Career history + TMDb enrichment (person categories)
    precursors: dict[str, AwardResult]  # keyed by PrecursorKey value
```

### Layer 3: Feature Record (flat, model-ready)

Produced by `modeling/feature_engineering.py`. Key feature groups (80+ total across
all categories):
- **Precursor features**: `{key}_winner`, `{key}_nominee` for each mapped award;
  composite GG features (`golden_globe_any_winner`, etc.)
- **Best Picture interaction features**: `has_pga_dga_combo`, `has_sag_dga_combo`, etc.
- **Critical reception**: `imdb_rating`, `rotten_tomatoes`, `metacritic`, `critics_consensus_score`
- **Commercial**: `log_budget`, `log_box_office`, `roi`, `is_blockbuster`
- **Release timing**: `release_month`, `is_awards_season_release`, `days_before_oscars`
- **Person features** (person categories): `prior_oscar_nominations`, `prior_oscar_wins`,
  `person_age_at_ceremony`, `person_popularity`
- **Oscar history**: `oscar_nominations_count`, `has_director_nomination` (BP only)

## Data Sources

### Primary: DLu/oscar_data
All Oscar nominations/winners from 1927–present, including IMDb IDs for films and nominees.

### OMDb API
Film ratings (IMDb, Rotten Tomatoes, Metacritic), runtime, genre, US domestic box office.

### TMDb API
Production budget, worldwide box office, production companies, person data (birth date,
popularity, filmography). Person lookup used for acting, directing, screenplay, cinematography.

### Precursor Awards (Wikipedia)
36 award tables across 10+ organisations: PGA, DGA, SAG, BAFTA, Golden Globe (Drama +
Musical/Comedy splits), Critics Choice, WGA, ASC, Annie Awards.

## Year Convention

Uses **ceremony year** (not film year):
- `--year-start 2000` = 72nd Academy Awards (films from 1999)
- `--year-end 2026` = 98th Academy Awards (films from 2025)

Conversion: `ceremony_number = year - 1928`

## Dataset Statistics (Generated 2026-02-19)

**Current dataset**: ceremonies 72–98 (1999–2025 films), 26 ceremony years.

### Nominees Per Year

| Category | Nominees/year | Notes |
|----------|---------------|-------|
| Best Picture | 5–10 | 5 until ceremony 82; variable 8–10 since; 10 fixed since ceremony 94 |
| All other categories | 5 | Fixed throughout |
| Animated Feature | 5 | Category introduced at ceremony 74 (2002) |

### Precursor Award Coverage

| Award | Available From | Coverage |
|-------|---------------|----------|
| DGA | 1980 | 100% for all categories |
| BAFTA | 1980 | 100% (84% for Animated — BAFTA animated started ~2007) |
| Golden Globe | 1980 | 100% for all categories |
| PGA | 1990 | 100% (87% for Animated — PGA animated started later) |
| SAG | 1996 | 100% for all categories |
| Critics Choice | 1996 | 100% (63% for Cinematography — award started ~2009) |
| WGA | varies | 100% for Original Screenplay |
| ASC | varies | 100% for Cinematography |
| Annie Awards | 2002 | 100% for Animated Feature |

### Data Quality Notes

- **Missing precursors → fillna(0)**: When a precursor award didn't exist for a year
  (e.g., SAG before 1996), the feature becomes 0. This means "award didn't exist"
  is treated identically to "not nominated." Appropriate for most models; documented
  in `PRECURSOR_YEAR_INTRODUCED` in `fetch_precursor_awards.py`.
- **Screenwriter TMDb enrichment is ~4%**: TMDb's person database skews toward
  on-screen talent; screenwriter pages are mostly stubs. Career history features
  remain 100% for screenplay. See
  [build_all_datasets README known issues](../one_offs/d20260218_build_all_datasets/README.md#known-issues--future-work)
  for fix options (Wikidata, IMDb person IDs).
- **Dataset reports**: Run
  `uv run python -m oscar_prediction_market.data.dataset_report path/to/raw.json`
  for completeness and coverage for any raw dataset file.
