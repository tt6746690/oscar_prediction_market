# Build All Oscar Category Datasets

**Storage:** `storage/d20260218_build_all_datasets/`

Build raw datasets for all 9 Oscar categories that have precursor award mappings,
as part of the Phase 1 multi-category expansion.

## Categories Built

| Category | Records | Winners | Ceremonies | Avg noms/yr |
|----------|---------|---------|------------|-------------|
| Best Picture | 206 | 26 | 72–98 | 7.6 |
| Directing | 135 | 26 | 72–98 | 5.0 |
| Actor Leading | 135 | 26 | 72–98 | 5.0 |
| Actress Leading | 135 | 26 | 72–98 | 5.0 |
| Actor Supporting | 135 | 26 | 72–98 | 5.0 |
| Actress Supporting | 135 | 26 | 72–98 | 5.0 |
| Original Screenplay | 135 | 26 | 72–98 | 5.0 |
| Cinematography | 135 | 26 | 72–98 | 5.0 |
| Animated Feature | 109 | 24 | 74–98 | 4.4 |

Best Picture has a variable field size (5–10 since ceremony 82). Animated Feature
starts at ceremony 74 (2002, first year of the category). All others have exactly
5 nominees per year.

## Dataset Quality

All 9 categories pass completeness checks. Detailed numbers from the full report
(run `dataset_report.py` for any category, see How to Run below).

### Metadata completeness

| Category | IMDb | RT | Metacritic | Box Office | Budget |
|----------|------|----|------------|------------|--------|
| Best Picture | 100% | 97% | 100% | 93% | 99% |
| Directing | 100% | 99% | 100% | 96% | 99% |
| Actor Leading | 100% | 98% | 100% | 91% | 96% |
| Actress Leading | 100% | 98% | 100% | 89% | 93% |
| Actor Supporting | 100% | 97% | 100% | 93% | 97% |
| Actress Supporting | 100% | 97% | 100% | 91% | 90% |
| Original Screenplay | 99% | 97% | 99% | 94% | 95% |
| Cinematography | 100% | 98% | 99% | 89% | 96% |
| Animated Feature | 99% | 96% | 98% | 89% | 85% |

Box office gaps (89–93%) are concentrated in unreleased 2026 nominees and a few
obscure foreign films. Budget gaps in early 1980s and 2026 nominees. All core
critical reception fields (IMDb, Metacritic) are ≥99% for every category.

### Precursor coverage

| Category | Low-coverage awards | Note |
|----------|---------------------|------|
| Animated Feature | BAFTA 84%, PGA 87% | Both started tracking animation after the Oscar category launched |
| Cinematography | Critics Choice 63% | Critics Choice Cinematography started ~2009; 27/43 years |
| All others | 100% across all awards | – |

All precursors are 100% covered for the 7 non-animated categories, with the single
structural exception of Critics Choice Cinematography (award introduced mid-dataset).

### Person data completeness (person categories only)

| Category | Career history | TMDb enrichment |
|----------|----------------|-----------------|
| Directing | 100% | 98% |
| Actor Leading | 100% | 100% |
| Actress Leading | 100% | 100% |
| Actor Supporting | 100% | 100% |
| Actress Supporting | 100% | 100% |
| Original Screenplay | 100% | **4%** |
| Cinematography | 100% | 98% |

Career history (prior Oscar nominations/wins) is complete for all categories.
TMDb enrichment (birth date, popularity) is near-complete for directors, actors, and
cinematographers, but essentially absent for screenwriters.

**Screenwriter TMDb coverage is ~4%.** The root cause is structural: TMDb's person
database skews heavily toward on-screen talent. Director and actor pages are richly
populated; most screenwriter pages are stubs or missing altogether. The TMDb person
search (`/search/person`) frequently returns no results for writer names.

## Known Issues & Future Work

### 1. Screenwriter TMDb enrichment (4%) — medium priority

**Problem:** Birth date and popularity features are unavailable for Original Screenplay
nominees. The `fetch_tmdb.py` person search fails silently for most writers; the
`PersonData` record is created but all enrichment fields are `None`.

**Potential fixes:**
- **Wikidata lookup:** Query Wikidata by name + profession to retrieve birth date.
  The Wikidata SPARQL endpoint has good screenwriter coverage and is free.
- **IMDb name search:** The oscar_data submodule includes IMDb person IDs for nominees.
  Could use `fetch_omdb.py` person endpoint or `cinemagoer` (formerly IMDbPY) to
  retrieve birth dates directly from IMDb.
- **Accept the gap:** Popularity/birth date features matter less for screenplay than
  for acting categories where stardom is a predictive signal. The model can run without
  person enrichment for screenplay.

### 2. Critics Choice Cinematography starts 2009 — low priority

**Problem:** 63% precursor coverage is structural — the award did not exist for the
first 10 years of the dataset (ceremonies 72–80).

**Potential fix:** `PRECURSOR_YEAR_INTRODUCED` already tracks this. The dataset
report could distinguish "structurally absent" (year < year_introduced) from "missing"
(year ≥ year_introduced but data unavailable) to give cleaner coverage numbers. The
model already handles this correctly via `fillna(0)` for pre-introduction years.

### 3. Animated Feature PGA/BAFTA gaps — low priority

**Problem:** Same structural issue as Critics Choice above. PGA started tracking
animated features post-2002; BAFTA animated award started ~2007.

**Potential fix:** Same as above — report-level distinction between structural absence
and true missing data. No action needed for the model.

### 4. Box office gaps for 2026 nominees — expected

**Problem:** Several 2025-release films nominated in 2026 have no worldwide box office
data yet (films still in theatrical run at dataset build time).

**Expected behavior:** These are `None` in the dataset and fill to 0 in features.
No action needed; data will be available if the dataset is rebuilt after wider release.

### 5. `_clean_person_name` in `fetch_precursor_awards.py` duplicates `normalize_person_name`

**Problem:** `WikipediaScraper._clean_person_name()` is a private method reimplementing
accent-stripping and whitespace normalisation — the same logic now lives in
`data/utils.py` as `normalize_person_name`.

**Fix:** Replace the private method with a call to `normalize_person_name` from
`data/utils.py`. Minor cleanup, no functional impact.

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"

# Build all 9 categories (no date filtering — all precursor awards available)
zsh oscar_prediction_market/one_offs/d20260218_build_all_datasets/build_all.sh --all

# Build with as-of date (filters precursor awards by announcement date)
zsh oscar_prediction_market/one_offs/d20260218_build_all_datasets/build_all.sh 2026-02-18

# Report for a single category
uv run python -m oscar_prediction_market.data.dataset_report \
    storage/d20260218_build_all_datasets/best_picture/oscar_best_picture_raw.json --per-year --samples 3

# Reports for all 9 categories at once
for f in storage/d20260218_build_all_datasets/*/oscar_*_raw.json; do
  echo "=== $(basename $(dirname $f)) ==="
  uv run python -m oscar_prediction_market.data.dataset_report "$f"
done
```

The build script is idempotent — it skips categories whose output already exists.
Delete a category's output file to force a rebuild.

## Pipeline Stages

The shared metadata and precursor awards are built once and reused across all
categories. Per-category merge uses the shared files to avoid redundant fetches.

1. **Oscar nominations** (per category) — Extract nominees from `oscars.csv`
2. **Film metadata** (shared) — Fetch OMDb + TMDb for the union of all films across
   all 9 categories (585 unique films). Shared to avoid duplicate API calls.
3. **Precursor awards** (shared) — Fetch all 36 precursor awards from Wikipedia
   (cached via diskcache). One copy for all categories.
4. **Merge** (per category) — Compose `NominationRecord`s from all sources:
   - Person career data (prior Oscar noms/wins) for person categories
   - TMDb person enrichment (birth date, popularity, filmography)
   - Category-specific precursor matching (film-level or person-level)

## Output Structure

```
storage/d20260218_build_all_datasets/
├── shared/
│   ├── film_metadata.json          # 585 unique films (union across all 9 categories)
│   └── precursor_awards.json       # All 36 precursor award tables
├── best_picture/
│   ├── oscar_nominations.json
│   └── oscar_best_picture_raw.json
├── directing/
│   ├── oscar_nominations.json
│   └── oscar_directing_raw.json
├── actor_leading/
│   └── ...
├── actress_leading/
├── actor_supporting/
├── actress_supporting/
├── original_screenplay/
├── cinematography/
├── animated_feature/
└── ...                             # + dataset_stats.txt if generated
```
