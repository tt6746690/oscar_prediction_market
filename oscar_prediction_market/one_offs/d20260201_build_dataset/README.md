# Build Dataset One-Off

**Date:** 2026-02-01
**Storage:** `storage/d20260201_build_dataset/`

## Goal

Build the canonical Oscar Best Picture dataset from raw sources. This is the base
dataset used by all modeling experiments. Downstream experiments reference the
intermediate files (oscar_nominations.json, film_metadata.json, precursor_awards.json)
from this directory.

## Pipeline

The dataset is built in 4 stages:

| Stage | Source | Output | Notes |
|-------|--------|--------|-------|
| 1. Oscar nominations | `oscar_data/oscars.csv` | `oscar_nominations.json` | Parsed from DLu's Oscar data repo |
| 2. Film metadata | OMDb + TMDb APIs | `film_metadata.json` | Ratings, box office, budget, runtime |
| 3. Precursor awards | Wikipedia | `precursor_awards.json` | PGA, DGA, SAG, BAFTA, Globe, Critics Choice |
| 4. Merge | All above | `oscar_best_picture_raw.json` | Composed records with optional `--as-of-date` filtering |

Stages 1-3 produce date-independent intermediate files. Stage 4's merge output
depends on `--as-of-date` which controls what precursor award results are visible
for the current season (2026).

## Usage

```bash
# Full rebuild (all awards visible)
bash oscar_prediction_market/one_offs/d20260201_build_dataset/build_dataset.sh --all

# As of specific date (filters current season precursors)
bash oscar_prediction_market/one_offs/d20260201_build_dataset/build_dataset.sh 2026-02-04
```

## Output Files

| File | Records | Description |
|------|---------|-------------|
| `oscar_nominations.json` | ~260 | Best Picture nominees, 2000-2026 |
| `film_metadata.json` | ~260 | OMDb + TMDb metadata per film |
| `precursor_awards.json` | ~1500 | Precursor award noms/wins across 6 awards |
| `oscar_best_picture_raw.json` | ~260 | Merged raw dataset (input to feature engineering) |

## Dataset Characteristics

- **Ceremony years:** 72nd (2000) through 98th (2026)
- **Films:** ~260 Best Picture nominees
- **Features available (raw):** Oscar nominations, film metadata (ratings, box office,
  budget, runtime, release date), precursor awards (6 awards × nom/win)
- **Missing data:** Budget coverage ~70%, domestic box office ~80%, all other fields ~95%+
- **Current season (2026):** 10 nominees. Precursor availability depends on `--as-of-date`.

## Takeaways

- OMDb and TMDb provide complementary metadata — OMDb for ratings/awards, TMDb for budget/revenue
- Precursor award matching uses fuzzy string matching (Levenshtein distance) since film
  titles differ across sources
- All API calls are cached via diskcache, so subsequent runs are fast
- The `--as-of-date` mechanism is critical for temporal experiments — it controls what
  the model "knows" at prediction time
