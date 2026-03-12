# Multi-Category Feature Ablation

**Date:** 2026-02-20
**Storage:** `storage/d20260220_feature_ablation/`
**Status:** Planning

## Goal

Systematic feature group ablation for all 9 Oscar categories × 4 model types.
Determine which feature groups matter per category, confirm reasonable feature
sets, and produce "standard" feature configs ready for production model training.

Extends the BP-only ablation from
[d20260213_feature_ablation](../d20260213_feature_ablation/) to all 9 trading
categories.

---

## Prior Art: BP Ablation Findings (d20260213)

The BP ablation established key principles that inform this multi-category design:

- **3 groups are sufficient for BP:** precursor_winners + precursor_noms +
  oscar_nominations (10 features with selection) beats the full 47-feature set.
- **FULL vs BASE diverges by model type:** LR prefers FULL (interaction features
  help), GBT strongly prefers BASE (interactions hurt LOYO Jaccard 0.991→0.855).
- **Critic scores, commercial, timing, film_metadata are noise for BP.**
- **Feature selection (importance threshold 0.80) is beneficial:** Reduces noise
  and improves both Brier and temporal stability.

Key question for multi-category: **Does precursor dominance hold across all
categories, or do person_career / film_metadata / critic_scores matter more
for acting/directing/technical categories?**

---

## Feature Engineering Audit — Decisions Record

Research and decisions from the Feb 20 feature engineering review. Kept here as
the canonical reference for why each category's features look the way they do.

### Part 1: Precursor Mappings

**Verdict: Correct as-is.** Each category maps to its direct precursor awards via
`CATEGORY_PRECURSORS`. No changes needed.

| Category | Precursor Orgs | Notes |
|----------|---------------|-------|
| Best Picture | PGA, DGA, SAG (ensemble), BAFTA, GG (drama+musical), CC | GG composite handled |
| Directing | DGA, BAFTA, GG, CC | DGA dual-use (also in BP) — correct |
| Actor Leading | SAG, BAFTA, GG (drama+musical), CC | GG genre-split → composite feature |
| Actress Leading | SAG, BAFTA, GG (drama+musical), CC | Same as Actor |
| Actor Supporting | SAG, BAFTA, GG, CC | No GG genre-split for supporting |
| Actress Supporting | SAG, BAFTA, GG, CC | Same as Sup Actor |
| Original Screenplay | WGA, BAFTA, CC, GG | GG Screenplay is not split original/adapted — diluted signal |
| Cinematography | ASC, BAFTA, CC | Only 3 orgs |
| Animated Feature | Annie, BAFTA, PGA, GG, CC | 5 orgs — richest set |

**Design principle: Clean direct precursors only.** No cross-category precursor
mappings (e.g., no PGA_BP → Director). Cross-category signal is captured via
`film_also_bp_nominated` and `oscar_total_nominations`.

### Part 2: Feature Engineering Per-Category

#### A. Remove constant (always-True) features from configs

These produce zero-variance features — waste model capacity, cause LR collinearity.

| Feature | Category | Why constant |
|---------|----------|-------------|
| `has_director_nomination` | Director | Every Director nominee's film has a directing nom |
| `has_acting_nomination` | All 4 acting categories | Every acting nominee's film has at least this acting nom |
| `has_screenplay_nomination` | Original Screenplay | The nomination IS a screenplay nomination |

**Action:** Remove from corresponding category configs.

#### B. Add existing features to configs where missing

| Feature | Add to | Rationale | TMDb Coverage |
|---------|--------|-----------|---------------|
| `person_age_at_ceremony` | Director | Age matters (young prodigy vs veteran legacy) | 92% |
| `person_total_film_credits` | Director | Career stature/prolificness signal | 92% |
| `person_tmdb_popularity` | Director | Public recognition/buzz proxy | 92% |

**Skip person TMDb features for Cinematography** (53% coverage — too sparse for
age/popularity). Career history (prev_noms/wins/overdue) from oscars.csv is
already included and has 100% coverage.

**Skip person TMDb features for Screenplay** (4% TMDb enrichment). Career
history from oscars.csv already included.

#### C. New features to implement

| Feature | Categories | Description |
|---------|-----------|-------------|
| `is_first_nomination` | All person categories (LR only) | `person_prev_noms_same_category == 0`. Breakthrough narrative signal. GBT can learn this from the raw count. |
| `genre_comedy` | All categories | Comedies rarely win — real negative signal. Currently missing from genre flags. |
| `studio_ghibli` | Animated Feature | Non-US winner (Spirited Away, Boy and the Heron). Current studios are all US. |

#### D. Features NOT to add (considered and rejected)

| Feature | Why skip |
|---------|---------|
| `total_awards_wins/nominations` from FilmMetadata | Can't time-gate — OMDb aggregates all awards with unknown timing. Would leak future info. |
| `person_prev_noms_percentile_in_year` | With 5 nominees, within-year percentile of career stats is too noisy |
| `precursor_sweep_count` (cross-category) | Complex; `precursor_wins_count` + base features capture this |
| `is_irv_era` for non-BP | Only BP uses IRV; all other categories use plurality |
| `nominees_in_year` for non-BP | Always 5 for non-BP |
| `log_imdb_votes` / `imdb_votes` | Moderate value, decided to skip for now |
| `domestic_international_ratio` | Moderate value, redundant with individual features |

#### E. Config corrections vs new feature code

New feature implementations needed: `is_first_nomination`, `genre_comedy`,
`studio_ghibli`. These are simple transforms — add to `feature_engineering.py`
FEATURE_REGISTRY.

Config changes: Remove constant features, add Director TMDb features. These are
JSON config edits only.

**Decision: Do NOT implement new features before ablation.** Run ablation with
the current feature set to establish baselines. New features can be added in a
follow-up pass and compared against the ablation baseline.

---

## Datasets

Use existing datasets from `storage/d20260218_build_all_datasets/`:

| Category | Raw Path | Records |
|----------|---------|---------|
| Best Picture | `best_picture/oscar_best_picture_raw.json` | 206 |
| Directing | `directing/oscar_directing_raw.json` | 135 |
| Actor Leading | `actor_leading/oscar_actor_leading_raw.json` | 135 |
| Actress Leading | `actress_leading/oscar_actress_leading_raw.json` | 135 |
| Actor Supporting | `actor_supporting/oscar_actor_supporting_raw.json` | 135 |
| Actress Supporting | `actress_supporting/oscar_actress_supporting_raw.json` | 135 |
| Original Screenplay | `original_screenplay/oscar_original_screenplay_raw.json` | 135 |
| Cinematography | `cinematography/oscar_cinematography_raw.json` | 135 |
| Animated Feature | `animated_feature/oscar_animated_feature_raw.json` | 109 |

These datasets were built with **all precursor data** (no `--as-of-date` gating).
Feature availability gating happens at transform time via `build_model --as-of-date`.

**as_of_date for ablation:** Use current date `2026-02-20`. This provides the
most complete feature set for current-season predictions (all precursor noms
announced; CC, GG, DGA winners announced; BAFTA, PGA, SAG, ASC, WGA, Annie
winners not yet).

---

## Models

4 model types per category (2 feature families):

| Model | Short | Param Grid | Feature Family | Feature Selection Support |
|-------|-------|-----------|----------------|--------------------------|
| Logistic Regression | lr | `lr_grid.json` | LR features | Yes (abs_coefficient > 0) |
| Conditional Logit | clogit | `conditional_logit_grid.json` | LR features | Yes (abs_coefficient > 0) |
| Gradient Boosting | gbt | `gbt_grid.json` | GBT features | Yes (importance > 0) |
| Calibrated Softmax GBT | cal_sgbt | `calibrated_softmax_gbt_grid.json` | GBT features | Yes (importance > 0) |

LR and Clogit share the same feature configs (percentile-in-year, log transforms,
interactions). GBT and Cal-Softmax-GBT share the same feature configs (raw values,
z-score-in-year). So we need **2 feature config families** per category, not 4.

---

## Feature Groups

Extend `generate_feature_ablation_configs.py` to accept `--category` and produce
category-specific feature groups. The group **structure** is the same for all
categories; the **features within each group** differ.

### Universal groups (all categories)

| Group | Description | LR variant | GBT variant |
|-------|-------------|-----------|-------------|
| **precursor_winners** | Category-specific precursor winner flags + aggregates | Individual winner flags + composites + `precursor_wins_count` + combo features (BP only) | Individual winner flags + `precursor_wins_count` |
| **precursor_noms** | Category-specific precursor nomination flags + aggregates | Individual nominee flags + composites + `precursor_nominations_count` + combo features (BP only) | Individual nominee flags + `precursor_nominations_count` |
| **oscar_nominations** | Cross-category Oscar nomination profile | `oscar_total_nominations`, `has_*_nomination` (minus trivially-True), `*_count`, `nominations_percentile_in_year` | Same minus percentile |
| **critic_scores** | Review aggregator scores | `critics_consensus_score`, `critics_audience_gap`, `*_percentile_in_year` | Raw `imdb_rating`, `rotten_tomatoes`, `metacritic`, `*_zscore_in_year` |
| **commercial** | Budget and box office | `log_budget`, `log_box_office_worldwide`, `*_percentile_in_year` | Raw `budget`, `box_office_*`, `*_zscore_in_year` |
| **timing** | Release timing and runtime | `release_month`, `release_month_sin/cos`, `is_awards_season_release`, `runtime_minutes`, `runtime_percentile_in_year` | Same minus sin/cos, plus `runtime_zscore_in_year` |
| **film_metadata** | Genre indicators and MPAA rating | `genre_drama`, `genre_biography`, `genre_war`, `genre_musical`, `rated_r` | Same |

### Conditional groups (category-dependent)

| Group | Categories | Description |
|-------|-----------|-------------|
| **person_career** | All PERSON_CATEGORIES (acting, directing, screenplay, cinematography) | `person_prev_noms/wins_same/any`, `person_is_overdue`, `film_also_bp_nominated` |
| **person_enrichment** | Acting (4) + Directing | `person_age_at_ceremony`, `person_total_film_credits`, `person_tmdb_popularity` |
| **animated_specific** | Animated Feature only | `is_sequel`, `studio_disney_pixar`, `studio_dreamworks`, etc. |
| **voting_system** | Best Picture only | `is_irv_era`, `nominees_in_year` |

### Total groups per category

| Category | Groups | Expected dominant |
|----------|--------|-------------------|
| Best Picture | 8 (universal + voting_system) | precursor_winners (proven) |
| Acting (4 categories) | 9 (universal + person_career + person_enrichment) | precursor_winners + person_career? |
| Directing | 9 (universal + person_career + person_enrichment) | precursor_winners (DGA ~80% hit rate) |
| Original Screenplay | 8 (universal + person_career, no enrichment) | precursor_winners (WGA/BAFTA) |
| Cinematography | 8 (universal + person_career, no enrichment) | precursor_winners (ASC) |
| Animated Feature | 8 (universal + animated_specific) | precursor_winners (Annie) + animated_specific? |

---

## Ablation Design

### Phase 1: Quick Pass (Additive Ablation)

Run additive ablation with feature selection for all 9 categories × 4 models.

**Ablation strategy:** Additive — start with `precursor_winners` (most predictive),
add groups one at a time in priority order. Priority order per category type:

**Film-level (BP, Animated):**
1. precursor_winners
2. precursor_noms
3. oscar_nominations
4. [animated_specific | voting_system]
5. critic_scores
6. commercial
7. timing
8. film_metadata

**Person-level (Acting, Directing, Screenplay, Cinematography):**
1. precursor_winners
2. precursor_noms
3. oscar_nominations
4. person_career
5. person_enrichment (if applicable)
6. critic_scores
7. commercial
8. timing
9. film_metadata

### Run count estimate

Per category:
- N_groups additive configs + 1 full baseline = ~9-10 configs
- × 4 models = ~36-40 runs
- With feature selection (importance threshold 0.80)

Total: 9 categories × ~38 runs = **~342 runs**

Each run does CV (26 folds) + train + predict. At ~10-30s per run, total ~1-3 hours.

### Phase 2: Targeted Deep Dives (if needed)

After Phase 1 results:
- Leave-one-out ablation for categories with surprising results
- Single-group isolation for categories where person_career competes with precursors
- Only for categories with strong enough signal to be worth trading

**Decision: defer Phase 2 until Phase 1 results are analyzed.**

---

## Implementation Plan

### Step 1: Extend `generate_feature_ablation_configs.py`

Add `--category` parameter. Refactor feature group definitions to be
category-aware:

```python
def get_feature_groups(category: OscarCategory) -> list[FeatureGroup]:
    """Return ordered feature groups for a category."""
    groups = [
        _precursor_winners_group(category),
        _precursor_noms_group(category),
        _oscar_nominations_group(category),
    ]
    # Conditional groups
    if category in PERSON_CATEGORIES:
        groups.append(_person_career_group(category))
    if category in PERSON_ENRICHMENT_CATEGORIES:
        groups.append(_person_enrichment_group())
    if category == OscarCategory.ANIMATED_FEATURE:
        groups.append(_animated_specific_group())
    if category == OscarCategory.BEST_PICTURE:
        groups.append(_voting_system_group())
    # Universal tail groups
    groups.extend([
        _critic_scores_group(),
        _commercial_group(),
        _timing_group(),
        _film_metadata_group(),
    ])
    return groups
```

Each `_*_group(category)` function reads from `CATEGORY_PRECURSORS` and the
existing feature configs to build the right feature list dynamically.

### Step 2: Fix category configs (remove constant features)

| Config Files | Change |
|-------------|--------|
| `director_lr_standard.json`, `director_gbt_standard.json` | Remove `has_director_nomination` |
| `actor_lr_standard.json`, `actor_gbt_standard.json` | Remove `has_acting_nomination` |
| `actress_lr_standard.json`, `actress_gbt_standard.json` | Remove `has_acting_nomination` |
| `sup_actor_lr_standard.json`, `sup_actor_gbt_standard.json` | Remove `has_acting_nomination` |
| `sup_actress_lr_standard.json`, `sup_actress_gbt_standard.json` | Remove `has_acting_nomination` |
| `screenplay_lr_standard.json`, `screenplay_gbt_standard.json` | Remove `has_screenplay_nomination` |

Also add to Director configs:
- `person_age_at_ceremony`
- `person_total_film_credits`
- `person_tmdb_popularity`

### Step 3: Create run scripts

```
one_offs/d20260220_feature_ablation/
├── PLAN.md                     # This document
├── README.md                   # Results (after experiments)
├── __init__.py
├── run_generate_configs.sh     # Generate ablation configs for all 9 categories
├── run_phase1.sh               # Phase 1: additive ablation for all categories
├── analyze_results.py          # Cross-category comparison tables
└── assets/                     # Plots
```

```
storage/d20260220_feature_ablation/
├── configs/                    # Copied configs (Ground truth for experiment)
│   ├── features/
│   │   ├── best_picture/       # Ablation configs per category
│   │   ├── directing/
│   │   ├── actor_leading/
│   │   └── ...
│   ├── param_grids/            # Copied from modeling/configs/
│   └── cv_splits/
├── best_picture/               # Results per category
│   ├── lr/                     # Per model type
│   │   ├── lr_additive_1_precursor_winners/
│   │   ├── lr_additive_2_precursor_nominations/
│   │   └── ...
│   ├── gbt/
│   ├── clogit/
│   └── cal_sgbt/
├── directing/
│   └── ...
└── summary/
    ├── phase1_results.csv      # Cross-category comparison
    └── phase1_results.json
```

### Step 4: Run Phase 1

```bash
cd "$(git rev-parse --show-toplevel)"
bash .../run_phase1.sh 2>&1 | tee storage/d20260220_feature_ablation/run.log
```

### Step 5: Analyze and produce README

Run `analyze_results.py` to produce:
- Per-category: best model type, best feature group combination, Brier/accuracy
- Cross-category comparison table
- Feature group importance ranking per category
- Recommendation: which groups to include in final "standard" config per category

---

## Success Criteria

1. **Per-category best feature subset identified:** For each of 9 categories, know
   which feature groups to include in the production config.
2. **Model type comparison:** For each category, know whether LR, GBT, Clogit, or
   Cal-Softmax-GBT performs best.
3. **Precursor dominance hypothesis tested:** Confirm (or refute) that precursor
   features dominate for all categories, not just BP.
4. **Person features value quantified:** For acting/directing, measure the marginal
   contribution of person_career and person_enrichment groups.
5. **Updated standard configs:** Revised `{category}_{model}_standard.json` files
   reflecting ablation findings.

---

## Open Questions

1. ~~Do we need to rebuild datasets?~~ **No.** Existing datasets have all precursor
   data. Feature availability gating happens at transform time via `--as-of-date`.

2. **Should we run feature selection for all models?** Feature selection
   (importance threshold 0.80) was beneficial for BP. Plan to use it for all runs.
   But Clogit and Cal-Softmax-GBT haven't been tested with feature selection — need
   to verify the pipeline handles their importance formats correctly. (Checked:
   Clogit produces `abs_coefficient`, Cal-Softmax-GBT produces `importance` —
   both handled by `_extract_selected_features`.)

3. **Phase 1 run time estimate.** ~342 runs × ~10-30s each = ~1-3 hours. Acceptable
   for a single execution. Can parallelize across categories if needed (each
   category is independent), but sequential is simpler.
