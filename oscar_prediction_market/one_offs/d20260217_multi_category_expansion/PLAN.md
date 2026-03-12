# Multi-Category Oscar Prediction Expansion

**Date:** 2026-02-17 (updated 2026-02-20)
**Goal:** Expand the Best Picture prediction + trading pipeline to cover all 9 Oscar
categories with live Kalshi markets, enabling multi-category arbitrage.

**Status:** Phase 0 (schema refactor), Phase 1-Data (all 9 datasets built), and
Phase 1.5 (feature engineering audit) are **complete**. Feature audit findings and
decisions are documented in
[d20260220_feature_ablation/PLAN.md](../d20260220_feature_ablation/PLAN.md).
Next: feature ablation (d20260220) → model training → trading.

## Decisions Log

Key decisions made during planning and implementation:

- **Schema:** `NominationRecord` composes `FilmData` + `PersonData` + `PrecursorAwardsComplete`.
  Precursors use **nested org models** (e.g., `r.precursors.sag.lead_actor.winner`), not flat
  fields. This was a Phase 2 improvement that was pulled forward.
- **Precursor awards:** Single `PrecursorAwardsComplete` record with ALL precursor orgs; feature
  selection per category decides what matters (via `CATEGORY_PRECURSORS` / `PrecursorSpec` registry)
- **Feature engineering:** Single `feature_engineering.py` with all features (film, person,
  animated-specific, precursor); per-category feature selection via JSON configs
- **Model architecture:** One model per category, separate feature config per
  model type × category
- **Trading:** `OscarMarket(event_ticker, nominee_tickers)` — generic market wrapper.
  `OSCAR_EVENT_TICKERS` maps all 9 categories to Kalshi tickers.
- **Configs vs code:** Kalshi tickers + precursor URLs = Python constants; feature
  configs + model hyperparams = JSON files
- **Training year range:** **2000–2026** for all categories. BP ablation found this
  outperforms longer windows (older data has no precursor signal).
- **Person enrichment:** TMDb person API for birthdate, popularity, filmography. Career
  history from oscars.csv. Both implemented and working.
- **Backward compatibility:** Not a concern — all downstream code refactored
- **Execution order:** Sequential: feature audit → model training → ablation → trading.
  No parallel work groups needed at this stage.
- **DGA dual-use:** DGA matched film-level for BP, person-level for Director. Same data,
  different match path. Implemented.
- **AwardsCalendar:** `PrecursorAward` enum has 9 orgs (PGA, DGA, SAG, BAFTA, GG, CC, WGA,
  ASC, Annie). `CALENDAR_2026` fully populated for all 9. `as_of_date` gating works for all
  categories.
- **Person × Film relationship:** Person-level categories match `(person_name, film_title)`.
  Film-level categories aggregate "did any person from this film win?"
- **Intermediate files:** Shared intermediates (oscar_nominations, film_metadata,
  precursor_awards) fetched once across all categories. Per-category datasets composed from
  shared intermediates via `build_dataset.py --category X --shared-dir`.
- **Person name normalization:** `normalize_person_name()` — strip accents, lowercase,
  collapse whitespace. `clean_screenplay_names()` for oscars.csv credits.

---

## 1. Kalshi Markets — Scope

| # | Category | Event Ticker | Volume | Nominees | Type |
|---|----------|--------------|--------|----------|------|
| 1 | **Best Picture** (modeled) | KXOSCARPIC-26 | $10.0M | 11 | film |
| 2 | **Best Actor** | KXOSCARACTO-26 | $5.5M | 5 | person |
| 3 | **Best Supporting Actor** | KXOSCARSUPACTO-26 | $2.0M | 5 | person |
| 4 | **Best Director** | KXOSCARDIR-26 | $1.5M | 5 | person/film |
| 5 | **Best Supporting Actress** | KXOSCARSUPACTR-26 | $1.5M | 5 | person |
| 6 | **Best Actress** | KXOSCARACTR-26 | $1.3M | 5 | person |
| 7 | **Best Cinematography** | KXOSCARCINE-26 | $1.2M | 5 | film |
| 8 | **Best Orig. Screenplay** | KXOSCARSPLAY-26 | $463K | 5 | person/film |
| 9 | **Best Animated Feature** | KXOSCARANIMATED-26B | $306K | 5 | film |

**Total addressable volume (beyond BP): ~$14.8M**

> **Note on nominee counts:** Kalshi may create more than 5 contracts per category (e.g., 17 was
> observed for some supporting/screenplay categories) as pre-nomination speculative markets.
> The Oscar nominates exactly 5 per category. Training data has 5 nominees per ceremony.
> Models are trained and evaluated over 5-nominee distributions.

---

## 2. Completed Work Summary

Everything below has been implemented and merged. This section documents what exists
so future phases can build on it.

### 2a. Schema (`data/schema.py`) — DONE

| Component | What it is |
|-----------|-----------|
| `OscarCategory` enum | 20 categories (all Oscar award types) |
| `PrecursorAward` enum | 9 orgs: PGA, DGA, SAG, BAFTA, GG, CC, WGA, ASC, Annie |
| `FILM_CATEGORIES` / `PERSON_CATEGORIES` | Classification of categories into film-level vs person-level |
| `AwardResult` | `{nominee: bool\|None, winner: bool\|None}` — per-award data |
| Org-level models | `PGAAwards`, `DGAAwards`, `SAGAwards`, `BAFTAAwards`, `GoldenGlobeAwards`, `CriticsChoiceAwards`, `WGAAwards`, `ASCAwards`, `AnnieAwards` |
| `PrecursorAwardsComplete` | Composes all 9 org models. Access: `r.precursors.sag.lead_actor.winner` |
| `PrecursorSpec` / `CompositeSpec` | Registry mapping categories → relevant precursor awards |
| `CATEGORY_PRECURSORS` | All 9 trading categories mapped to their precursor specs |
| `CATEGORY_COMPOSITES` | Golden Globe genre-split → combined features (BP, Actor, Actress) |
| `FilmData` | `film_id, title, metadata (FilmMetadata), oscar_noms (OscarNominationInfo)` |
| `PersonData` | `name, prev_noms/wins (same/any), birth_date, tmdb_popularity, total_film_credits` |
| `NominationRecord` | One row = one nomination: `category, ceremony, year_film, category_winner, nominee_name, film, person, precursors` |
| `NominationDataset` | Collection wrapper: `category, year_start, year_end, record_count, records` |
| `AwardsCalendar` | Full 2025 + 2026 calendars for all 9 orgs. Supports `as_of_date` gating. |
| `normalize_person_name()` | Strip accents, lowercase, collapse whitespace |
| `clean_screenplay_names()` | Strip "Written by"/"Screenplay by" prefixes |

### 2b. Data Pipeline (`data/build_dataset.py`) — DONE

```
CLI: --mode {oscar,metadata,precursors,merge,all} --category <OscarCategory>
     --year-start 2000 --year-end 2026 --output-dir <path>
     --shared-dir <path>  # shared metadata/precursors
     --extra-input-dirs    # additional dirs for metadata union
     --as-of-date          # current-season filtering
```

5-stage pipeline: oscar → metadata → precursors → person (auto in merge) → merge.
All stages produce shared intermediates except merge which is per-category.

### 2c. Precursor Fetching (`data/fetch_precursor_awards.py`) — DONE

| What | Count | Status |
|------|-------|--------|
| `AWARD_URLS` | 36 entries | All populated (7 BP + 29 new) |
| `AWARD_TABLE_CONFIG` | 36 entries | Film + person column hints per award |
| `PRECURSOR_YEAR_INTRODUCED` | 36 entries | First year each award was given |
| Person-level parsing | All person-level awards | `(person_name, film_title)` matching |
| Generic parser | `_fetch_generic_award()` | Handles all 36 table formats |

### 2d. Feature Engineering (`modeling/feature_engineering.py`) — DONE (initial pass; audit needed)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| BASE_FEATURES | 16 | Oscar noms, cross-cat noms, release timing, genre, runtime, IRV |
| LR_SPECIFIC | 14 | Percentile-in-year, log transforms, critics scores |
| GBT_SPECIFIC | 16 | Raw ratings/commercial, z-score-in-year |
| PERSON_FEATURES | 9 | Career history, is_overdue, age, TMDb popularity/credits, BP crossover |
| ANIMATED_FEATURES | 6 | is_sequel, studio identity (Disney/Pixar, DreamWorks, etc.) |
| CATEGORY_PRECURSOR_FEATURES | ~70 | Dynamically generated from `CATEGORY_PRECURSORS` (winner + nominee per spec) |
| COMPOSITE_FEATURES | ~6 | Golden Globe genre-split → combined (BP, Actor, Actress) |
| BP_INTERACTION_FEATURES | 2 | PGA+DGA combo features |

`TransformFn = Callable[[NominationRecord, TransformContext], Any]` — fully multi-category.

### 2e. Feature Configs — DONE (25 files)

| Category | LR | GBT | Special |
|----------|-----|-----|---------|
| BP | standard, full, minimal | standard, full, minimal | clogit, softmax_gbt, cal_softmax_gbt |
| Director | standard | standard | |
| Actor Leading | standard | standard | |
| Actress Leading | standard | standard | |
| Actor Supporting | standard | standard | |
| Actress Supporting | standard | standard | |
| Original Screenplay | standard | standard | |
| Cinematography | standard | standard | |
| Animated Feature | standard | standard | |

Config structure: `{name, description, model_type, features: [feature_names]}`.
LR configs use percentile-in-year; GBT configs use raw + z-score-in-year.
Person categories include person features; film categories don't.

### 2f. All 9 Datasets Built — DONE

Via `one_offs/d20260218_build_all_datasets/`. Output: `storage/d20260218_build_all_datasets/`.

| Category | Records | Ceremonies | Metadata | Precursors | Person Data |
|----------|---------|------------|----------|------------|-------------|
| Best Picture | 206 | 72–98 | 97–100% | 100% | N/A |
| Directing | 135 | 72–98 | 99–100% | 100% | 100% career, 92% TMDb |
| Actor Leading | 135 | 72–98 | 97–100% | 100% | 100% career, 96% TMDb |
| Actress Leading | 135 | 72–98 | 97–100% | 100% | 100% career, 93% TMDb |
| Actor Supporting | 135 | 72–98 | 91–100% | 100% | 100% career, 96% TMDb |
| Actress Supporting | 135 | 72–98 | 97–100% | 100% | 100% career, 94% TMDb |
| Original Screenplay | 135 | 72–98 | 94–99% | 100% | 100% career, 4% TMDb |
| Cinematography | 135 | 72–98 | 89–100% | 100% | 100% career, 53% TMDb |
| Animated Feature | 109 | 74–98 | 85–99% | 84–100% | N/A |

Known gaps:
- **Original Screenplay TMDb enrichment = 4%** — TMDb is actor-centric; screenwriters are
  poorly covered. Person age/popularity features will be mostly null.
- **Animated BAFTA/PGA precursors = 84–87%** — these guilds started tracking animated
  features later than the Oscar category (2002).

### 2g. Modeling Pipeline — DONE (generic)

All modules are **category-agnostic**, parameterized by `--raw-path` and `--feature-config`:

| Module | Interface |
|--------|-----------|
| `data_loader.py` | `load_data(model_type, raw_path, as_of_date)` — any category dataset |
| `evaluate_cv.py` | `--raw-path` + `--feature-config` + `--model-config` + `--cv-split` |
| `train_predict.py` | `--raw-path` + `--feature-config` + `--mode {train,predict,both}` |
| `build_model.py` | End-to-end: CV → (feature selection) → train → predict. `--raw-path` parameterizes category. |
| `models.py` | LR, GBT, ConditionalLogit, SoftmaxGBT, CalibratedSoftmaxGBT |
| `cv_splitting.py` | Temporal splitters (leave-one-out, expanding/sliding window) |
| `evaluation.py` | Brier, AUC, accuracy, calibration — all category-agnostic |

### 2h. Trading Layer — DONE (mostly)

| Module | Status |
|--------|--------|
| `OscarMarket(event_ticker, nominee_tickers)` | Generic. `OSCAR_EVENT_TICKERS` for 9 categories. |
| `OscarMarket.best_picture()` factory | Convenience for BP 2026 |
| `BacktestEngine` | Category-agnostic. Runs on `dict[str, float]` predictions/prices. |
| `TradingConfig` / `BacktestConfig` | Category-agnostic strategy parameters |
| `edge.py`, `kelly.py`, `signals.py` | Abstract — no category references |
| **Non-BP nominee tickers** | **NOT DONE** — only BP 2025/2026 tickers hardcoded |
| **`discover_nominee_tickers()`** | **NOT DONE** — deferred |

### 2i. Related Completed Experiments

| One-off | Key Finding |
|---------|-------------|
| `d20260217_multinomial_modeling` | Conditional Logit matches Binary LR (65.8%) with better calibration. Softmax GBT fails (50.4%). Binary GBT best for trading (+23.5%). |
| `d20260218_build_all_datasets` | All 9 datasets built and validated. Quality ≥80% across the board. |
| `d20260219_backtest_regression` | Directional spread fix + no-prediction sell fix. Both correctness improvements. |

---

---

## 3. Remaining Work — Path to Multi-Category Arbitrage

### End Goal

For each of 9 Oscar categories: trained model → 2026 probability predictions →
compare to Kalshi market prices → identify arbitrage → execute trades across a
diversified multi-category portfolio.

### Remaining Phases (Sequential)

| Phase | What | Depends On | Deliverable |
|-------|------|------------|-------------|
| **Phase 1.5: Feature Audit** | Review per-category features; remove screenplay person features; verify precursor mappings | Configs (done) | Audited feature configs, ready for training |
| **Phase 2: Model Training** | Train LR + GBT + Clogit for all 8 non-BP categories via CV | Feature audit (done) | Per-category CV results, 2026 predictions, model comparison |
| **Phase 3: Ablation & Tuning** | Feature ablation + hyperparameter tuning per category | Phase 2 baselines | Refined feature configs, tuned models |
| **Phase 4: Trading Integration** | Nominee tickers, signal generation, backtesting for all categories | Phase 2/3 models | Trade signals, backtest results per category |
| **Phase 5: Portfolio Strategy** | Cross-category correlation, diversification, combined position sizing | Phase 4 signals | Multi-category trading strategy |

**Parallelism opportunity:** Phase 2 training across categories is embarrassingly
parallel — each category's CV is independent. Same for Phase 3 ablation. Could run
all 8 simultaneously if using multiple worktrees. But sequential within a single
worktree is simpler and avoids merge conflicts.

---

## 4. Phase 1.5: Feature Audit

Before model training, audit per-category feature configs to ensure correctness.
The initial feature configs were created programmatically from templates — need
a manual review pass.

### 4.0a. Scope

- **Remove person features from Original Screenplay configs.** TMDb enrichment
  is 4% for screenwriters — person features (age, popularity, credits) will be
  mostly null and add noise. Rely on WGA + BAFTA + CC Screenplay precursors.
- **Verify precursor mappings per category.** Confirm each category's feature
  config includes the right precursors from `CATEGORY_PRECURSORS`.
- **Check feature availability by ceremony year.** Some precursors were introduced
  later (e.g., Annie Awards started 2002). Verify features don't produce all-null
  columns for early years.
- **Review composite features.** Golden Globe genre-split composite features
  (`golden_globe_any_*`) are relevant for BP, Actor Leading, Actress Leading.
  Confirm they are NOT included in categories without genre splits.

### 4.0b. Deliverables

- Updated `original_screenplay_lr_standard.json` and `original_screenplay_gbt_standard.json`
  with person features removed
- Verified remaining 15 configs (no changes expected unless issues found)
- Brief audit notes in this plan or the one-off README

---

## 5. Phase 2: Model Training

### 5a. Scope

Train and evaluate models for 8 non-BP categories. BP already has extensive
modeling (d20260201 through d20260214 experiments, plus multinomial comparison).

**Categories:** Directing, Actor Leading, Actress Leading, Actor Supporting,
Actress Supporting, Original Screenplay, Cinematography, Animated Feature.

### 5b. Model Types per Category

Start with **Binary LR + Binary GBT** for all 8 categories.

| Model | Why include | Concern |
|-------|-----------|---------|
| Binary LR | Robust with small data, interpretable, good calibration | May underfit complex interactions |
| Binary GBT | Captures non-linearities, handles missing data | Overfitting risk with 135 records (109 for Animated) |

**Conditional Logit:** Add as a second pass for categories where calibration matters
(i.e., where we want to trade). The `d20260217_multinomial_modeling` experiment showed
Clogit matches LR for BP with better calibration and naturally-normalized probabilities.
Worth testing for Directing and acting categories — but not a first-pass priority since
it adds complexity and the current LR/GBT comparison gives a solid baseline.

**Rationale for not running 5-model comparison everywhere:** Softmax GBT fails with
~26 ceremonies of data (observed in BP experiment). With 26 ceremonies per non-BP
category (24 for Animated), it will fail worse. Cal. Softmax GBT is marginal.
Binary LR + GBT covers the useful model space for a first pass.

### 5c. CV Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CV split | Leave-one-year-out | Standard for our BP experiments |
| CV years | 2000–2025 | Full available range (2026 = prediction target) |
| Hyperparameter search | Nested CV with existing param grids | `lr_grid.json`, `gbt_grid.json` |
| Feature configs | Existing `{category}_{model_type}_standard.json` | Already created with sensible feature sets |

### 5d. Per-Category Considerations

| Category | Records | Strong Signal Expected? | Notes |
|----------|---------|------------------------|-------|
| Directing | 135 | **Yes** | DGA → Oscar Director has ~80% hit rate historically. Should be easiest non-BP category. |
| Actor Leading | 135 | **Yes** | Good TMDb enrichment (96%). SAG + BAFTA + GG acting precursors well populated. |
| Actress Leading | 135 | **Yes** | Similar to Actor. GG genre-split (Drama/Musical) handled by `CompositeSpec`. |
| Actor Supporting | 135 | **Moderate** | SAG + BAFTA + GG precursors. Lower budget/box-office coverage (91%). |
| Actress Supporting | 135 | **Moderate** | Similar to Actor Supporting. More upsets historically. |
| Original Screenplay | 135 | **Moderate** | TMDb enrichment only 4% — person features (age, popularity) will be mostly null. WGA + BAFTA + CC Screenplay precursors are the key signal. |
| Cinematography | 135 | **Moderate** | TMDb enrichment 53%. ASC + BAFTA Cinematography precursors. |
| Animated Feature | 109 | **Yes** | Only 24 ceremonies but strong structural signal: Disney/Pixar dominance, Annie Award highly predictive. LR should work well. |

### 5e. Expected Challenges

1. **Small data for GBT.** 135 records (26 ceremonies × 5 noms) is thin. GBT with
   tree-based splits may overfit. Mitigation: strong regularization in `gbt_grid.json`
   (low `max_depth`, high `min_samples_leaf`, `learning_rate` ≤ 0.1).

2. **Animated Feature = 109 records.** Even thinner. LR may be the only viable model.
   This is fine — precursor signal is strong enough for a simple model.

3. **Screenplay TMDb gap.** Person features will be mostly null for Original Screenplay.
   **Decision: remove person features from screenplay configs** (Phase 1.5 audit task).
   Rely on WGA + BAFTA + CC Screenplay precursors as the primary signal.

4. **GG acting genre split.** For Actor/Actress Leading, the Golden Globe splits by
   Drama vs Musical/Comedy. `CompositeSpec` already creates combined
   `golden_globe_any_*` features. Need to verify these work correctly in CV.

### 5f. Deliverables

Per category:
- CV metrics (Brier score, accuracy, top-1 accuracy, AUC)
- Feature importance analysis
- 2026 probability predictions (train on 2000–2025, predict 2026)
- Model comparison (LR vs GBT vs Clogit)

Summary:
- Cross-category model comparison table
- Identified categories with strong vs weak predictive signal
- Recommendation on which categories are worth trading

### 5g. One-Off Structure

Reuse the existing `d20260217_multi_category_expansion/` one-off for model training
experiments (it's the umbrella for all multi-category work).

```
one_offs/d20260217_multi_category_expansion/
├── PLAN.md             # This document
├── README.md           # Results + findings
├── run.sh              # Orchestrates all 8 categories × 3 model types
├── run_category.sh     # Per-category: CV + train + predict
├── analyze_results.py  # Cross-category comparison table + plots
└── assets/             # Plots
```

```
storage/d20260217_multi_category_expansion/
├── directing/
│   ├── lr_cv/          # CV output
│   ├── gbt_cv/
│   ├── clogit_cv/
│   ├── lr_train/       # Final model + 2026 predictions
│   ├── gbt_train/
│   └── clogit_train/
├── actor_leading/
│   └── ...
├── ... (8 category dirs)
└── summary/
    ├── cross_category_comparison.json
    └── cross_category_comparison.csv
```

---

## 6. Phase 3: Ablation & Tuning

### 6a. When to Do This

After Phase 2 produces baseline results. Run ablation for **all 8 categories**
regardless of volume — even low-volume categories (Screenplay $463K, Animated $306K)
deserve tuning since precursor signal may be very strong and the analysis is
informative regardless of trading intent.

**Priority ordering by market volume** (for sequencing, not exclusion):
1. Actor ($5.5M) → Supporting Actor ($2.0M) → Director ($1.5M) — high volume
2. Supporting Actress ($1.5M) → Actress ($1.3M) → Cinematography ($1.2M) — medium
3. Screenplay ($463K) → Animated ($306K) — low volume but still worth tuning

### 6b. Feature Ablation

Same methodology as `d20260213_feature_ablation` (which was done for BP):
- Start from `{category}_{model_type}_standard.json` (all plausible features)
- Ablate features one-at-a-time to measure marginal contribution
- Generate refined `{category}_{model_type}_ablated.json` config

**Key question per category:** How much do precursor features dominate vs
film metadata vs person career? This tells us what signal we're actually using.

### 6c. Hyperparameter Tuning

Same methodology as `d20260206_hyperparameter_tuning`:
- Nested CV with existing param grids
- Focus on regularization strength (especially for GBT with small data)

### 6d. Parallel Opportunity

Feature ablation across categories is embarrassingly parallel — each category's
ablation is independent. Could run all 8 simultaneously with separate shell
processes. The run scripts just need different `--raw-path` and `--feature-config`.

---

## 7. Phase 4: Trading Integration

### 7a. Nominee Ticker Mapping

Need to map model predictions (nominee names) → Kalshi tickers for 8 new categories.

**Options:**
1. **Hardcode 2026 tickers** — ~40 tickers total (5 per category × 8 categories).
   Quick, works for 2026. Need to update annually.
2. **Build `discover_nominee_tickers()`** — query Kalshi API `custom_strike` field.
   More robust, works across years.

**Rec:** Start with (1) for speed. We only need 2026. Build (2) as improvement later.

### 7b. Signal Generation

For each category: load trained model → load current Kalshi prices → compute edge →
Kelly sizing → trade signals. Same pipeline as BP (`d20260214_trade_signal_ablation`).

**Adaptation needed:**
- Signal generation scripts from `d20260214_trade_signal_ablation` reference BP-specific
  paths — need parameterization by category
- `model_snapshots.py` is BP-focused for temporal snapshots — may need generalization
  or a new multi-category version

### 7c. Per-Category Backtesting

Run the `BacktestEngine` for each category using historical Kalshi prices.
The backtest engine is already category-agnostic.

**Challenge:** Do we have historical Kalshi prices for non-BP categories?
Kalshi may only have 2026 data for new Oscar categories. If so, backtesting
options are:
1. **Hold-to-settlement only** — single snapshot, no temporal trading
2. **Use available price history** — even a few weeks of prices enables
   temporal simulation
3. **Skip backtest, go straight to live signals** — use CV metrics as our
   confidence measure instead of backtest P&L

### 7d. Deliverables

Per category:
- Edge analysis (model prob vs market price for each nominee)
- Trade signals (buy/sell recommendations with Kelly sizing)
- Backtest results (if historical prices available)

Aggregate:
- Multi-category signal dashboard
- Total portfolio exposure across categories

---

## 8. Phase 5: Portfolio Strategy

### 8a. Cross-Category Outcome Correlation

Key question: How correlated are outcomes across categories? This determines
the diversification benefit of multi-category trading.

| Correlation Type | Example | Expected | Implication |
|-----------------|---------|----------|-------------|
| Same film, different categories | "Sinners" wins BP AND Director | High positive | Correlated bets — don't double-count edge |
| Film vs acting categories | BP winner's actors win acting? | Moderate positive | Some signal overlap |
| Acting categories (different genders) | Lead Actor and Lead Actress | Low | Independent alpha |
| Different films | Animated vs BP | Near zero | True diversification |

**Analysis plan:** For each historical year in training data (2000–2025), compute
which categories' winners came from the same film. Build an outcome correlation
matrix. This directly informs position sizing.

### 8b. Position Sizing Across Categories

Options:
1. **Independent Kelly per category** — treat each category as a separate bet.
   Simple but ignores correlations (may over-allocate to correlated bets).
2. **Portfolio-level Kelly** — joint optimization across all categories, accounting
   for correlations. More sophisticated, harder to implement.
3. **Category-weighted allocation** — fixed allocation by market volume or by model
   confidence. Simple heuristic.

**Rec:** Start with (1) — independent Kelly per category with a global bankroll
constraint (e.g., max allocation caps per category). Use correlation analysis from
7a to flag when we're overweight on correlated bets — e.g., if we're buying the
same film in both BP and Director, reduce position in the smaller market.

### 8c. Volume-Weighted Allocation Caps

Categories with more volume have tighter spreads and better liquidity.
Suggested max allocation per category:

| Category | Volume | Max Allocation | Rationale |
|----------|--------|----------------|-----------|
| BP ($10M) | High | 30% | Most liquid, most studied |
| Actor ($5.5M) | High | 20% | Second most liquid |
| Sup. Actor ($2M) | Medium | 10% | |
| Director ($1.5M) | Medium | 10% | |
| Sup. Actress ($1.5M) | Medium | 10% | |
| Actress ($1.3M) | Medium | 10% | |
| Cinematography ($1.2M) | Low | 5% | |
| Screenplay ($463K) | Low | 3% | Low liquidity, may not fill |
| Animated ($306K) | Low | 2% | Very low liquidity |

These are starting points — actual allocation depends on edge size and model quality
per category.

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GBT overfits on 135 records | Poor OOS predictions | Strong regularization; accept LR as primary for small-data categories |
| Animated Feature has only 24 ceremonies | Model may not generalize | LR with precursor features; accept lower confidence interval |
| Screenplay TMDb enrichment = 4% | Person features are noise | Remove person features from screenplay config; rely on WGA/BAFTA/CC precursors |
| No historical Kalshi prices for non-BP | Can't run temporal backtests | Hold-to-settlement backtest or skip; use CV metrics as confidence measure |
| Cross-category correlation reduces diversification | Portfolio benefit overstated | Measure correlations before allocating; reduce allocation to correlated categories |
| Nominee name mismatch (model → Kalshi) | Trades go to wrong nominees | Manual verification of nominee-ticker mapping for all 40 tickers |
| Low volume categories ($300K–$500K) | Can't fill orders, wide spreads | Cap allocation at 2–3%; use maker orders only |
| Model quality varies wildly across categories | Some categories untradeable | Accept it — only trade categories with positive CV results |

---

## 10. Open Questions

1. **Data pooling across acting categories.** All 4 acting categories have similar
   structure (person-level features, similar precursors). Pooling (520 records instead
   of 135) could help GBT generalize. But it conflates lead vs supporting and actor vs
   actress dynamics. Worth an experiment after Phase 2 baselines — try later if
   per-category models underperform.

2. **Model update frequency.** Depends on how often new precursor features become
   available. The `as_of_date` parameter supports incremental feature availability
   (e.g., SAG announced before BAFTA). Update predictions when a new precursor
   award is announced — not on a fixed schedule.

### Resolved Questions

- **Ceremony date:** ~Mid-March 2026. Not imminent but want predictions well before.
- **Skip Screenplay/Animated?** No — model and tune all 9 categories regardless of volume.
- **Clogit everywhere?** Yes — run Clogit as a Phase 2 baseline alongside LR + GBT.
  Low marginal cost, valuable for selecting production model per category.
- **Screenplay person features?** Remove — TMDb enrichment is 4%, adds noise. Done in
  Phase 1.5 feature audit.

---

## 11. Historical Context — Phase 0 & Phase 1 Implementation Notes

### What went smoothly

- **Feature output names stayed stable.** Feature configs (JSON listing output column names)
  needed only filename renames — not content changes. Transform functions read from nested
  schema (`r.precursors.pga.bp.winner`) but output flat column names (`pga_winner`).

- **Generic trading modules needed zero changes.** `edge.py`, `kelly.py`, `signals.py`,
  `types.py` all operated on abstract types.

- **Nested precursor schema was worth doing early.** The plan originally deferred nesting to
  Phase 2, but we implemented it in Phase 0. This made `CATEGORY_PRECURSORS` + `PrecursorSpec`
  much cleaner — each spec references `(org, sub)` instead of flat field names.

- **Shared intermediates saved time.** Building metadata and precursors once for 585 unique
  films across all 9 categories avoided ~9× redundant API calls.

### Gotchas encountered

- **Mypy vs Pydantic `Field(None, ...)`:** Use plain `= None` for optional fields with no
  validation constraints. Only use `Field(...)` when you need `ge=`, `min_length=`, etc.

- **Golden Globe split was the largest single complication.** Split into drama/musical for
  BP + lead acting. Required `CompositeSpec` and `_any_true()` helper.

- **One-off files are the long tail.** Schema refactor touched 6 core files predictably,
  then required propagation to ~9 one-off files. Grep-and-replace workflow was essential.

- **Screenplay name cleaning.** oscars.csv stores credits as "Written by PERSON" or
  "Screenplay by PERSON, PERSON". Needed `clean_screenplay_names()` to extract just names.

### What we'd do differently

- **Do a dry-run grep BEFORE designing.** Start by grepping every old type reference to
  build the full dependency graph → plan edit order with confidence.

- **Commit incrementally.** The entire Phase 0 was one commit. Logical commits (schema →
  core → FE → one-offs → configs) would make review easier.

- **`data_completeness_report.py` should be parameterized.** Currently BP-specific. Should
  accept `--category` and check the relevant precursor specs per category.

---

## 12. Appendix: Completed File Changes (Phase 0 + Phase 1-Data)

### Files modified

| File | Nature of change |
|------|-----------------|
| `data/schema.py` | Full redesign: nested precursor models, category enums, person data, calendar expansion |
| `data/build_dataset.py` | `--category` arg, per-category precursor mapping, person data pipeline, shared-dir support |
| `data/fetch_precursor_awards.py` | 36 AWARD_URLS, generic parser, person-level extraction, PRECURSOR_YEAR_INTRODUCED |
| `modeling/feature_engineering.py` | `NominationRecord`-based transforms, person features, animated features, dynamic precursor features |
| `modeling/data_loader.py` | `category_winner` target, `raw_path` parameterization |
| `trading/oscar_market.py` | `OscarMarket` generic class, `OSCAR_EVENT_TICKERS` for 9 categories |
| `modeling/configs/features/` | 8 BP configs renamed with `bp_` prefix; 17 new category configs created |
| 9 existing one-off files | Import/type updates for new schema |

### Files unchanged (already generic)

`trading/edge.py`, `trading/kelly.py`, `trading/signals.py`, `trading/types.py`,
`trading/kalshi_client.py`, `modeling/models.py`, `modeling/cv_splitting.py`,
`modeling/evaluation.py`, `modeling/evaluate_cv.py`, `modeling/train_predict.py`,
`data/fetch_omdb.py`, `data/fetch_tmdb.py`
