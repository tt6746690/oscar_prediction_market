# Feature Ablation & Stability Analysis

**Storage:** `storage/d20260213_feature_ablation/`, `storage/d20260213_lr_feature_ablation/`

Two-phase investigation: (1) diagnose and fix LR instability via regularization grid
redesign (Feb 13), (2) systematic feature group ablation and temporal stability testing
to find the optimal feature subset (Feb 13-14).

## Motivation

LR predictions were wildly inconsistent: P(One Battle after Another) ranged from 10.5%
(counterfactual baseline) to 30.9% (temporal snapshots) to 72.8% (hand-tuned Feb 7 model).
Root cause: the old LR param grid includes `l1_ratio=0.0` (pure L2/Ridge) and high C
values (0.05-1.0), always selecting pure Ridge and retaining 25-43 features on ~10
effective training examples.

## Summary

**Best LR config:** `lr_standard` (25 features) with wide grid + importance threshold 0.80
- Accuracy: 80.8%, Brier: 0.0502, LOYO Jaccard: 1.000

**Best GBT config:** `gbt_standard` (17 features)
- Accuracy: 76.9%, Brier: 0.0523, LOYO Jaccard: 0.991

Both use the **3-group subset**: `precursor_winners` + `precursor_nominations` + `oscar_nominations`

## Default Configs (modeling/configs/)

After ablation, the source configs were updated to the best-performing subsets:

| Config | Features | Description |
|--------|----------|-------------|
| `lr_standard.json` | 25 | Best LR subset (3 groups, FULL features with interactions) |
| `gbt_standard.json` | 17 | Best GBT subset (3 groups, BASE features without interactions) |
| `lr_full.json` | 47 | All LR features (for ablation baselines) |
| `gbt_full.json` | 45 | All GBT features (for ablation baselines) |
| `lr_minimal.json` | 14 | Minimal subset |
| `gbt_minimal.json` | 11 | Minimal subset |
| `lr_grid.json` | — | Wide regularization grid (72 combos) |
| `gbt_grid.json` | — | GBT hyperparameter grid (36 combos) |

## Experiments

### 1. Single-date ablation (`run_single_date.sh`)

23 configs at as-of 2026-02-07:
- **Group 1:** Grid comparison (wide, sparse, moderate)
- **Group 2:** Sparse grid + pipeline stages (thresh, max, VIF)
- **Group 3:** Wide grid + individual filters
- **Group 4:** GBT baseline vs interactions
- **Group 5:** Additive subset validation (FULL vs BASE)

### 2. Temporal stability (`run_temporal.sh`)

10 configs × 11 snapshot dates = 110 runs

Tests prediction stability as precursor ceremonies occur:
- Dates: 2025-11-30 through 2026-02-07
- Metrics: temporal jitter, OBaA prediction deltas

### 3. Group ablation (`run_group_ablation.sh`)

100 runs: 2 modes × 2 models × 25 configs

Feature group ablation types:
- **Additive:** Start with precursor_winners, add groups incrementally
- **Leave-one-out:** Start full, remove one group at a time
- **Single-group:** Each group in isolation

Feature groups (8):
1. `precursor_winners` — DGA, PGA, SAG, BAFTA, GG, CC winner flags
2. `precursor_nominations` — Nomination flags + counts
3. `oscar_nominations` — Total noms, major category flags, acting count
4. `critic_scores` — Metacritic, RT, IMDb ratings
5. `commercial` — Budget, box office
6. `timing` — Release month, awards season flag
7. `film_metadata` — Runtime, genre flags
8. `voting_system` — Preferential voting era indicator

## Findings

### LR instability root cause and fix (Feb 13)

| Experiment | Grid | Features | P(One Battle) |
|------------|------|----------|---------------|
| Feb 7 original (hand-picked) | — | 8 | 72.8% |
| Feb 11 temporal snapshot | old | 43 | 30.9% |
| Feb 12 counterfactual (baseline) | old | ~25 | 10.5% |
| Feb 13 ablation A (old grid) | old | 25 | 27.0% |

**Fix — tighter param grid (`lr_grid_v2.json`):**
- C: [0.005, 0.01, 0.02, 0.05, 0.1] (was [0.05, 0.1, 0.2, 0.5, 1.0])
- l1_ratio: [0.5, 0.8, 1.0] — no pure L2 (was [0.0, 0.5, 1.0])
- 30 grid points (was 60)

**Selected features (new grid):** `precursor_wins_count`, `dga_winner`,
`sag_ensemble_nominee`, `critics_choice_winner`, `has_editing_nomination`

New grid always selects C=0.1, l1_ratio=0.8 → 5 features. All post-hoc filters are
no-ops with the new grid. Old grid always selects l1_ratio=0.0 (pure L2/Ridge)
regardless of filtering — the grid search never benefits from L1 sparsity when the
search space includes pure L2.

**Group 1: Grid comparison + Group 2: Old grid + individual filters:**

| Config | Grid | Filter | Feat | Brier | Best C | Best l1 | P(OBaA) |
|--------|------|--------|------|-------|--------|---------|---------|
| lr_a (old) | old | none | 25 | 0.0561 | 0.1 | 0.0 | 27.0% |
| lr_b (new) | new | none | 5 | 0.0652 | 0.1 | 0.8 | 24.2% |
| lr_c-f (new+filters) | new | various | 5 | 0.0652 | 0.1 | 0.8 | 24.2% |
| lr_g (old+thresh95) | old | thresh 0.95 | 20 | 0.0525 | 0.1 | 0.0 | 26.4% |
| lr_h (old+thresh90) | old | thresh 0.90 | 17 | 0.0517 | 0.1 | 0.0 | 26.5% |
| lr_i (old+thresh80) | old | thresh 0.80 | 13 | **0.0496** | 0.2 | 0.0 | 25.0% |
| lr_j (old+max10) | old | max 10 | 10 | 0.0509 | 0.2 | 0.0 | 28.1% |
| lr_k (old+max15) | old | max 15 | 15 | 0.0518 | 0.1 | 0.0 | 26.8% |
| lr_m (old+vif5) | old | VIF 5 | 20 | 0.0653 | 1.0 | 0.0 | 20.3% |
| lr_o (old+full pipeline) | old | t.90+m15+V5 | 12 | 0.0526 | 0.2 | 0.0 | 38.0% |
| GBT baseline | — | — | 12 | 0.0566 | — | — | 15.8% |

VIF filtering is counterproductive: it removes correlated features but not necessarily
the least important ones. lr_m (VIF=5) jumps to C=1.0 (almost unregularized), worst
among old-grid configs.

**Summary used in subsequent sections:**

| Config | Grid | Filter | Feat | Brier | P(OBaA) |
|--------|------|--------|------|-------|---------|
| lr_a (old) | old | none | 25 | 0.0561 | 27.0% |
| lr_b (new) | new | none | 5 | 0.0652 | 24.2% |
| lr_i (old+thresh80) | old | thresh 0.80 | 13 | **0.0496** | 25.0% |
| lr_j (old+max10) | old | max 10 | 10 | 0.0509 | 28.1% |
| lr_o (old+full pipeline) | old | t.90+m15+V5 | 12 | 0.0526 | 38.0% |
| GBT baseline | — | — | 12 | 0.0566 | 15.8% |

New grid trades Brier (0.0652 vs 0.0496 for old+filter) for simplicity and stability.
The old full pipeline P(OBaA)=38.0% illustrates instability — it happens to weight
One Battle higher in this run. VIF filtering is counterproductive (removes by
correlation, not predictive value; model compensates by jumping to C=1.0).

### Expanding 29→47 features hurts uniformly (Feb 14)

Expanded feature set from 29→47 (LR) / 45 (GBT) by adding:
- **Percentile-in-year** (6 new): nominations, metacritic, RT, IMDb, box office, budget
- **Z-score-in-year** (7 new, GBT): metacritic, RT, IMDb, box office, budget, runtime
- **Log transforms** (2 new, LR): log_budget, log_box_office_worldwide
- **Interaction features** (3 new, GBT): precursor_wins_count, precursor_nominations_count,
  has_pga_dga_combo

New moderate grid: C=[0.02, 0.05, 0.1, 0.2] × l1_ratio=[0.3, 0.5, 0.7] ×
class_weight=[balanced, null] (24 points).

**Single-date results (as-of 2026-02-07, 47/45 features, feature selection):**

| Config | Grid | Filter | Feat | Brier | Acc | P(OBaA) |
|--------|------|--------|------|-------|-----|---------||
| lr_wide | wide | none | 43 | 0.0586 | 73.1% | 30.9% |
| lr_sparse | sparse | none | 6 | 0.0661 | 69.2% | 25.1% |
| lr_moderate | moderate | none | 8 | 0.0555 | 73.1% | 22.6% |
| lr_wide_thresh80 | wide | thresh 0.80 | 19 | 0.0531 | 73.1% | 28.8% |
| lr_wide_max10 | wide | max 10 | 10 | 0.0530 | 76.9% | 25.7% |
| lr_wide_max15 | wide | max 15 | 15 | **0.0529** | 76.9% | 27.0% |
| lr_wide_vif5 | wide | VIF 5 | 34 | 0.0638 | 73.1% | 37.9% |
| gbt_baseline | — | — | 21 | 0.0622 | 76.9% | 12.1% |
| gbt_interactions | — | — | 24 | 0.0669 | 69.2% | 7.1% |

**Comparison with Feb 13 (29→47 features):**

| Config | Feb 13 feat | Feb 13 Brier | Feb 14 feat | Feb 14 Brier | Delta |
|--------|------------|-------------|------------|-------------|-------|
| wide, no filter | 25 | 0.0561 | 43 | 0.0586 | +0.0025 |
| wide + thresh80 | 13 | **0.0496** | 19 | 0.0531 | +0.0035 |
| wide + max10 | 10 | 0.0509 | 10 | 0.0530 | +0.0021 |
| sparse, no filter | 5 | 0.0652 | 6 | 0.0661 | +0.0009 |
| GBT baseline | 12 | 0.0566 | 21 | 0.0622 | +0.0056 |

Expanding from 29→47 features consistently degrades Brier by 2-6 points. The
additional percentile/zscore features are mostly noise for LR (which standardizes
internally) and dilute the importance rankings. GBT suffers worse (+0.0056) because
z-score features add redundant information that the tree splits on unnecessarily.

**GBT interaction features hurt:** Adding precursor_wins_count, precursor_nominations_count,
and has_pga_dga_combo to GBT (42→45 features) worsens Brier from 0.0622 to 0.0669 and
accuracy from 76.9% to 69.2%. These are engineered aggregates that LR needs (for linear
combination) but GBT can derive from the raw indicator features directly.

### Group ablation: precursor winners dominate

LR leave-one-out (no feature selection, 47 features):

| Removed group | Brier | dBrier |
|---------------|-------|--------|
| precursor_winners | 0.0827 | **+0.0241** |
| precursor_nominations | 0.0615 | +0.0029 |
| oscar_nominations | 0.0599 | +0.0013 |
| film_metadata | 0.0573 | **−0.0013** |
| timing | 0.0575 | **−0.0011** |
| critic_scores | 0.0586 | 0.0000 |

Precursor winners are overwhelmingly dominant. Removing film_metadata or timing
*improves* performance — they're pure noise.

**GBT leave-one-out (no feature selection):**

| Removed group | Brier | dBrier |
|---------------|-------|--------|
| precursor_winners | 0.1114 | **+0.0430** |
| precursor_nominations | 0.0744 | +0.0060 |
| oscar_nominations | 0.0721 | +0.0037 |
| timing | 0.0656 | **−0.0028** |
| commercial | 0.0682 | −0.0002 |

Same hierarchy: precursor_winners >> everything else. Removing timing helps GBT too.

Best additive combinations (with feature selection, thresh 0.80):

| Groups | Feat | Brier | Acc |
|--------|------|-------|-----|
| precursors + oscar_noms (**additive_3**) | 10 | **0.0502** | **80.8%** |
| + critic + commercial (additive_5) | 14 | 0.0526 | 76.9% |
| all 8 groups (full) | 19 | 0.0531 | 73.1% |
| precursor_winners only | 2 | 0.0574 | 76.9% |

### FULL vs BASE features diverge by model type

| Config | Feat in | Feat sel | Brier | Acc | LOYO Jaccard |
|--------|---------|----------|-------|-----|-------------|
| lr_add3_full_wide_t80 | 25 | 10 | **0.0502** | **80.8%** | **1.000** |
| lr_add3_base_wide_t80 | 22 | 10 | 0.0513 | 76.9% | **1.000** |
| gbt_add3_base | 17 | 9 | **0.0523** | **76.9%** | **0.991** |
| gbt_add3_full | 23 | 15 | 0.0643 | 69.2% | 0.855 |
| gbt_add4_full | 29 | 19 | 0.0621 | 69.2% | 0.942 |

LR prefers FULL (interaction features help linear combination). GBT strongly prefers
BASE (interaction features dilute selection, dropping LOYO Jaccard 0.991→0.855).

Adding critic_scores to GBT (additive_4) doesn't help: GBT additive_4 (0.0621 Brier,
69.2% acc) is worse than GBT additive_3 BASE (0.0523, 76.9%). The critic_scores
features add 10 more features with LOYO Jaccard 0.942. The Feb 7 single-date result
showed additive_4 helping GBT — but that was without feature selection and with fewer
base features.

### LOYO Jaccard stability (nonzero feature overlap across 26 folds)

| Config | Jaccard | Feat range |
|--------|---------|------------|
| lr_wide (all variants) | **1.000** | constant per config |
| lr_moderate | **1.000** | 8-8 |
| lr_sparse | 0.940 | 5-6 |
| gbt_interactions | 0.990 | 22-24 |
| gbt_baseline | 0.854 | 15-20 |

Wide grid LR gets perfect Jaccard because it always picks l1_ratio=0.0 (ridge),
retaining all features in every fold. Moderate grid also achieves perfect stability
with only 8 features. GBT baseline has the lowest stability (0.854) — tree-based
importance varies more with training data composition.

**Why GBT has low LOYO Jaccard despite ensembling:** GBT baseline Jaccard=0.854 (feature
range 15-20) because ~5 tail features toggle between zero and nonzero importance across
folds. The top ~8 features (`dga_winner`, `critics_choice_winner`, `metacritic`,
`imdb_zscore_in_year`, `sag_ensemble_nominee`, `has_editing_nomination`,
`metacritic_zscore_in_year`, `budget_zscore_in_year`) are always nonzero. But
`major_category_count`, `nominees_in_year`, `acting_nomination_count`, `golden_globe_winner`,
`release_month` oscillate. GBT ensembles trees **sequentially** (boosting), not
independently (bagging). Each tree fits residuals of the previous one, so a tail feature
that captures a small residual pattern gets zero importance when that pattern is absent.
A Random Forest would give more stable importance via independent trees + averaging.

### Temporal stability: additive_3 reduces LR jitter by 25%

**Full temporal stability (jitter across 11 snapshots, 47/45 features):**

| Config | Feat (mean) | Jitter | OBaA jitter | Hamnet jitter |
|--------|-------------|--------|-------------|---------------|
| lr_sparse | 4 | **0.0224** | 0.037 | 0.035 |
| lr_moderate | 7 | **0.0224** | 0.024 | 0.035 |
| lr_wide_thresh80 | 5 | 0.0240 | 0.050 | 0.044 |
| gbt_baseline | 11 | 0.0264 | 0.126 | 0.014 |
| lr_wide_max10 | 7 | 0.0287 | 0.051 | 0.041 |
| gbt_interactions | 15 | 0.0320 | 0.062 | 0.066 |

GBT baseline has moderate aggregate jitter but extreme OBaA jitter (0.126) — its
prediction for One Battle swings ±12.6pp between consecutive snapshots.

**Additive_3 temporal stability comparison:**

| Config | Snaps | Jitter | Feat (mean) | OBaA jitter | Hamnet jitter |
|--------|-------|--------|-------------|-------------|---------------|
| lr_add3_moderate | 10 | **0.0168** | 5.4 | 0.032 | 0.027 |
| lr_add3_wide_t80 | 9 | 0.0230 | 4.0 | 0.055 | 0.042 |
| gbt_add3 | 10 | 0.0272 | 6.3 | 0.059 | 0.048 |
| gbt_add4 | 11 | 0.0491 | 9.4 | 0.067 | 0.117 |

| Config | Previous jitter | additive_3 jitter | Delta |
|--------|----------------|-------------------|-------|
| lr_moderate | 0.0224 | **0.0168** | −25% |
| lr_wide_thresh80 | 0.0240 | 0.0230 | −4% |
| gbt_baseline | 0.0264 | 0.0272 | +3% |

Fewer features = less noise to amplify across snapshots. GBT is neutral because
its instability is from boosting mechanics, not feature count.

GBT additive_4 (+ critic_scores) has the worst temporal jitter tested (0.0491) —
Hamnet jitters ±11.7pp between snapshots.

### Temporal stability vs LOYO Jaccard — why they diverge

| | Temporal stability | LOYO Jaccard |
|---|---|---|
| What changes | Available features (precursor ceremonies over time) | Training data (entire year removed) |
| Training set | Always 2000-2025, same data | Missing one year, different data |
| Feature pool | Grows: 4 features (Nov) → 21 features (Feb) | Always the same candidate pool |
| Measures | Prediction smoothness as info arrives | Feature selection robustness to data perturbation |

GBT has decent temporal stability because the training data is always the same 26 years —
adding a new precursor feature shifts predictions incrementally. But LOYO removes an entire
year (one of ~10 positive examples), which can dramatically shift which tail features get
importance.

**For trading, temporal stability matters more.** We need predictions that don't whipsaw
when new ceremonies are announced — that drives trading decisions. LOYO Jaccard is a model
robustness diagnostic: low Jaccard warns the model may be fitting noise, but doesn't
directly cause trading losses. Use LOYO Jaccard as a tiebreaker and red flag, not the
primary selection criterion.

### Feature subset recommendation

| Group | Include? | Evidence |
|-------|----------|----------|
| **precursor_winners** | **Yes — core** | Removing: +0.0241 LR, +0.0430 GBT |
| **precursor_nominations** | **Yes — core** | Additive: 0.0574→0.0547 (LR) |
| **oscar_nominations** | **Yes — core** | additive_3 = best: 0.0502, 80.8% |
| critic_scores | No | Destabilizes GBT temporally (0.0491 jitter) |
| commercial | No | No improvement in additive_5 over additive_3 |
| timing | No | Removing *improves* both LR and GBT |
| film_metadata | No | Removing *improves* LR |
| voting_system | No | Zero impact (dBrier = 0.0000) |

### Best overall configs

| | Best LR | Best GBT |
|--|---------|----------|
| Config | additive_3 FULL, wide + thresh 0.80 | additive_3 BASE |
| Features in | 25 | 17 |
| Features selected | 10 | 9 |
| Accuracy | 80.8% | 76.9% |
| Brier | 0.0502 | 0.0523 |
| LOYO Jaccard | 1.000 | 0.991 |

### Detailed takeaways

1. **Expanding 29→47 features hurts uniformly.** The new percentile/zscore features
   add noise across all configs. Best full-feature Brier degrades from 0.0496→0.0529.
2. **Three groups are sufficient.** Precursor winners + precursor noms + oscar noms (10
   features with feature selection) achieves the best result: 0.0502 Brier, 80.8% accuracy.
3. **FULL vs BASE diverges by model type.** LR prefers FULL additive_3 (0.0502 vs 0.0513).
   GBT strongly prefers BASE (0.0523 vs 0.0643). Interaction/aggregation features in FULL
   hurt GBT's feature selection (15 vs 9 features, LOYO Jaccard 0.855 vs 0.991).
4. **Additive_3 improves LR temporal stability by 25%.** Moderate grid drops from 0.0224
   to 0.0168 jitter with the restricted feature set.
5. **Critic scores are a trap for GBT.** additive_4 gives the worst temporal jitter (0.0491),
   especially Hamnet (0.117 jitter). Single-date results don't survive temporal validation.
6. **GBT interaction features are counterproductive.** precursor_wins_count/count/combo
   worsen GBT from 0.0622→0.0669.
7. **Film metadata and voting system are pure noise.** Removing either improves both models.
8. **Wide grid LR achieves perfect fold stability via density, not parsimony.** Jaccard=1.0
   because Ridge retains all features. Moderate grid also gets perfect stability with only
   8 features — the better approach.

**Recommendation:** LR additive_3 FULL with wide+thresh80 is the best overall model
(80.8% acc, 0.0502 Brier, LOYO Jaccard 1.000). For GBT, use additive_3 BASE (76.9%,
0.0523, Jaccard 0.991). Drop critic_scores, commercial, timing, film_metadata, and
voting_system from all future experiments.

## Storage Structure

```
storage/d20260213_feature_ablation/
├── configs/                    # Experiment configs (copied from source)
│   ├── features/               # 57 feature configs (base + ablation)
│   ├── param_grids/            # 4 grids (lr_wide, lr_sparse, lr_moderate, gbt)
│   └── cv_splits/              # 6 CV strategies
├── single_date/                # 23 single-date experiments
├── temporal/                   # 10 configs × 11 dates
├── group_ablation/             # 100 ablation runs
│   ├── no_fs/                  # Without feature selection
│   └── with_fs/                # With feature selection
└── stability_metrics.csv       # Summary metrics for all configs
```

## Usage

```bash
cd "$(git rev-parse --show-toplevel)"

# Generate configs (idempotent)
bash oscar_prediction_market/one_offs/d20260213_feature_ablation/run_generate_configs.sh

# Run experiments
bash .../run_single_date.sh 2>&1 | tee storage/d20260213_feature_ablation/single_date/run.log
bash .../run_temporal.sh 2>&1 | tee storage/d20260213_feature_ablation/temporal/run.log
bash .../run_group_ablation.sh 2>&1 | tee storage/d20260213_feature_ablation/group_ablation/run.log

# Analyze results
uv run python -m oscar_prediction_market.one_offs.d20260213_feature_ablation.analyze_results \
    --exp-dir storage/d20260213_feature_ablation --section all
```

## Related

- [modeling/README.md](../../modeling/README.md) — Model pipeline documentation
- [d20260207_feature_ablation/](../d20260207_feature_ablation/) — Earlier Feb 5+7 feature ablation studies
