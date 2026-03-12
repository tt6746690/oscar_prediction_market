# Oscar Prediction — Modeling

ML pipeline for predicting Oscar winners across 9 categories using temporal
cross-validation.

## Module Overview

```
modeling/
├── data_loader.py                      # Load raw JSON, apply features, prepare train/test splits
├── feature_engineering/                # Feature definitions, transforms, availability dates (package)
│   ├── engine.py                       # Feature computation engine
│   ├── features.py                     # Individual feature definitions
│   ├── groups.py                       # Feature group registry
│   ├── helpers.py                      # Shared helpers for feature computation
│   ├── registry.py                     # Feature registry and lookup
│   ├── transforms.py                   # Feature transforms (log, cyclical, etc.)
│   └── types.py                        # Feature type definitions
├── models/                             # Model wrappers with unified interface (package)
│   ├── base.py                         # Abstract base model
│   ├── lr.py                           # Logistic Regression (incl. Conditional Logit)
│   ├── gbt.py                          # Gradient Boosting (incl. Calibrated Softmax GBT)
│   ├── bagged.py                       # Bagged ensemble wrapper
│   ├── configs.py                      # Model config definitions
│   ├── loading.py                      # Load model from config
│   ├── persistence.py                  # Model save/load
│   ├── registry.py                     # Model type registry
│   └── types.py                        # Model type definitions
├── calibration.py                      # Probability calibration (temperature-scaled softmax)
├── cv_splitting.py                     # CV strategies: expanding, LOYO, sliding window, bootstrap
├── evaluation.py                       # Discrimination + calibration metrics (micro/macro)
├── prediction_io.py                    # Load raw model predictions from disk
├── hyperparameter_tuning.py            # Nested and non-nested CV tuning with combined scoring
├── evaluate_cv.py                  [CLI] Cross-validation evaluation (simple or nested)
├── train_predict.py                [CLI] Train models and/or generate predictions
├── build_model.py              [CLI] End-to-end pipeline: CV → feature selection → train → predict
├── generate_tuning_configs.py      [CLI] Generate hyperparameter grid JSONs
├── generate_feature_ablation_configs.py [CLI] Generate feature ablation experiment configs
├── utils.py                            # Path resolution, year/ceremony conversion
└── configs/                            # JSON configs for models, features, CV splits, param grids
```

## Quick Start

```bash
MODULE=oscar_prediction_market.modeling

# Cross-validation evaluation
uv run python -m $MODULE.evaluate_cv \
    --model-config configs/models/logistic_regression.json \
    --feature-config configs/features/lr_standard.json \
    --cv-split configs/cv_splits/expanding_window.json \
    --name lr_standard --output-dir /path/to/output

# Train a model and predict
uv run python -m $MODULE.train_predict \
    --mode both \
    --model-config configs/models/gradient_boosting.json \
    --feature-config configs/features/gbt_standard.json \
    --train-years 2000-2025 --test-years 2026 \
    --output-dir /path/to/output

# Or reuse best config from a CV run
uv run python -m $MODULE.train_predict \
    --mode both --cv-output /path/to/cv_output \
    --train-years 2000-2025 --test-years 2026 \
    --output-dir /path/to/output

# End-to-end pipeline: CV → feature selection → train → predict
uv run python -m $MODULE.build_model \
    --name lr_baseline \
    --param-grid configs/param_grids/lr_grid.json \
    --feature-config configs/features/lr_baseline.json \
    --cv-split configs/cv_splits/leave_one_year_out.json \
    --train-years 2000-2025 --test-years 2026 \
    --output-dir storage/my_experiment --feature-selection

# Generate tuning grids
uv run python -m $MODULE.generate_tuning_configs --model-type logistic_regression
uv run python -m $MODULE.generate_feature_ablation_configs --model-type gradient_boosting \
    --ablation-types leave_one_out additive
```

## Models

### Logistic Regression (`lr`)
- StandardScaler preprocessing
- ElasticNet (L1/L2 via `l1_ratio`), SAGA solver
- Model-specific features: log transforms, cyclical encoding, interactions

### Conditional Logit (`clogit`)
- Multinomial choice model — enforces probabilities sum to 1 within each ceremony
- Mathematically equivalent to temperature-scaled softmax on LR log-odds
- Best single model for 8/9 categories (see Recommended Settings)

### Gradient Boosting (`gbt`)
- No feature scaling needed
- Raw feature values (no transforms)
- Native feature importance

### Calibrated Softmax GBT (`cal_sgbt`)
- Binary GBT base with `calibration.py` post-processing
- Temperature-scaled softmax on GBT log-odds → normalized multinomial probabilities
- Best model for animated_feature; competitive across all categories

## Recommended Settings (from [feature ablation study](../one_offs/d20260220_feature_ablation/))

### Per-category model selection

| Category | Model | Brier | Acc % |
|----------|-------|-------|-------|
| best_picture | Clogit | 0.051 | 76.9% |
| directing | Clogit | 0.035 | 88.5% |
| actor_leading | Clogit | 0.067 | 80.8% |
| actress_leading | Clogit | 0.039 | 80.8% |
| actor_supporting | Clogit | 0.029 | 92.3% |
| actress_supporting | Clogit | 0.021 | 92.3% |
| original_screenplay | Clogit | 0.025 | 92.3% |
| cinematography | Clogit | 0.022 | 92.3% |
| animated_feature | Cal-SGBT | 0.022 | 95.8% |

Clogit wins 8/9 categories. Cal-SGBT only wins animated_feature.
Metrics: LOYO CV across 26 ceremonies (2000–2025), feature selection threshold
t=0.90.

### Pipeline defaults

1. **Feature set**: `lr_full` (clogit) / `gbt_full` (cal_sgbt) — per-category
2. **Feature selection**: ON, importance threshold **t=0.90**
3. **No max_features cap** — threshold is sufficient
4. **Clogit alpha grid**: [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
5. **VIF filter**: OFF

### Per-category feature configs

Feature sets **differ per category** — each includes different conditional feature
groups. Configs in [`configs/features/`](configs/features/):

| Category | Clogit (`lr_full`) | Cal-SGBT (`gbt_full`) |
|----------|--------------------|-----------------------|
| best_picture | [`best_picture_lr_full.json`](configs/features/best_picture_lr_full.json) (51) | [`best_picture_gbt_full.json`](configs/features/best_picture_gbt_full.json) (50) |
| directing | [`directing_lr_full.json`](configs/features/directing_lr_full.json) (47) | [`directing_gbt_full.json`](configs/features/directing_gbt_full.json) (46) |
| actor_leading | [`actor_leading_lr_full.json`](configs/features/actor_leading_lr_full.json) (51) | [`actor_leading_gbt_full.json`](configs/features/actor_leading_gbt_full.json) (50) |
| actress_leading | [`actress_leading_lr_full.json`](configs/features/actress_leading_lr_full.json) (51) | [`actress_leading_gbt_full.json`](configs/features/actress_leading_gbt_full.json) (50) |
| actor_supporting | [`actor_supporting_lr_full.json`](configs/features/actor_supporting_lr_full.json) (47) | [`actor_supporting_gbt_full.json`](configs/features/actor_supporting_gbt_full.json) (46) |
| actress_supporting | [`actress_supporting_lr_full.json`](configs/features/actress_supporting_lr_full.json) (47) | [`actress_supporting_gbt_full.json`](configs/features/actress_supporting_gbt_full.json) (46) |
| original_screenplay | [`original_screenplay_lr_full.json`](configs/features/original_screenplay_lr_full.json) (44) | [`original_screenplay_gbt_full.json`](configs/features/original_screenplay_gbt_full.json) (43) |
| cinematography | [`cinematography_lr_full.json`](configs/features/cinematography_lr_full.json) (43) | [`cinematography_gbt_full.json`](configs/features/cinematography_gbt_full.json) (42) |
| animated_feature | [`animated_feature_lr_full.json`](configs/features/animated_feature_lr_full.json) (47) | [`animated_feature_gbt_full.json`](configs/features/animated_feature_gbt_full.json) (46) |

Numbers in parentheses are feature counts. `lr_full` has 1 more than `gbt_full`
(`nominations_percentile_in_year`, LR-only). Category differences come from
conditional feature groups:
- **BP** (51/50): voting_system features (irv_era, nominees_in_year)
- **Acting** (47–51/46–50): person_career + person_enrichment; leading adds GG composite
- **Directing** (47/46): person_career + person_enrichment
- **Screenplay** (44/43): person_career only
- **Cinematography** (43/42): fewest category-specific precursors
- **Animated** (47/46): animated-specific features (studio, sequel flags)

## Feature Sets

Features are model-specific — LR gets transformed features, GBT gets raw values.

Features have `available_from` dates — the pipeline automatically filters features
that wouldn't be available at prediction time via `--as-of-date`.

## Cross-Validation Strategies

| Strategy | Config | Description |
|----------|--------|-------------|
| Expanding window | `expanding_window.json` | Train on all prior years (min 5), test on next |
| Leave-one-year-out | `leave_one_year_out.json` | Each year is held out; all others train |
| Sliding window | `sliding_window_10.json` | Fixed 10-year training window |

Nested CV is supported for unbiased hyperparameter tuning (inner loop selects config,
outer loop evaluates). Combined scoring uses accuracy + log-loss with complexity-aware
tie-breaking.

## Evaluation Metrics

Both micro (pooled) and macro (per-year averaged) aggregation.

**Discrimination:** accuracy, top-3/5, MRR, AUC-ROC
**Calibration:** Brier score, log loss, mean winner probability

## Data Flow

```
Raw JSON dataset
    → data_loader (load + feature engineering)
    → evaluate_cv (CV evaluation) or train_predict (train + predict)
    → evaluation (metrics)
    → analysis/ (comparison tables, plots)
```

## Configs

All configs live in `configs/` as JSON:

- `configs/models/` — individual model configs (LR, GBT with tuned hyperparameters)
- `configs/features/` — feature set definitions per model type
- `configs/cv_splits/` — CV strategy configs
- `configs/param_grids/` — hyperparameter grids (60 LR configs, 36 GBT configs)
