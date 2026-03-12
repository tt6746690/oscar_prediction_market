# Within-Year Relative Features (Percentile / Z-Score)

**Storage:** `storage/d20260209_relative_features/`

Many numeric features (critic scores, box office, budget, runtime) shift across
decades — a Metacritic 80 means different things in 2002 vs 2022. This experiment
tests whether within-year normalization (percentile for LR, z-score for GBT/XGB)
improves predictions.

## Setup

- New features: 7 base metrics (metacritic, RT, IMDb, worldwide/domestic box office,
  budget, runtime) x 2 normalization types
- LR gets percentile variants (bounded [0,1]); GBT/XGB get z-score variants (unbounded)
- 3 variants (baseline, relative_only, absolute+relative) x 3 models = 9 configs
- 2-pass CV: full-feature LOYO → feature selection → selected-feature LOYO

## Findings

### GBT baseline remains strongest; relative features don't beat it

| Model | Variant | # Feat | Accuracy | Top-3 | MRR | Log-Loss | Brier |
|-------|---------|--------|----------|-------|-----|----------|-------|
| LR | baseline | 13 | 76.9% | 92.3% | 85.2% | 0.1873 | 0.0506 |
| LR | relative_only | 11 | 76.9% | 88.5% | 85.0% | 0.1970 | 0.0574 |
| LR | absolute+relative | 13 | 76.9% | 88.5% | 85.3% | **0.1803** | 0.0513 |
| **GBT** | **baseline** | **15** | **80.8%** | 92.3% | **87.8%** | 0.2076 | 0.0594 |
| GBT | relative_only | 13 | 73.1% | **96.2%** | 84.9% | 0.2078 | 0.0608 |
| GBT | absolute+relative | 19 | 76.9% | **96.2%** | 86.2% | 0.2089 | 0.0611 |
| XGB | baseline | 17 | 76.9% | 92.3% | 85.4% | 0.2000 | 0.0551 |
| **XGB** | **relative_only** | **17** | **80.8%** | **96.2%** | **88.1%** | 0.2064 | 0.0594 |
| XGB | absolute+relative | 17 | 76.9% | **100.0%** | 86.5% | 0.1980 | 0.0571 |

### XGB relative_only is curiously strong (80.8%) but poorly calibrated

XGB relative_only beats baseline by 3.9pp accuracy and achieves better MRR (88.1%
vs 85.4%). XGB may benefit from z-score normalization because its regularization
handles normalized features better than GBT's greedy splits. But mean P(win) drops
to 46.5% vs 59.0% — calibration suffers despite accuracy improvement.

### LR benefits from combining absolute + relative for calibration

Best log-loss (0.1803) comes from absolute+relative, suggesting percentile features
improve calibration. But Top-3 drops from 92.3% to 88.5% — the model sometimes
ranks the winner lower.

### Important relative features

`imdb_zscore_in_year`, `budget_zscore_in_year`, and `metacritic_zscore_in_year` are
the most important relative features across models. Runtime and box office z-scores
contribute minimally.

### Practical conclusion

No relative-feature variant beats GBT baseline (80.8%). The raw absolute values
that GBT splits on already capture the signal. Z-scores add redundant information
that dilutes rather than enhances. Keep relative features available in the
feature pool but don't prioritize them.
