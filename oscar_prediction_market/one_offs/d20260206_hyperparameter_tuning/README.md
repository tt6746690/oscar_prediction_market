# Hyperparameter Tuning

**Storage:** `storage/d20260205_hyperparameter_tuning/`,
`storage/d20260205_hyperparameter_tuning_ii/`,
`storage/d20260206_hyperparameter_tuning_iii/`,
`storage/d20260206_hyperparameter_tuning_iv/`

Four rounds of hyperparameter tuning (Feb 5-6) that converged on the final
GBT and LR configurations used throughout the project. Also includes realistic
deployment accuracy estimates using nested CV.

## Findings

### Feb 5: Nested CV gives honest deployment estimates

**Storage:** `storage/d20260205_hyperparameter_tuning/`

Nested CV for unbiased accuracy estimates. Compares expanding window
(time-respecting) vs LOYO (ceiling estimate).

| Model | Strategy | CV Type | Accuracy | Log-Loss |
|-------|----------|---------|----------|----------|
| LR | Expanding | Nested | 59.1% | 0.212 |
| LR | LOYO | Nested | 63.0% | 0.209 |

- Honest deployment estimate: **59.1% accuracy** (expanding window respects time)
- LOYO ceiling: 63% — room to improve if temporal patterns generalize
- Selection bias (non-nested vs nested) is typically 4-7pp — always use nested CV
  for final reporting

### Feb 5: Feature availability matters — ~60-67% accuracy without full precursors

**Storage:** `storage/d20260205_hyperparameter_tuning_ii/`

Testing what accuracy is achievable with only features available as of Feb 5, 2026
(missing PGA, DGA, SAG, BAFTA winners).

| Model | Features | Condition | Accuracy | Log-Loss | Top-3 |
|-------|----------|-----------|----------|----------|-------|
| GBT | 20 | As-of today | **66.7%** | 0.340 | 81.5% |
| LR | 23 | As-of today | 59.3% | 0.457 | 77.8% |
| GBT | 24 | All features | **66.7%** | 0.273 | 96.3% |
| LR | 29 | All features | **66.7%** | 0.263 | 92.6% |

- Without precursor winners, accuracy drops from ~67% to 59-67%
- GBT handles missing winners better than LR (maintains 66.7% vs 59.3%)
- `critics_choice_winner` becomes the most important feature when other winners
  are unavailable
- Wait for more precursor winners before making high-confidence predictions

### Feb 6: GBT grid search — conservative hyperparameters win

**Storage:** `storage/d20260206_hyperparameter_tuning_iii/`

Grid search over 47 GBT configs using LOYO CV. Now using `evaluate_cv.py` with
joblib parallelization.

Best config: n_estimators=50, learning_rate=0.05, max_depth=2,
min_samples_split=5, min_samples_leaf=2, subsample=0.8. → **73.1% accuracy**.

- Conservative hyperparameters win: smallest tree count, slowest learning rate,
  shallowest depth
- Consistent with small-data regime — aggressive regularization prevents overfitting
- `critics_choice_winner` dominates feature importance (54%) when DGA is unavailable

### Feb 6: Two-round grid refinement — final configs

**Storage:** `storage/d20260206_hyperparameter_tuning_iv/`

Round 1's best configs were at grid edges (minimum tree count, minimum depth),
so Round 2 extended the search toward even simpler models.

**Final configs (used throughout all later experiments):**

| Model | Key Parameters |
|-------|---------------|
| LR | C=0.5, l1_ratio=1.0, max_iter=2000 |
| GBT | n_estimators=25, lr=0.1, max_depth=2 |

**Results:**

| Metric | LR | GBT |
|--------|-----|-----|
| Accuracy | 69.2% (18/26) | **73.1% (19/26)** |
| Top-3 | 84.6% | **88.5%** |
| MRR | 0.789 | **0.809** |
| Log-Loss | 0.240 | **0.222** |

- **GBT outperforms LR by 3.9pp** (73.1% vs 69.2%)
- GBT went even simpler: n=25 (vs n=50 in Round 1) — small-data regime favors
  aggressive regularization
- LR shifted to pure L1 (l1_ratio=1.0) with higher C=0.5
- Depth=2 consistently optimal for GBT — deeper trees overfit
- `critics_choice_winner` dominates both models when DGA is filtered
- Both models predict **One Battle after Another** as 2026 winner

### Classic "upset years" both models miss

2005 (Million Dollar Baby), 2006 (Crash), 2011 (King's Speech), 2015 (Birdman),
2017 (Moonlight), 2019 (Green Book), 2022 (CODA). GBT uniquely gets 2020
(Parasite) right.
