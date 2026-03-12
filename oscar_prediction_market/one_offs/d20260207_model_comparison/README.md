# Model Comparison — LR vs GBT vs XGBoost

**Storage:** `storage/d20260207_compare_model/`

Head-to-head comparison of three model types: Logistic Regression, sklearn Gradient
Boosting, and XGBoost. All non-bagged, with LOYO grid search on 2000-2025,
as-of-date 2026-02-06.

## Setup

- XGBoost uses the same tree-based feature engineering as GBT (raw values, no log transforms)
- Each model's nonzero-importance feature set was used (LR: 8, GBT/XGB: 10)
- XGBoost's nonzero features were derived from a full-feature first pass

## Findings

### GBT is the best single model; XGBoost adds no value

| Model | Accuracy | Top-3 | Log-Loss | MRR | Brier | AUC-ROC |
|-------|----------|-------|----------|-----|-------|---------|
| LR | 69.2% (18/26) | 84.6% | **0.2173** | 0.802 | **0.0584** | **0.900** |
| **GBT** | **73.1%** (19/26) | **88.5%** | 0.2203 | **0.816** | 0.0597 | 0.883 |
| XGBoost | 69.2% (18/26) | **88.5%** | 0.2377 | 0.809 | 0.0672 | 0.893 |

Best hyperparameters:

| Model | Key Parameters |
|-------|---------------|
| LR | C=0.2, l1_ratio=0.0 (pure L2), solver=saga, max_iter=2000 |
| GBT | n_estimators=25, lr=0.1, max_depth=2 |
| XGBoost | n_estimators=50, lr=0.05, max_depth=2, colsample_bytree=1.0 |

### XGBoost makes identical errors to LR

| Model | Incorrect Years | Count |
|-------|----------------|-------|
| LR | 2005, 2006, 2011, 2015, 2017, 2019, 2020, 2022 | 8 |
| GBT | 2005, 2006, 2011, 2015, 2017, 2019, 2022 | 7 |
| XGBoost | 2005, 2006, 2011, 2015, 2017, 2019, 2020, 2022 | 8 |

XGBoost's 8 errors are identical to LR's — it converges to similar decision
boundaries at this dataset scale. GBT's 7 errors are a strict subset; GBT uniquely
gets 2020 (Parasite) right.

### Dataset is too small for XGBoost's complexity

The dataset (~196 samples, 26 years) is too small for XGBoost's additional complexity
(regularization, column sampling) to help. Simpler models dominate. XGBoost settled
on shallow trees (depth=2) with lower learning rate (0.05), consistent with
small-data overfitting avoidance.

### 2026 predictions

All three models agree on **One Battle after Another** as the 2026 top pick.

| Film | LR | GBT | XGBoost |
|------|-----|-----|---------|
| One Battle after Another | 72.8% | 51.0% | 54.9% |
| Sinners | 20.9% | 4.7% | 7.1% |
| Marty Supreme | 19.4% | 10.1% | 19.8% |
