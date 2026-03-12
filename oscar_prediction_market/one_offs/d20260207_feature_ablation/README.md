# Feature Ablation — Early Studies

**Storage:** `storage/d20260205_ablate_features/`, `storage/d20260207_ablate_features/`

Two rounds of systematic feature ablation (Feb 5 and Feb 7) that established the key
insight: **precursor winners are the only feature group that matters for accuracy.** The
[later ablation (Feb 13-14)](../d20260213_feature_ablation/) built on these findings with
more configs, temporal stability analysis, and feature subset validation.

## Feb 5: Initial ablation with expanding window CV

**Storage:** `storage/d20260205_ablate_features/`

### Feature groups

| Group | Description | Features |
|-------|-------------|----------|
| precursor_awards | PGA, DGA, SAG, BAFTA, etc. | 6-8 |
| oscar_nominations | Total noms, category breakdowns | 5-6 |
| critic_scores | RT, MC, IMDb | 2-3 |
| commercial | Budget, box office | 2 |
| timing | Release month, awards season | 2-3 |

### LR results (sorted by accuracy)

| Configuration | Features | Accuracy | Log-Loss |
|---------------|----------|----------|----------|
| only_precursor_awards | 8 | **72.7%** | 0.277 |
| additive_3_critic_scores | 16 | 68.2% | **0.209** |
| additive_2_oscar_noms | 14 | 68.2% | 0.219 |
| full (all 5 groups) | 21 | 59.1% | 0.319 |
| without_precursor_awards | 13 | 27.3% | 0.523 |

### Key finding: more features = worse performance

8 features (72.7%) → 21 features (59.1%). Classic small-sample overfitting. Commercial
and timing features actively hurt — removing them improves accuracy.

For calibrated probabilities (trading): use precursor + oscar + critic (16 features,
0.209 log-loss). For maximum accuracy: use only precursor awards (8 features, 72.7%).

## Feb 7: Ablation with tuned hyperparameters

**Storage:** `storage/d20260207_ablate_features/`

Expanded ablation using best hyperparameters from [tuning](../d20260206_hyperparameter_tuning/)
(LR: C=0.5, l1_ratio=1.0; GBT: n=25, lr=0.1, depth=2). 40 total experiments.

### Precursor winners dominate — removing them drops GBT by 23pp

**GBT leave-one-out accuracy impact:**

| Removed Group | Accuracy | Delta vs Full |
|---------------|----------|---------------|
| timing | 73.1% | 0.0 |
| precursor_nominations | 73.1% | 0.0 |
| commercial | 73.1% | 0.0 |
| oscar_nominations | 69.2% | -3.8 |
| critic_scores | 69.2% | -3.8 |
| **precursor_winners** | **50.0%** | **-23.1** |

The 7 incorrect years are identical across all 73.1% GBT variants — no feature
subset changes *which* years are correct, only whether the model reaches the plateau.

### LR with L1 is completely insensitive to feature set (given precursor winners)

All LR variants with precursor winners achieve identical 69.2% accuracy and identical
incorrect years, regardless of whether you give it 2 or 23 features. L1 (l1_ratio=1.0)
zeros out all but ~8 features regardless of input size. The only difference is
calibration.

### GBT extracts value from critic scores beyond precursor winners

**GBT additive build-up:**

| Step | Groups | #Feat | Accuracy | Composite |
|------|--------|-------|----------|-----------|
| 1 | precursor_winners | 2 | 69.2% | 0.731 |
| 2 | + precursor_noms | 8 | 69.2% | 0.759 |
| 3 | + oscar_noms | 13 | 69.2% | 0.738 |
| **4** | **+ critic_scores** | **16** | **73.1%** | **0.795** |
| 5 | + commercial | 18 | 73.1% | 0.787 |
| 6 | + timing (full) | 20 | 73.1% | 0.787 |

Accuracy jumps from 69.2% to 73.1% only when critic_scores are added (step 4).
Adding commercial and timing after that slightly degrades composite score.

### Nonzero importance sets are Pareto-optimal

Pruning to only nonzero-importance features matches or beats the full model on
every metric:

| Model | Variant | #Feat | Accuracy | Log-Loss | MRR |
|-------|---------|-------|----------|----------|-----|
| GBT | full | 20 | 73.1% | 0.223 | 0.809 |
| **GBT** | **nonzero** | **10** | **73.1%** | **0.221** | **0.816** |
| LR | full | 23 | 69.2% | 0.240 | 0.789 |
| **LR** | **nonzero** | **8** | **69.2%** | **0.231** | **0.796** |

Fewer features, same accuracy, better calibration. No reason to use the full set.

### The 7 "upset years" are a hard ceiling

No feature combination changes which years are correctly predicted — only whether
the model reaches the ~73% plateau. Improvements likely require different data
sources, not more features from the same sources.
