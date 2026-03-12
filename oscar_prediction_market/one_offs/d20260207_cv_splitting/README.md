# CV Splitter Comparison — How Much Training Data?

**Storage:** `storage/d20260207_cv_splitting/`

Compared 6 CV splitting strategies to determine how much training data the final
model should use, and whether temporal structure in Oscar voting patterns matters.

## Setup

- Tuned hyperparameters: GBT (n=25, lr=0.1, depth=2), LR (C=0.5, l1_ratio=1.0)
- Nonzero-importance feature sets (GBT: 10, LR: 8)
- Non-nested, fixed configs. Bootstrap 500 resamples for 95% CIs

## Findings

### Use all available data — performance increases monotonically

![storage/d20260207_cv_splitting/plots/accuracy_vs_window_size.png](assets/accuracy_vs_window_size.png)

Sliding(N=5) → 42.9%, Sliding(N=20) → 66.7%, LOYO → 73.1% for GBT. LR is
flatter — L1 regularization handles small samples better. Expanding window uses
all prior data and is the realistic deployment strategy.

### Nothing is statistically significant given small dataset

![storage/d20260207_cv_splitting/plots/bootstrap_ci_accuracy.png](assets/bootstrap_ci_accuracy.png)

All CIs overlap heavily. With 6-26 test years, no splitter difference is
statistically significant. Fundamental limitation of the 26-year dataset.

### Combined results

| Splitter | GBT Acc | GBT n | LR Acc | LR n | GBT 95% CI | LR 95% CI |
|----------|---------|-------|--------|------|------------|-----------|
| Sliding(N=5) | 42.9% | 21 | 61.9% | 21 | [23.8%, 64.4%] | [38.1%, 81.0%] |
| Sliding(N=10) | 62.5% | 16 | 62.5% | 16 | [43.8%, 87.5%] | [43.8%, 87.5%] |
| Sliding(N=15) | 63.6% | 11 | 54.5% | 11 | [36.4%, 90.9%] | [27.3%, 81.8%] |
| Sliding(N=20) | 66.7% | 6 | 66.7% | 6 | [16.7%, 100%] | [16.7%, 100%] |
| Expanding | 61.9% | 21 | 61.9% | 21 | [38.1%, 81.0%] | [38.1%, 81.0%] |
| **LOYO** | **73.1%** | **26** | **69.2%** | **26** | **[55.7%, 92.3%]** | **[50.0%, 88.5%]** |

### GBT needs more data than LR

GBT collapses to 42.9% with only 5 training years (~40 examples); LR stays at
61.9%. Tree ensembles need volume; L1-regularized LR is more sample-efficient.

### Honest deployment estimate: ~62% accuracy

LOYO vs Expanding gap (~11pp GBT, ~7pp LR) is mostly a data volume effect, not
temporal leakage. Expanding's early folds train on only 5-8 years, dragging down
its average. LOYO 73% is an optimistic ceiling.
