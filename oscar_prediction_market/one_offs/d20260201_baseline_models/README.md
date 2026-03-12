# Baseline Models

**Storage:** `storage/d20260202_oscar_initial_baseline/`

Initial comparison of Logistic Regression vs Gradient Boosting across feature sets
for Oscar Best Picture prediction. Establishes the accuracy ceiling and failure cases.

## Setup

- 206 Best Picture nominees, ceremonies 72-98 (27 years)
- Leave-one-year-out expanding window (min 5 years training)
- 22 test ceremonies
- Feature sets: minimal (21), standard (26), full (45)

## Findings

### LR significantly outperforms GBT in this initial setup

| Model | Feature Set | Accuracy | Top-3 | Brier | Log Loss |
|-------|-------------|----------|-------|-------|----------|
| LR | minimal (21) | **77.3%** | 100% | 0.053 | 0.201 |
| LR | standard (26) | **77.3%** | 95.5% | **0.047** | **0.186** |
| LR | full (45) | 63.6% | 100% | 0.067 | 0.260 |
| GBT | standard (26) | 54.5% | 100% | 0.077 | 0.378 |

Simpler model generalizes better on small data. Full feature set hurts performance
(63.6%) — commercial/timing features are noise. Winner is almost always in top 3
predictions (95-100% top-3 accuracy).

Performance improves with more training data: first half 72.7%, second half 81.8%.

### Classic "upset years" are consistently hard

- Ceremony 78 (2006): Crash beat Brokeback Mountain — model gave Brokeback 88% probability
- Ceremony 89 (2017): Moonlight beat La La Land — model gave La La Land 99.8%, Moonlight 0.2%
- These upset years remain hard across all model variants in later experiments.

### Leakage caveat

This baseline used `oscar_total_wins` as a feature, which is leakage (only known
after the ceremony). Later experiments removed this and showed lower but more realistic
accuracy.
