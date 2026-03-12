# Bagged Ensemble — Same-Class Variance Reduction

**Storage:** `storage/d20260207_bagged_ensemble/`

Bootstrap aggregation (bagging) trains K=100 copies of the same model on bootstrap
samples, then averages their predicted probabilities. Primary goal: uncertainty
quantification via per-bag probability distributions, not accuracy improvement.

## Setup

- K=100 bootstrap bags per model
- Generic `BaggedClassifierModel` wrapper in `models.py` (wraps any base `ModelConfig`)
- Integrated into discriminated union: bagged configs are first-class citizens
- LOYO CV (26 years, 2000-2025), as-of 2026-02-06

## Findings

### Bagging does not improve accuracy

| Config | Accuracy | Top-3 | Log-Loss | MRR | Brier |
|--------|----------|-------|----------|-----|-------|
| **GBT single** | **73.1%** (19/26) | 88.5% | **0.2208** | **0.816** | **0.0597** |
| GBT bagged | 69.2% (18/26) | **92.3%** | 0.2351 | 0.812 | 0.0638 |
| LR single | 69.2% (18/26) | 84.6% | 0.2305 | 0.796 | 0.0608 |
| LR bagged | 69.2% (18/26) | 84.6% | **0.2280** | 0.796 | **0.0597** |

Bagging hurt GBT by flipping 2020 (Nomadland) from correct to incorrect. Bootstrap
resampling diluted whatever signal the single GBT found for that year. LR was
unaffected — identical error sets, confirming LR's stability under resampling.

### The real value: uncertainty quantification

The per-bag distribution quantifies model confidence. For the 2026 top pick
(One Battle after Another):

| Stat | GBT Bagged | LR Bagged |
|------|-----------|----------|
| Mean | 61.8% | 76.7% |
| Std | 20.0% | 7.9% |
| Min | 12.1% | 52.0% |
| Q75 | 78.3% | 82.0% |
| Max | 94.6% | 93.0% |
| Confidence (mean/std) | 3.1x | 9.7x |

GBT shows high uncertainty (std=20%) — some bags see the top pick at 12%, others
at 95%. LR is much tighter (std=8%) — the lowest bag still gives 52%. GBT's
sensitivity to bootstrap samples reflects its dependence on which training years
are included.

![storage/d20260207_bagged_ensemble/figures/top_pick_distributions.png](assets/top_pick_distributions.png)

![storage/d20260207_bagged_ensemble/figures/all_nominees_distributions.png](assets/all_nominees_distributions.png)

### Trading implications from bag distributions

1. **Confidence-weighted sizing.** LR's tight distribution (9.7x mean/std) suggests
   higher confidence in the top pick than GBT (3.1x). Position size should scale
   with confidence ratio, not just point estimate.
2. **No-overlap test.** For both models, the 2nd-place max across all bags is below
   the 1st-place Q25 — in no bootstrap does the runner-up overtake the favorite.
3. **Model disagreement as risk signal.** GBT gives Secret Agent 9.8% while LR gives
   0.3%. Films with >5% disagreement should be treated as uncertain.
4. **Tail risk.** GBT's 12.1% minimum for the top pick means there exist plausible
   scenarios where the favorite barely registers. If a market prices the favorite
   above 80%, the GBT bag distribution suggests this is overpriced given model uncertainty.
