# DGA Winner Sensitivity Analysis

**Storage:** `storage/d20260207_include_dga_winner/`

For each of the 5 DGA nominees, simulate "what if this film wins DGA?" and show how
predictions change across LR, GBT, and XGBoost. Demonstrates that DGA winner is the
single most important feature for Oscar Best Picture prediction and identifies massive
trading edges in upset scenarios.

See also: [DGA Counterfactual Analysis](../d20260212_counterfactual_analysis/) for a
more rigorous framework applied to the same question, and
[DGA Price Impact](../d20260208_dga_price_impact/) for post-DGA market reaction analysis.

## Setup

- As-of-date 2026-02-07 (DGA winner feature now available)
- Used all features from `feature_engineering.py` (40 LR / 35 GBT/XGB)
- Feature selection via nonzero-importance from full-feature model training
- Baseline comparison: 02-06 model (no DGA feature)

**Important — OOD baseline caveat.** The 02-07 model with `dga_winner=False` for all
nominees is out-of-distribution: in training data (2000-2025), every year has exactly
1 film with `dga_winner=True`. The correct baseline is the **02-06 model** (trained
without the DGA feature). Scenario predictions (exactly 1 film has `dga_winner=True`)
are in-distribution and reliable.

## Findings

### DGA winner is the #1 feature — model accuracy jumps 5-8pp

| Model | Accuracy | Top-3 | Log-Loss | MRR | Brier | # Features |
|-------|----------|-------|----------|-----|-------|------------|
| LR | 76.9% (20/26) | 92.3% | **0.1873** | 0.852 | **0.0506** | 13 |
| **GBT** | **80.8%** (21/26) | 92.3% | 0.2076 | **0.878** | 0.0594 | 15 |
| XGBoost | 76.9% (20/26) | 92.3% | 0.2000 | 0.854 | 0.0551 | 17 |

Improvement over 02-06 experiment: GBT 73.1% → 80.8%, LR 69.2% → 76.9%. `dga_winner`
is #1 or #2 most important feature across all models. DGA winner → Oscar winner in
18/26 years (69.2%) over 2000-2025.

### Upset DGA outcomes create massive trading edges

| DGA Winner | LR (02-06→scenario) | GBT (02-06→scenario) | XGB (02-06→scenario) | Mean | Kalshi | Edge |
|------------|---------------------|----------------------|----------------------|------|--------|------|
| Frankenstein | 2.7%→15.6% (+12.9) | 4.2%→53.9% (+49.7) | 3.5%→28.8% (+25.3) | **32.8%** | ~1% | **+31.8%** |
| Hamnet | 5.5%→34.2% (+28.7) | 11.8%→70.4% (+58.6) | 11.5%→38.4% (+26.9) | **47.7%** | 8% | **+39.7%** |
| Marty Supreme | 19.4%→53.1% (+33.7) | 10.1%→83.6% (+73.5) | 19.8%→61.7% (+41.9) | **66.1%** | 2% | **+64.1%** |
| One Battle | 72.8%→87.1% (+14.3) | 51.0%→84.3% (+33.3) | 54.9%→72.6% (+17.7) | **81.3%** | 73% | **+8.3%** |
| Sinners | 20.9%→60.6% (+39.7) | 4.7%→84.3% (+79.6) | 7.1%→44.8% (+37.7) | **63.3%** | 19% | **+44.3%** |

**If One Battle wins DGA (expected outcome):** model mean 73.2% vs market 71% → +2.2% edge.
Tiny — not actionable. The market already priced in the expected DGA outcome.

**If Marty Supreme wins DGA:** model mean 51.0% vs market 3% → **+48pp edge**. Largest
single trading opportunity. If Marty wins DGA, buy Marty Yes aggressively.

### GBT reacts most strongly to DGA

GBT gives 54-84% to any DGA winner (highest sensitivity). LR is most muted (12-71%).
Tree models place high weight on a single strong binary feature.

### 2026 scenario predictions — top 5 per model

**02-06 Baseline** (no DGA feature):

| Rank | LR | GBT | XGB |
|------|-----|-----|-----|
| 1 | One Battle (72.8%) | One Battle (51.0%) | One Battle (54.9%) |
| 2 | Sinners (20.9%) | Secret Agent (12.6%) | Marty (19.8%) |
| 3 | Marty (19.4%) | Hamnet (11.8%) | Hamnet (11.5%) |

**If One Battle wins DGA:**

| Rank | LR | GBT | XGB |
|------|-----|-----|-----|
| 1 | **One Battle (87.1%)** | **One Battle (84.3%)** | **One Battle (72.6%)** |
| 2 | Sinners (10.8%) | Secret Agent (10.3%) | Marty (19.8%) |

**If Sinners wins DGA:**

| Rank | LR | GBT | XGB |
|------|-----|-----|-----|
| 1 | **Sinners (60.6%)** | **Sinners (84.3%)** | **Sinners (44.8%)** |
| 2 | One Battle (34.8%) | Secret Agent (10.3%) | Marty (19.8%) |

### Bayesian update vs retrained model — retrained is better

| DGA Winner | Bayes Mean | Retrained Mean | Kalshi |
|------------|------------|----------------|--------|
| Frankenstein | 65.5% | 32.8% | ~1% |
| Hamnet | 65.3% | 47.7% | 8% |
| Marty Supreme | 64.9% | 66.1% | 2% |
| One Battle | 60.1% | 81.3% | 73% |
| Sinners | 65.1% | 63.3% | 19% |

Bayesian approach gives ~65% to ANY DGA winner regardless of identity — nearly
uniform because the formula collapses to: posterior ≈ DGA base rate (69.2%). It
ignores film-specific features entirely. Frankenstein (no other precursor wins) gets
the same ~65% as One Battle (CC + GG winner).

The retrained model preserves film-specific context: One Battle jumps to 81.3% when
it also wins DGA (feature interactions with CC+GG), while Frankenstein only reaches
32.8%. **Retrained >> Bayesian for scenario analysis.**

### More feature nonzero counts due to larger feature pool and DGA reshuffling

LR: 8→13 features, GBT: 10→15, XGB: 10→17. Two factors:
1. Larger input feature pool (40/35 vs 21-24 in prior experiments)
2. Adding `dga_winner` (strongest predictor) reshuffles the importance landscape —
   secondary features that were previously redundant now contribute marginal signal
