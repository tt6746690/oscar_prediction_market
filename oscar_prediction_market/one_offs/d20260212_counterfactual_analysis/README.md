# DGA Winner Counterfactual Analysis

**Storage:** `storage/d20260212_counterfactual_analysis/dga_winner/`

Counterfactual analysis framework for feature events: when a binary feature becomes
available (e.g., DGA winner announced), how does each possible outcome change model
win probabilities, and where are the trading edges vs the market?

Applied to the DGA winner announcement (Feb 7, 2026). Builds on the earlier
[DGA sensitivity analysis](../d20260207_dga_sensitivity/) with a cleaner experimental
design: counterfactuals modify raw data at the `PrecursorAwardsResult` level, ensuring
derived features (`precursor_wins_count`, `has_pga_dga_combo`) are automatically consistent.

## Setup

- Baseline models: as-of-date 2026-02-06 (no DGA winner feature)
- Scenario models: as-of-date 2026-02-07 (DGA winner feature available)
- 3 model types: LR, GBT, XGBoost, each with full feature selection pipeline
- 5 counterfactual scenarios (one per DGA nominee)

## Findings

### LOYO CV results (selected features, 2000-2025)

| Model | Role | Accuracy | Top-3 | Log-Loss | MRR | Brier | # Feat |
|-------|------|----------|-------|----------|-----|-------|--------|
| LR | baseline (02-06) | **76.9%** | 92.3% | **0.1671** | 0.857 | **0.0464** | 11 |
| LR | scenario (02-07) | 73.1% | 92.3% | 0.1997 | 0.836 | 0.0561 | 25 |
| GBT | baseline (02-06) | 73.1% | 88.5% | 0.2203 | 0.816 | 0.0597 | 10 |
| GBT | scenario (02-07) | **80.8%** | **96.2%** | 0.1994 | **0.875** | 0.0566 | 12 |
| XGB | baseline (02-06) | 69.2% | 88.5% | 0.2331 | 0.804 | 0.0656 | 12 |
| XGB | scenario (02-07) | **80.8%** | 84.6% | 0.2127 | 0.856 | 0.0585 | 10 |

DGA winner feature improves GBT and XGB by ~8-12pp accuracy. LR scenario actually
drops 3.8pp — the 25-feature selected set overfits vs the lean 11-feature baseline,
a feature selection artifact from the larger as-of-date feature pool.

### DGA winner is the single most impactful feature for trading edge

| DGA Winner | LR (base→scen) | GBT (base→scen) | XGB (base→scen) | Mean | Kalshi | Edge |
|------------|----------------|------------------|------------------|------|--------|------|
| Frankenstein | 0.6%→12.1% | 4.2%→47.4% | 2.1%→25.5% | **28.3%** | 1% | **+27.3%** |
| Hamnet | 2.1%→28.0% | 11.8%→64.5% | 8.4%→39.4% | **44.0%** | 6% | **+38.0%** |
| Marty Supreme | 8.9%→39.3% | 10.1%→66.4% | 23.1%→47.5% | **51.0%** | 3% | **+48.0%** |
| One Battle | 10.5%→70.8% | 51.0%→82.1% | 62.4%→66.7% | **73.2%** | 71% | **+2.2%** |
| Sinners | 2.4%→30.8% | 4.7%→60.7% | 6.9%→33.2% | **41.6%** | 20% | **+21.6%** |

Upset DGA outcomes create 20-48pp edges vs pre-DGA market prices. The expected
outcome (One Battle wins DGA) yields only +2.2pp — not actionable.

### LR baseline is notably miscalibrated in this run

| Nominee | LR | GBT | XGB | Kalshi (Feb 6) |
|---------|-----|-----|-----|----------------|
| One Battle | 10.5% | 51.0% | 62.4% | 71% |
| Marty Supreme | 8.9% | 10.1% | 23.1% | 3% |
| Sinners | 2.4% | 4.7% | 6.9% | 20% |

LR gives One Battle only 10.5% vs 71% market — a feature selection artifact from the
larger as-of-date feature pool (11 auto-selected features vs 8 hand-picked in the
Feb 7 experiment). The [LR stabilization work](../d20260213_feature_ablation/) addressed
this root cause.

### Trading strategy: pre-DGA upset insurance

Buy cheap Yes contracts on DGA underdogs before the ceremony:
- Marty at 3c, Frankenstein at 1c, Hamnet at 6c
- If any wins DGA, model expects 28-51% Oscar win probability
- Creates massive arbitrage (20-48pp edges) vs pre-DGA market prices

Market efficiently prices the expected outcome — DGA confirmation for the favorite
is a non-event. All the value is in upset scenarios.

### GBT reacts most strongly to DGA

GBT gives 47-82% to any DGA winner (highest sensitivity). LR is most muted (12-71%).
XGB is in between (26-67%). Tree models place disproportionate weight on a single
strong binary feature.

### Counterfactual framework is generalizable

The `FeatureEvent` abstraction can be applied to any binary feature with multiple
candidates (BAFTA winner, PGA winner, SAG ensemble winner). Modify raw data →
re-run feature engineering → predict with scenario-trained model.

### Comparison with Feb 7 DGA analysis

This experiment broadly confirms the [Feb 7 DGA sensitivity analysis](../d20260207_dga_sensitivity/)
(which used a slightly different feature set). Key differences:
- LR baseline here (10.5% for One Battle) is much lower than Feb 7's (72.8%) — the
  Feb 7 experiment used 8 hand-picked features vs this one's 11 auto-selected. Different
  feature selection → very different LR calibration.
- GBT is consistent: 51.0% baseline for One Battle here vs 51.0% in Feb 7.
- Mean scenario probabilities are similar order of magnitude.
- Non-DGA-eligible nominees barely move (<2pp shift regardless of scenario). DGA winner
  status is almost entirely a within-eligible redistribution.

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"
bash oscar_prediction_market/one_offs/d20260212_counterfactual_analysis/run.sh \
    2>&1 | tee storage/d20260212_counterfactual_analysis/run.log
```
