# PLAN: Config Selection Methodology Improvements

**Date:** 2026-03-03
**Context:** The buy-hold backtest selects configs by maximizing `ev_pnl_blend`
(analytical E[PnL] using 50/50 model+market probabilities) subject to
CVaR tail-risk constraints. Investigation reveals that EV-optimal configs
are not actual-PnL-optimal. This document analyzes the gap, proposes
alternatives, and recommends next steps.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Key Data Tables Explained](#2-key-data-tables-explained)
3. [Why Overconfident Models Get Higher EV (Toy Example)](#3-why-overconfident-models-get-higher-ev-toy-example)
4. [Should We Use Low-Inflation Models?](#4-should-we-use-low-inflation-models)
5. [A1: Calibration — Does clogit Need More?](#5-a1-calibration)
6. [A2: Market-Weighted EV — Empirical Investigation](#6-a2-market-weighted-ev)
7. [Alternative Objectives (B1–B4)](#7-alternative-objectives)
8. [C1: Model-Diverse Portfolio of Configs](#8-c1-model-diverse-portfolio)
9. [C2: Fix Model, Optimize Params Only](#9-c2-fix-model-optimize-params)
10. [D1: Bootstrap Resampling of Category Outcomes](#10-d1-bootstrap-resampling)
11. [Portfolio of Configs vs Single Ensemble Model](#11-portfolio-vs-ensemble)
12. [Can We Trust 2 Years of Data?](#12-can-we-trust-2-years)
13. [Recommendations and Next Steps](#13-recommendations)
14. [Open Questions](#14-open-questions)

---

## 1. Problem Statement

The current optimization selects configs that **maximize analytical E[PnL]
using blend probabilities**, subject to CVaR risk constraints. The EV is:

$$\text{EV}_{\text{blend}} = \frac{1}{2}\left(\sum_k q_k^{\text{model}} \cdot \text{PnL}_k + \sum_k q_k^{\text{market}} \cdot \text{PnL}_k\right)$$

where $q_k$ = probability of nominee $k$ winning, and $\text{PnL}_k$ = portfolio
profit if nominee $k$ wins. **This is NOT Monte Carlo** — it's an exact
weighted sum. MC is only used for CVaR tail-risk estimation.

**The gap:** The EV-optimal config ranks only #703/3528 (top 20%) by actual
realized PnL across 2024+2025. The hindsight-best config earns 76% more
actual PnL ($1,756 vs $996).

| | EV-optimal config | Hindsight-best config |
|---|---|---|
| Model | clogit | avg_ensemble |
| avg_ev_pnl_blend | **$3,778** | $1,664 |
| avg_actual_pnl | $996 | **$1,756** |
| 2024 actual | $1,003 | $785 |
| 2025 actual | $988 | $2,727 |
| Actual rank (of 3,528) | #703 (top 20%) | #1 |

**Root cause:** Model probability overconfidence inflates EV. Models with
sharper (more confident) probability distributions get higher EV because:
(a) Kelly bets more on high-confidence picks, and (b) EV weights those
picks more heavily. When the confidence is accurate, this is correct.
When it's overconfident, EV is systematically inflated.

---

## 2. Key Data Tables Explained

### Abbreviations

| Abbreviation | Meaning |
|---|---|
| `ev_pnl_model` | Expected PnL computed using **model** probabilities as weights |
| `ev_pnl_market` | Expected PnL computed using **market-implied** probabilities (from YES prices) |
| `ev_pnl_blend` | Average of model and market EV: $(EV_{model} + EV_{market}) / 2$ |
| `actual_pnl` | Realized PnL when settled against the **known actual winner** |
| `cvar_5` | Conditional Value at Risk at 5% — mean PnL in the worst 5% of Monte Carlo outcome scenarios |
| `avg_*` | Average of the metric across 2024 and 2025 (per-entry normalized) |
| `kf` | Kelly fraction — fraction of full Kelly bet to use (e.g., 0.25 = quarter-Kelly) |
| `bet` / `edge` | Buy edge threshold — minimum net edge required to enter a position |
| `km=multi_outcome` | Multi-outcome Kelly sizing (joint optimization across all nominees in a category) |
| `km=independent` | Independent Kelly sizing (each nominee sized independently) |
| `side=all/yes/no` | Allowed trading directions: `all`=buy YES+NO, `yes`=only YES, `no`=only NO |
| `inflation ratio` | `mean(EV) / mean(actual PnL)` — how much EV overstates reality. 1.0 = perfectly calibrated |
| `loss_bound_pct` / `L` | Maximum acceptable loss as % of per-entry bankroll for CVaR constraint |

### What "Blend EV / Actual Ratio" Means

This ratio = `mean(ev_pnl_blend) / mean(actual_pnl)` for a group of configs.

- **Ratio = 1.0x**: EV perfectly predicts average actual PnL (well-calibrated)
- **Ratio > 1.0x**: EV overstates reality (model is overconfident)
- **Ratio < 1.0x**: EV understates reality (model is underconfident)

Per-model inflation ratios (average across all configs for each model):

| Model | 2024 inflation | 2025 inflation | Overall |
|---|---|---|---|
| avg_ensemble | 1.24x | 1.28x | **1.27x** |
| clogit_cal_sgbt_ens | 1.17x | 1.70x | 1.57x |
| cal_sgbt | 7.89x | 1.46x | 1.74x |
| gbt | 3.78x | 1.66x | 1.83x |
| lr | 1.96x | 1.84x | 1.87x |
| clogit | 1.57x | **4.90x** | **3.39x** |

**Implication:** When we rank all 3,528 configs by EV, clogit configs dominate
the top because their EV is most inflated, not because their realized PnL is
highest. The top 50 configs by EV are **100% clogit**. The top 50 by actual
are **0% clogit** (44% avg_ensemble, 42% gbt, 10% cal_sgbt).

---

## 3. Why Overconfident Models Get Higher EV (Toy Example)

### Setup

A category has 5 nominees. Market prices = 20¢ each (uniform). The true winner
is nominee A.

| Nominee | Market price | True model (well-calibrated) | Overconfident model |
|---|---|---|---|
| A (winner) | 20¢ | 30% | **55%** |
| B | 20¢ | 25% | 15% |
| C | 20¢ | 20% | 12% |
| D | 20¢ | 15% | 10% |
| E | 20¢ | 10% | 8% |

### Channel 1: Position Sizing (Kelly bets more on confident picks)

The edge on nominee A:
- True model: $\text{edge} = 0.30 - 0.20 = 0.10$
- Overconfident: $\text{edge} = 0.55 - 0.20 = 0.35$

Kelly fraction (simplified independent Kelly): $f^* = \frac{q - p}{1 - p}$
- True model: $f^* = 0.10 / 0.80 = 12.5\%$ of bankroll
- Overconfident: $f^* = 0.35 / 0.80 = 43.75\%$ of bankroll → **3.5x larger**

With $1,000 bankroll, overconfident model buys **3.5x more contracts** on A.
If A wins, payoff is **3.5x larger**. If A loses, loss is also 3.5x larger.

### Channel 2: EV Weighting (model probs weight the PnL scenarios)

$$\text{EV}_\text{model} = \sum_k P_\text{model}(k) \cdot \text{PnL}_k$$

- PnL if A wins = large positive (big position at 20¢, settles at $1)
- PnL if B/C/D/E wins = negative (lose cost of A contracts, modest B/C/D/E payouts)

The overconfident model assigns **55%** weight to the A-wins scenario (high PnL)
vs 30% weight from the calibrated model. So even with identical positions, the
EV computation would be higher because it's weighting the good scenario more.

### Combined Effect

Both channels multiply. With $1,000 bankroll on this single category:

| | True model | Overconfident |
|---|---|---|
| Contracts on A | 625 @ 20¢ = $125 cost | 2,188 @ 20¢ = $437.50 cost |
| PnL if A wins | $500 | $1,750 |
| PnL if A loses | −$125 | −$437.50 |
| EV (model weights) | 0.30×500 + 0.70×(−125) = **$62.50** | 0.55×1750 + 0.45×(−437.50) = **$765.63** |
| True EV (true weights) | 0.30×500 + 0.70×(−125) = **$62.50** | 0.30×1750 + 0.70×(−437.50) = **$218.75** |

The overconfident model's **self-assessed EV = $766** is **12x** the true model's
$62.50. But its **true EV = $219** is only 3.5x higher (reflecting the larger
but still positive real edge). The gap between $766 and $219 is pure inflation
from using overconfident probabilities as both sizing inputs AND EV weights.

### Takeaway

**A model that says "I'm 55% confident" when it should say "I'm 30% confident"
will (a) bet 3.5x more aggressively and (b) weight the favorable outcome 1.8x
more in EV.** The combined effect is a dramatic EV inflation that makes the
model *look* better than it is. This is exactly what clogit does relative to
avg_ensemble.

---

## 4. Should We Use a Low-Inflation Model?

**Yes, but with nuance.** Lower inflation means the EV ranking is more
trustworthy, so EV-based parameter optimization within that model works better.

Within-model EV↔actual correlations (Spearman rank):

| Model | Spearman ρ (EV vs actual) | Inflation | EV-best actual capture |
|---|---|---|---|
| cal_sgbt | **0.976** | 1.74x | 92% |
| clogit | 0.959 | 3.39x | 85% |
| clogit_cal_sgbt_ens | 0.948 | 1.57x | 79% |
| gbt | 0.950 | 1.83x | 77% |
| avg_ensemble | **0.935** | **1.27x** | **75%** |
| lr | 0.907 | 1.87x | 75% |

"EV-best actual capture" = actual PnL of EV-optimal config / actual PnL of
hindsight-optimal config within that model. Interesting: avg_ensemble has the
lowest inflation but also the lowest capture ratio. This is because within
avg_ensemble, the EV-optimal config (`kf=0.5, edge=0.04`, very aggressive) is
quite different from the actual-optimal config (`kf=0.05, edge=0.15`,
conservative). The low edge threshold deploys more capital but on lower-quality
bets.

**Key insight:** Low inflation doesn't automatically mean better config selection.
The within-model rank correlation matters more. All models have very high
within-model Spearman (>0.90), suggesting that EV correctly ranks configs
*within a model*. The problem is **cross-model** comparison.

---

## 5. A1: Calibration — Does clogit Need More?

### Background

Clogit (conditional logit = McFadden's choice model) outputs probabilities via
per-group softmax on linear predictors:

$$P(\text{win}_i | \text{group}) = \frac{\exp(x_i^T\beta)}{\sum_j \exp(x_j^T\beta)}$$

These probabilities sum to 1 by construction and are "calibrated" in the
sense that they form a valid probability distribution. **However, this is
structural calibration (proper distribution), not statistical calibration
(predicted prob matches empirical frequency).**

A model can sum to 1 and still be overconfident (putting too much mass on
the top nominee). The 4.90x EV inflation in 2025 proves clogit IS
overconfident — it assigns probabilities that are too extreme (too much
mass near 0 and 1, not enough near the center).

### What cal_sgbt does differently

`cal_sgbt` applies a **temperature parameter** $T$ to the softmax:

$$P_i = \frac{\exp(\text{logit}(p_i) / T)}{\sum_j \exp(\text{logit}(p_j) / T)}$$

- $T > 1$: **flattens** probabilities (less confident)
- $T < 1$: **sharpens** probabilities (more confident)
- $T = 1$: no change

Clogit implicitly uses $T=1$ (no temperature adjustment). If we applied
temperature scaling to clogit's outputs with $T > 1$, we'd reduce its
overconfidence and its EV inflation.

### Would re-calibrating clogit help?

**It would reduce EV inflation but wouldn't change the actual PnL** — because
the positions are determined by the raw model probabilities and market prices
at trading time, NOT by post-hoc calibration. The positions are already locked
in during the backtest.

Calibration would only help if applied **before trading** — i.e., temperature-
scale the clogit probs, then feed those into edge/Kelly computation. This would
change the actual positions and thus actual PnL.

**Recommendation:** This is equivalent to building a `cal_clogit` model (clogit
with temperature scaling). Could be interesting but is a **model change**, not
a scoring-layer fix. Probably medium-priority — the simpler approach is to
just use a model that's already less overconfident (avg_ensemble).

---

## 6. A2: Market-Weighted EV — Empirical Investigation

### Setup

Instead of 50/50 blend (`ev_blend = 0.5 × ev_model + 0.5 × ev_market`),
sweep the market weight from 0% (pure model) to 100% (pure market):

$$\text{EV}(w) = w \cdot \text{EV}_\text{market} + (1-w) \cdot \text{EV}_\text{model}$$

### Results

| w_market | Inflation | Pearson r | Spearman ρ | Top-1 actual | Model selected |
|---|---|---|---|---|---|
| 0.00 | 3.80x | 0.778 | **0.888** | $981 | clogit |
| 0.25 | 2.84x | 0.778 | 0.887 | $981 | clogit |
| 0.50 (current) | 1.89x | 0.777 | 0.882 | $996 | clogit |
| 0.70 | 1.12x | 0.769 | 0.859 | $996 | clogit |
| 0.75 | **0.93x** | 0.762 | 0.844 | $996 | clogit |
| 0.90 | 0.36x | 0.624 | 0.693 | **$1,038** | clogit |
| 1.00 | −0.02x | −0.242 | −0.253 | $357 | lr |

### Key Findings

1. **Inflation closest to 1.0 at w≈0.70–0.75**, meaning 70-75% market weight
   would make EV approximately unbiased on average.
2. **But rank correlation *decreases* as we add more market weight.** At w=0.50,
   Spearman=0.882; at w=0.75, Spearman=0.844. Pure model (w=0) has the best
   rank correlation (0.888).
3. **The selected model is ALWAYS clogit** regardless of blend weight (until
   w=1.0 where it becomes lr). Changing the blend weight doesn't solve the
   model selection problem.
4. **The top-1 actual PnL barely changes** across the useful range ($981-$1,038).

### Conclusion

Changing the blend weight is a **marginal improvement at best**. It reduces
inflation but doesn't change which model gets selected, and the rank
correlation actually degrades. **The blend weight is not the lever that fixes
the config selection problem.** The issue is model selection, not EV weighting.

---

## 7. Alternative Objectives (B1–B4)

### B1: Rank by Actual PnL in Past Years (Leave-One-Year-Out)

**Idea:** For 2026 deployment, pick the config that had the best actual PnL
averaged across 2024+2025.

**Problem:** Cross-year actual PnL correlation is weak (Spearman ρ = 0.44):

| Selection criterion | 2024 test actual | 2025 test actual |
|---|---|---|
| Train on 2024 actual → test 2025 | — | $1,287 (42% of best) |
| Train on 2025 actual → test 2024 | $368 (36% of best) | — |
| Train on 2024 EV → test 2025 | — | $1,065 (35% of best) |
| Train on 2025 EV → test 2024 | $988 (96% of best) | — |

The actual-PnL-based selection doesn't reliably transfer across years because
which model happens to be right depends on which nominees win — highly
idiosyncratic. With only 2 years, this is basically coin-flip territory.

**Verdict:** Too little data for pure actual-based selection. Could work as
a *tiebreaker* alongside EV (see B3).

### B2: Maximize min(actual across years) — Minimax

**Idea:** Pick config with best worst-year actual PnL to avoid strategies
that get lucky in one year.

**Assessment:** Interesting from a robustness perspective. In the current
data, the minimax-optimal is likely a moderate-risk config that doesn't
bet too aggressively in either year. Worth computing but suffers from the
same N=2 years problem. Could favor overly conservative configs.

### B3: Hybrid Objective (EV + Actual)

**Idea:**

$$\text{score} = \alpha \cdot \text{EV}_\text{blend} + (1-\alpha) \cdot \text{avg\_actual\_pnl}$$

**Assessment:** This directly addresses the gap by grounding EV in reality.
The Spearman between EV and actual is 0.88 — they're correlated but not
perfectly. A hybrid would capture both the structural information in EV
(which bets have positive edge) and the empirical performance (which bets
actually worked).

**Problem:** Both components are measured on the same data. There's no
out-of-sample validation. If we tune α to make the hybrid work well on
2024+2025, we may be overfitting to those 2 years.

**Potential approach:** Use α=0.5 (equal weight) as a principled default
rather than tuning it, then evaluate on 2026 to see if it helps.

### B4: Maximize Actual PnL Subject to EV Floor (Inverse Pareto)

**Idea:** Instead of max EV s.t. risk constraint, do max actual_pnl s.t.
EV ≥ some minimum threshold, so EV acts as a sanity check rather than
the objective.

**Assessment:** This is elegant but problematic — it directly optimizes
for historical actual PnL, which is overfitting to 2 realized outcomes.
The EV floor only ensures the strategy "makes sense" structurally.

---

## 8. C1: Model-Diverse Portfolio of Configs

### Idea

Instead of picking one config, allocate capital across multiple configs
from *different models*. Each model sub-portfolio uses its own EV-optimal
config. Capital split equally.

### Empirical Results

Using each model's EV-optimal config (from the unconstrained frontier):

| Portfolio | Avg actual | 2024 | 2025 | Min year |
|---|---|---|---|---|
| clogit alone | $996 | $1,003 | $988 | $988 |
| avg_ensemble alone | $1,311 | $556 | $2,065 | $556 |
| **clogit + avg_ens** | **$1,153** | **$779** | **$1,527** | **$779** |
| clogit + avg_ens + gbt | $1,210 | $607 | $1,812 | $607 |
| All 4 models | $1,265 | $498 | $2,033 | $498 |

### Analysis

The **clogit + avg_ensemble pair** is interesting:
- clogit is strong in 2024 ($1,003), weak in 2025 ($988)
- avg_ensemble is weak in 2024 ($556), strong in 2025 ($2,065)
- Combined: $779 / $1,527 — more balanced across years

But adding more models (gbt, cal_sgbt) actually *hurts* 2024 because those
models performed poorly in 2024 despite strong 2025. The min-year gets worse.

### Does This Make Sense Conceptually?

**Yes, if models have uncorrelated errors.** The benefit of a portfolio of
configs is *diversification of model error*. If clogit is overconfident in
some categories where avg_ensemble is conservative (and vice versa), the
portfolio averages out the errors.

However, there's a key distinction from C2 (see next section): this approach
still lets EV pick each model's config independently, which may give each
sub-model an aggressive config. The avg_ensemble EV-optimal config uses
`kf=0.5, edge=0.04` which is very aggressive and somewhat unrepresentative
of avg_ensemble's best performance.

**Recommendation:** If pursuing C1, also consider using risk-constrained
(Pareto) configs per model rather than unconstrained EV-optimal.

---

## 9. C2: Fix Model, Optimize Params Only

### Idea

Since the across-model EV ranking is unreliable (clogit dominates by inflation,
not accuracy), remove model selection from the optimizer. Fix the model
(choose based on structural arguments or inflation analysis) and only use
EV to choose trading parameters (kelly_fraction, edge_threshold, etc.).

### Empirical Results

For each model, selecting the best *trading config* by EV:

| Model (fixed) | EV-best actual | Capture vs model's hind-best | Overall rank |
|---|---|---|---|
| cal_sgbt | $1,433 | 92% | #141 |
| gbt | $1,322 | 77% | #274 |
| avg_ensemble | **$1,311** | 75% | #293 |
| clogit_cal_sgbt_ens | $1,256 | 79% | #349 |
| clogit | $996 | 85% | #703 |
| lr | $960 | 75% | #800 |

**Fixing model to cal_sgbt** and using EV for params gives $1,433 actual —
far better than the unconstrained EV-optimal ($996, clogit) and close to
hindsight best ($1,756). This is the **second-best** approach tested so far.

**Fixing model to avg_ensemble** gives $1,311, also much better than clogit.

### Within-Model Parameter Sensitivity (avg_ensemble)

| Parameter | Best actual value | Best EV value | Aligned? |
|---|---|---|---|
| edge_threshold | 0.15 ($805 mean) | 0.04 ($965 mean) | **NO** — EV prefers aggressive |
| kelly_fraction | 0.05 ($766 mean) | High kf | **NO** — EV prefers aggressive |
| kelly_mode | multi_outcome ($975) | multi_outcome ($1288) | Yes |
| allowed_directions | all ($993 mean) | all ($1210 mean) | Yes |
| fee_type | maker ($710 mean) | maker ($900 mean) | Yes |

**Key tension:** Even within a well-calibrated model, EV prefers lower edge
thresholds (more trades, more capital deployed) while actual PnL prefers
higher thresholds (fewer but higher-quality trades). This is because EV
gives positive weight to marginally-positive-edge bets, while the actual
outcome is binary — near-zero-edge bets are losers as often as winners.

### Recommendation

**Fix model to avg_ensemble or cal_sgbt.** Use EV for mode/direction/fee
choices (these align well) but be skeptical of EV's preference for aggressive
edge/kelly parameters. Consider using the Pareto frontier (with risk
constraints) to select more conservative edge/kelly combos.

---

## 10. D1: Bootstrap Resampling of Category Outcomes

### Idea

Instead of computing actual PnL as a single number (settle all categories
against known winners), resample which categories are included and which
nominees win. This gives uncertainty bands on actual PnL.

### How It Would Work

For a given ceremony year:
1. There are K categories (K=8 for 2024, K=9 for 2025)
2. For each bootstrap sample b = 1, ..., B:
   a. Draw K categories with replacement from the K available
   b. For each drawn category, keep the actual winner
   c. Compute portfolio PnL for this resampled set
3. Now we have B bootstrap PnL samples per config
4. Report confidence intervals, rank stability, etc.

### What This Addresses

- **Category concentration risk**: If a config's strong performance comes
  from 1-2 categories that happened to have large payoffs, bootstrap will
  show high variance (the config isn't robustly good)
- **Rank stability**: If a config is #1 in 80% of bootstrap samples, it's
  more robustly #1 than a config that's #1 in 20% of samples
- **Small-sample correction**: With only 8-9 categories, one lucky/unlucky
  category can dominate. Bootstrap helps quantify this.

### Limitations

- Still can't resample *which nominees win* without making distributional
  assumptions (that's what the MC CVaR already does with model probs)
- Only resamples which categories contribute, not the outcome uncertainty
  within each category
- K=8-9 is small for bootstrap — some practitioners recommend K≥20 for
  reliable bootstrap CIs

### Assessment

**Moderately useful.** Would help quantify how robust the actual-PnL rankings
are. If the top-10-by-actual configs have overlapping bootstrap CIs, we know
not to over-index on point rankings. Could be implemented relatively easily
by reshuffling the per-category PnL columns in `scenario_pnl.csv`.

---

## 11. Portfolio of Configs vs Single Ensemble Model

### The Question

If we already have `avg_ensemble` (which averages probabilities across models
before trading), what do we gain from splitting capital across separately-
traded model-specific configs?

### Key Distinction

| Aspect | avg_ensemble (prob-level) | Portfolio of configs (capital-level) |
|---|---|---|
| Averaging happens | Before edge computation | After PnL realization |
| Position sizing | Single set of Kelly bets on averaged probs | Model-specific Kelly bets, then avg PnL |
| Edge quality | Moderate edges (averaged out extremes) | Some high edges (model-specific) |
| Diversification | Probs diversified, positions concentrated | Positions diversified |
| Model disagreement | Cancels out in prob space | Each model trades its own view |

### When Portfolio Helps More

1. **When models disagree strongly**: If clogit says A wins and gbt says B wins,
   avg_ensemble gives both moderate probability, producing weak edges on both.
   A portfolio has a strong clogit bet on A and a strong gbt bet on B — whoever
   turns out right, the portfolio benefits from one model's conviction.

2. **When Kelly sizing is convex in edge**: Larger edges produce disproportionately
   larger bets. Averaging probs before Kelly may undersize compared to the
   average of model-specific Kelly bets (Jensen's inequality effect).

### When Ensemble Helps More

1. **When models are wrong in similar ways**: If all models are overconfident on
   the same nominee, the portfolio amplifies this error. The ensemble at least
   averages the overconfidence down.

2. **When the true winner has moderate support across models**: Ensemble
   averages give a coherent signal. Portfolio trades cancel each other.

### Empirical Evidence

From the C1 analysis:
- avg_ensemble alone: $1,311 avg actual
- 4-model portfolio: $1,265 avg actual
- clogit+avg_ens portfolio: $1,153 avg actual

**avg_ensemble alone beats all tested portfolios on average.** This suggests
that for Oscar prediction, the prob-level ensemble is more effective than
capital-level diversification. The models' errors are correlated enough that
portfolio-level diversification doesn't add much.

**But min-year is a different story**: clogit+avg_ens has min_year=$779 vs
avg_ensemble's min_year=$556. The portfolio is more *robust* even if its
average is lower.

---

## 12. Can We Trust 2 Years of Data?

### The Problem

With N=2 years (2024, 2025), each with 8-9 categories, we have ~17
category-level outcomes to learn from. This is extremely thin for:
- Estimating which model is best (model selection)
- Estimating optimal trading parameters (parameter selection)
- Validating the EV metric against realized performance

### Cross-Year Stability

| Metric | Value |
|---|---|
| Spearman correlation(actual_2024, actual_2025) | **0.44** |
| Best model 2024 | clogit (by actual) |
| Best model 2025 | cal_sgbt (by actual) |
| % configs profitable both years | 89% |
| Mean actual PnL | $622 |

The 0.44 Spearman correlation means year-to-year rankings are weakly related.
The best model *flips* between years. Most configs are profitable (this is
mostly a function of having positive edges on average, not model skill).

### What We CAN Infer

1. **All models have positive expected edge** — 89% of configs are profitable
   in both years. The market is beatable.
2. **EV directionally correct** — Spearman ρ(EV, actual) = 0.88 overall.
   Higher EV configs tend to have higher actual PnL.
3. **Model overconfidence varies predictably** — clogit is consistently most
   overconfident. This is a structural property, not a 2-year artifact.
4. **Multi-outcome Kelly > independent Kelly** — consistent across both years
   and all models.
5. **Side=all > yes-only > no-only** — consistent across years.

### What We CANNOT Infer

1. **Which model is best** — flips between years, N=2 is not enough
2. **Optimal edge threshold** — 0.04 vs 0.15 makes a huge difference but
   we can't distinguish with 2 years
3. **Optimal kelly_fraction** — similar issue

### How To Build More Trust

**Option 1: More backtest years.** If we can get Kalshi market data for earlier
Oscar ceremonies (2022, 2023), backtest those too. Each additional year helps
significantly since N is so small.

**Option 2: Cross-category analysis within years.** Instead of treating each
year as one observation, treat each (year × entry_point × category) as a
separate observation. This gives ~100+ data points but they're not independent
(categories within a year share the same model training data).

**Option 3: Accept the uncertainty and focus on robust strategies.** Pick
configs that are good-enough across many plausible futures rather than
trying to find the single best config. This points toward:
- Conservative edge thresholds (higher is more robust)
- Low-inflation models (less sensitive to prob accuracy)
- Risk constraints (CVaR to limit downside)

---

## 13. Recommendations

Based on all evidence, here is a **prioritized list** of actionable changes:

### Tier 1: High Impact, Low Effort

**R1: Fix model to avg_ensemble (or cal_sgbt), drop clogit from config selection.**

Rationale: clogit's 3.4x EV inflation makes it dominate EV rankings despite
not being the best actual performer. By fixing the model, we remove the biggest
source of error from config selection. avg_ensemble has the lowest inflation
(1.27x) and was the hindsight best. cal_sgbt has the best within-model capture
(92%).

Action: In `scenario_scoring.py`, filter to a single model before running
Pareto. Or simply report per-model Pareto frontiers and pick from the
avg_ensemble or cal_sgbt frontier.

Expected impact: ~$300-400 additional actual PnL vs current clogit-based
selection.

**R2: Use moderate edge threshold (0.10-0.15) rather than EV-optimal (0.04-0.06).**

Rationale: Within avg_ensemble, higher edge thresholds have higher actual PnL
despite lower EV. The EV-preferred low thresholds deploy capital into
marginal bets that don't reliably pay off.

Action: Constrain the edge threshold search range, or pick from the 0.10-0.15
portion of the Pareto frontier.

### Tier 2: Medium Impact, Medium Effort

**R3: Report per-model Pareto frontiers alongside the combined one.**

This gives the user visibility into model-specific risk-return tradeoffs
rather than always seeing clogit dominate. No changes to methodology, just
additional output tables.

**R4: Add "EV inflation" as a diagnostic metric.**

For each config, report `ev_pnl_blend / actual_pnl` when printing the Pareto.
This flags configs whose EV is unrealistically high.

**R5: Investigate a `cal_clogit` model (clogit + temperature scaling).**

If clogit's structural properties are valuable (proper multinomial, group-
aware training) but its overconfidence is problematic, adding a learnable
temperature parameter could give best of both worlds. Medium modeling effort.

### Tier 3: Worth Investigating, Higher Effort

**R6: Bootstrap analysis (D1) to quantify actual PnL uncertainty.**

Would help distinguish between genuinely robust configs and those that got
lucky. Moderate implementation effort but high diagnostic value.

**R7: Hybrid objective (B3) with equal weights (α=0.5).**

Would be interesting to see if this changes the Pareto frontier meaningfully.
Risk of overfitting to 2 years.

**R8: Expand to more backtest years if data is available.**

Most impactful long-term improvement. Each additional year of Oscar market
data dramatically improves our ability to validate the framework.

### Not Recommended

- **A3 (Shrink model probs)**: Redundant with A1/R5 (temperature scaling)
- **B4 (Inverse Pareto)**: Overfits to realized outcomes
- **C3 (Bayesian averaging)**: Overengineered for N=2 years of data
- **D2 (Synthetic years)**: Would create unrealistic correlations

---

## 14. Open Questions

1. **Should we run the 2026 live deployment with avg_ensemble instead of clogit?**
   The evidence says yes, but we'd be making a model switch very close to
   deployment based on backtest analysis of historical data. Are we comfortable
   with this?

2. **Can we get Kalshi market data for 2022 and 2023 Oscar ceremonies?**
   Even rough data would help. Two more years would more than double our
   validation set.

3. **For the Pareto risk-return selection, should the "return" axis be:**
   - EV_blend (current — correlated with actual but inflated)
   - Actual PnL (purely backward-looking — only 2 data points)
   - Hybrid (B3 — principled but possibly overfit)
   - Per-model EV (R1 — avoids cross-model inflation)

4. **Is multi-model capital portfolio (C1) worth implementing?**
   The empirical evidence is mixed — avg_ensemble alone beats portfolios on
   average, but portfolios have better worst-year performance. Depends on
   whether we prioritize expected return or robustness.

5. **What is the actual temperature that would minimize clogit's inflation?**
   A quick experiment: sweep T from 0.5 to 3.0 on clogit probabilities,
   recompute EV, check inflation ratio. Could inform R5.

6. **Should edge_threshold be a "risk dial" separate from the optimization?**
   Currently it's swept as part of the config grid. An alternative: let the
   user choose edge_threshold based on risk preference (like loss_bound_pct
   for CVaR), and only optimize kelly_fraction and mode.
