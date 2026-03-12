# Plan: EV + Risk-Constrained Config Selection

**Branch:** `feature/buy-hold-backtest`
**Date:** 2026-02-28 (initial), 2026-03-01 (QA + CVaR extension)
**Storage:** `storage/d20260225_buy_hold_backtest/`

## Motivation

The existing robustness score is an ad-hoc weighted composite of rank-based
components (35% P&L rank, 20% worst-category rank, 15% Sharpe rank, 15%
neighbor stability, 10% profitable fraction, 5% loss-bounded flag). Problems:

1. **Ad-hoc weights** (35/20/15/15/10/5) — no principled basis.
2. **Rank-based** — insensitive to absolute differences between configs.
3. **Known-winner only** — "worst case" means "which category lost most given
   the actual winner," not "what if the model is wrong about who wins."
4. **Binary loss bound** — loss_bounded is 0/1 at -20%, not granular.
5. **Worst case obscures probability** — the portfolio worst assumes every
   category simultaneously gets its worst possible winner, an event with
   probability ~$10^{-9}$. This is far too conservative for decision-making.

We replace this with a principled EV + risk-constrained framework that
examines **all possible winner scenarios** and uses both worst-case and
**CVaR (Conditional Value at Risk)** constraints.

---

## Design

### Core Metrics

For each (category, entry_point, config), we have `result.settlements` — a
dict mapping every possible winner to a `SettlementResult`. Combined with
model/market probabilities, we compute:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Worst PnL** | $\min_w \text{PnL}(w)$ | Dollar loss if worst nominee wins |
| **Best PnL** | $\max_w \text{PnL}(w)$ | Dollar gain if best nominee wins |
| **EV PnL (model)** | $\sum_w p_{\text{model}}(w) \cdot \text{PnL}(w)$ | Expected PnL using model beliefs |
| **EV PnL (market)** | $\sum_w p_{\text{market}}(w) \cdot \text{PnL}(w)$ | Expected PnL using market prices |
| **EV PnL (blend)** | Average of model and market EV | Hedged expected PnL |
| **CVaR_α** | $E[\text{PnL} \mid \text{PnL} \leq \text{VaR}_\alpha]$ | Expected PnL in worst α% of joint outcomes |
| **Capital deployed** | Sum of position outlays | Dollars at risk |
| **Actual PnL** | Settlement for known winner | What actually happened |

### Per-Winner PnL Persistence

The full per-winner PnL distribution is persisted in `scenario_pnl.csv`:

```
category, model_type, entry_snapshot, config_label, nominee, pnl, model_prob, market_prob
```

This enables CVaR computation without re-running backtests. Each entry_pnl
row expands to ~5-10 scenario rows (one per nominee). Estimated size: ~1.5M
rows for 2025, ~50MB.

### Aggregation Levels

1. **Per entry point** — one row per (category, entry, config). Raw scenario data.
2. **Per-entry-time portfolio** — sum across categories at one entry time.
   Unit of deployment: one entry = all categories at one point in time.
3. **Config-level** — **average** across entry times. Represents expected
   single-deployment performance across the Oscar season.
4. **Cross-year** — average of per-year config-level scores.

#### Per-Entry-Time Normalization (Key Design Choice)

**Rationale:** In live trading, you choose ONE entry time (e.g., post-DGA)
and deploy across all categories at that moment. Averaging over entry times
gives a robust estimate of single-deployment performance across the season.

**Previous approach** summed across all entry times, making the portfolio EV
proportional to number of entry points (7 for 2024, 9 for 2025). This created
an artificial 1.45× bankroll asymmetry between years.

**New approach:**
1. For each entry time: sum metrics across categories → per-entry portfolio
2. Average per-entry portfolios across entry times → config-level score

This makes the effective per-entry bankroll = $N_\text{cats} \times \$1{,}000$:
- 2024: $8,000 (8 categories)
- 2025: $9,000 (9 categories)
- Ratio: 1.125× (vs previous 1.446×)

The residual 12.5% difference (8 vs 9 categories) is a real structural
difference — 2025 has Cinematography as an additional category. We express
all metrics as **% of per-entry bankroll** to normalize this away.

**Empirical validation:** Spearman ρ = 0.9998 between new and old rankings,
top-10 configs identical. The normalization doesn't change which configs are
best; it changes the interpretation and scale of the numbers.

**Open note (Option A vs Option B for worst-case):** We chose Option A —
average worst-case across entry times ("expected worst-case at a typical
entry point"). Option B — max worst-case across entries (more conservative)
— was considered but rejected for consistency: both EV and worst-case should
represent the same "pick a single entry time" frame.

#### Portfolio-Level Worst Case

Per-entry worst case = $\sum_c \min_{w_c} \text{PnL}_c(w_c)$.

Categories have independent winners, so the portfolio worst assumes the worst
possible winner in EVERY category simultaneously. This is conservative — the
joint probability of this event is $\prod_c P(\text{worst}_c)$, typically
~$10^{-9}$ with 9 categories. CVaR addresses this by weighting outcomes by
probability.

### Config Selection: Two Constraint Types

#### Type 1: Worst-Case Bound (original)

$$\max_{\text{config}} \; \text{EV\_PnL}(\text{config}) \quad \text{s.t.} \quad \text{WorstPnL}(\text{config}) \geq -L \times B$$

Sweep $L \in \{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00\}$.
Equivalent to CVaR at $\alpha = 0\%$ (absolute worst).

#### Type 2: CVaR Bound (new)

$$\max_{\text{config}} \; \text{EV\_PnL}(\text{config}) \quad \text{s.t.} \quad \text{CVaR}_\alpha(\text{config}) \geq -L \times B$$

Fix $\alpha \in \{5\%, 10\%, 25\%\}$, sweep $L$ at same points.

**Presentation:** Overlay Pareto frontiers from different α values on one
figure. The α=0% line (worst-case) is most restrictive; higher α frontiers
admit more configs → higher achievable EV at the same L. This directly answers:
"how much more EV do I unlock by using CVaR_5% instead of worst-case?"

### CVaR Computation via Monte Carlo

**Why Monte Carlo:** Per-category, there are only 5–10 possible winners, so
category-level CVaR is trivial. But at the portfolio level (9 categories),
the joint outcome space is $10 \times 5^8 \approx 3.9M$ scenarios. Monte Carlo
is the most practical way to sample this space.

**Algorithm:**
1. For each category at a given (entry_time, config): load per-winner PnLs
   and blend probabilities from `scenario_pnl.csv`
2. Draw N independent samples: for each sample, draw one winner per category
   from the blend probability distribution
3. Compute portfolio PnL per sample = sum of per-category realized PnLs
4. Sort samples ascending. CVaR_α = mean of bottom ⌊α × N⌋ samples.

**Sample size calibration:** Sweep N ∈ {1K, 5K, 10K, 50K, 100K, 500K} with
10 repeated seeds. Plot CVaR estimate ± std. Pick N where std < 1% of the
estimate. Include calibration results in the README.

**Probability weighting:** Use blend probabilities (avg of model + market)
as the default for MC draws. This is consistent with EV blend.

### Cross-Year Aggregation

- **EV**: average of 2024 and 2025 config-level EV (per-entry-time averaged)
- **Worst case**: must pass $\geq -L \times B$ independently in BOTH years
- **CVaR**: must pass $\geq -L \times B$ independently in BOTH years
  (each year's CVaR computed from that year's distribution)

**Note:** An alternative "pooled" cross-year CVaR (treating each year as a
50% probability scenario, then MC within each) was considered but rejected
for consistency with the worst-case framework.

### Table Enrichment

All Pareto frontier tables include:

| Column | Formula | Meaning |
|--------|---------|---------|
| L | Loss bound % | Risk tolerance parameter |
| Avg EV | See aggregation | Expected single-entry PnL |
| Actual 2024/2025 | Realized PnL | What actually happened |
| Worst 2024/2025 | See worst-case | Absolute worst scenario |
| CVaR_5% | MC | Expected PnL in worst 5% of outcomes |
| Cap Deployed | Avg capital deployed at one entry | Actual dollars at risk |
| Deploy % | Cap / Bankroll | Fraction of bankroll used |
| ROI on Deployed | Actual PnL / Cap Deployed | Capital efficiency |
| ROI on Bankroll | Actual PnL / Per-entry bankroll | Return on allocation |
| Config | Trading parameters | Which config is Pareto-optimal |

---

## Decisions & Assumptions

### Settled (Phase 1: 2026-02-28)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Optimization criterion = absolute PnL in dollars | ROI on outlay is constant across KF for a given market; only dollar PnL differentiates configs. Ranking invariant to constant bankroll scaling. |
| 2 | Report ROI on outlay as diagnostic (not optimizer target) | Measures per-bet quality, can't rank configs by itself |
| 3 | Probability weights for EV: model, market, and average | Model EV is the trading thesis; market EV is consensus check; blend hedges |
| 4 | Portfolio-level worst case = sum of per-category worst cases | Categories have independent winners; no combinatorial explosion |
| 5 | Zero-capital configs included at 0 PnL | Config isn't triggered → no risk, no return. Included for completeness. |
| 6 | Cross-year: average EV, constraints must pass in BOTH years | Conservative: config must not blow up in either year |
| 7 | Replace old robustness score completely | New system subsumes all useful components |
| 8 | Must re-run backtests to persist all-winner settlements | Current CSVs discard settlement data; entry_pnl.csv only has actual-winner PnL |

### Settled (Phase 2: 2026-03-01 — QA + CVaR extension)

| # | Decision | Rationale |
|---|----------|-----------|
| 9 | Per-entry-time normalization: average across entries, not sum | Matches live trading use case (pick one entry time). Reduces cross-year bankroll asymmetry from 1.45× to 1.125×. Rank correlation with old method: ρ = 0.9998 — same configs selected, better interpretation. |
| 10 | Average worst-case across entries (Option A) | Consistent with EV averaging. Represents "expected worst-case at a typical entry." Alternative (Option B: max worst-case) is more conservative but inconsistent with EV's averaging frame. |
| 11 | Report all metrics as % of per-entry bankroll | Per-entry bankroll = N_cats × $1K. 2024: $8K, 2025: $9K. Normalizes the residual 8-vs-9 category difference. |
| 12 | Monte Carlo CVaR at portfolio level | Per-category distributions have only 5–10 outcomes (too coarse for meaningful tail stats). Portfolio-level joint space is ~3.9M — MC sampling is practical and gives stable estimates. |
| 13 | Persist per-winner PnL + probs in scenario_pnl.csv | Enables CVaR computation without backtest re-runs. ~50MB per year. |
| 14 | CVaR Pareto: fix α, sweep L (Approach 1) | Natural extension of worst-case Pareto. Overlay α ∈ {0%, 5%, 10%, 25%} curves on one figure. α=0% = worst-case (existing). Higher α → less conservative → more feasible configs → higher achievable EV. |
| 15 | Cross-year CVaR: independent per-year constraints | Same structure as worst-case. Alternative (pooled) rejected for consistency. |
| 16 | Keep $1K/category in actual backtest, normalize in reporting | Avoids discretization issues at very small bankrolls ($125/cat). Post-hoc normalization is exact for independent Kelly (position size ∝ bankroll) and approximately correct for multi-outcome Kelly. |

### Key Finding: Multi-Outcome Kelly Ignores kelly_fraction

The `multi_outcome_kelly()` optimizer maximizes $E[\log W]$ — a concave
function with a unique global maximum. `kelly_fraction` only seeds the
initial guess (`x0`); the optimizer converges to the same allocation regardless.

Verified empirically with toy example: KF=0.10, 0.25, 0.50, 1.00 all produce
identical multi-outcome allocations. This means ~half the grid is redundant
for multi-outcome configs.

**Decision**: Note this in analysis. Don't re-run with a collapsed grid (too
disruptive), but collapse duplicate results when computing scores.

### Key Finding: Bankroll Normalization Doesn't Change Rankings

Empirically verified three approaches:
- **A** ($1K/cat, entry-avg): sum across categories per entry, avg across entries
- **B** ($1K total, entry-avg): same as A but divide by N_cats
- **C** (return %): ev / per-entry bankroll

Results: ρ(A,B) = 0.9998, ρ(B,C) = 1.0000, top-10 overlap = 10/10.
B and C are mathematically identical (same rescaling). A differs only because
cross-year averaging applies different denominators (8 vs 9) — effect is a
max rank difference of 86 out of 3,528 configs with mean shift of 13.3 ranks.

**Decision**: Use Approach A in the scorer ($1K/cat) but express results as %
of per-entry bankroll for apples-to-apples cross-year comparison.

### Proportional Scaling Destroys Risk Bounds

If Kelly recommends deploying $300 of a $1000 bankroll, and you scale up 3.3×
to deploy the full $1000, the worst case goes from -30% to -100% of bankroll.
Kelly's allocation is optimal precisely because it balances EV against ruin risk.

**Decision**: Evaluate configs on what Kelly recommends (not on scaled-up
allocations). Report deployment rate as a diagnostic. Scaling for live trading
is a separate decision.

---

## Implementation Plan

### Step 1: Modify run_backtests.py to persist per-winner scenarios

**1a.** Keep existing columns in `entry_pnl.csv`:
- `worst_pnl`, `best_pnl`, `ev_pnl_model`, `ev_pnl_market`, `ev_pnl_blend`
- `actual_pnl`, `capital_deployed`, `n_positions`

**1b.** Add new output: `scenario_pnl.csv` with per-winner breakdown:
- `category, model_type, entry_snapshot, config_label, nominee, pnl, model_prob, market_prob`
- One row per (category, entry, config, nominee)
- Size: ~1.5–2.8M rows for 2025, ~50MB

### Step 2: Update scenario scoring module

Modify `scenario_scoring.py`:

- **Per-entry-time normalization**: `compute_portfolio_scores()` now averages
  across entry times instead of summing. First sum across categories per entry
  time, then average across entry times → one row per config.
- **Monte Carlo CVaR**: new function `compute_portfolio_cvar()` that:
  1. Loads `scenario_pnl.csv`
  2. For each (entry_time, config), does MC sampling across categories
  3. Computes CVaR_α at portfolio level per entry time
  4. Averages CVaR across entry times
- **CVaR Pareto frontier**: new function `compute_cvar_pareto()` — same as
  `compute_pareto_frontier()` but uses CVaR_α as the constraint instead of
  worst_pnl. Sweep (α, L) grid.
- **MC sample size calibration**: function `calibrate_mc_sample_size()` —
  runs CVaR at N ∈ {1K, 5K, 10K, 50K, 100K, 500K} × 10 seeds for a few
  representative configs. Reports convergence.

### Step 3: Update visualization script

Add to `analyze_scenario_plots.py`:

1. **CVaR Pareto overlay**: α ∈ {0%, 5%, 10%, 25%} on one figure per year and cross-year
2. **MC convergence plot**: CVaR estimate ± std vs N for sample size calibration
3. **CVaR vs worst-case scatter**: compare the two risk measures per config

### Step 4: Table enrichment

Add to Pareto frontier tables: capital deployed, deploy %, ROI on deployed,
ROI on bankroll, CVaR columns.

### Step 5: Re-run and analyze

1. Re-run backtests for 2024 and 2025 (persists scenario_pnl.csv)
2. Run scenario scoring (with per-entry normalization + MC CVaR)
3. Generate plots (including CVaR Pareto overlay + MC convergence)
4. Generate enriched tables

### Step 6: Update README

- Preserve existing worst-case Pareto section (updated with per-entry-time
  normalization and enriched table columns)
- Add new section: CVaR Config Selection (methodology + MC calibration + Pareto
  frontiers at α ∈ {5%, 10%, 25%})
- Add comparison section: Worst-Case vs CVaR — where they agree, where they
  diverge, which to use for live trading
- Add recommedation with reference to both frameworks
