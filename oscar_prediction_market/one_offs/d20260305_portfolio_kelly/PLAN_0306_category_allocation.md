# PLAN: Category Allocation Investigation (Rethink)

**Date:** 2026-03-06
**Context:** Original portfolio-kelly investigation used EV as the sole
reallocation signal. The concurrent config-selection-sweep investigation
(§4: "Why Not EV or CVaR for Config Selection?") showed that EV is
structurally inflated and anti-correlated with actual P&L within the
targeted grid (ρ = -0.57 to -0.90 across configs). This plan rethinks
the approach given what we now know.

---

## What Changed

1. **EV is unreliable.** Config sweep showed within-model EV↔actual Spearman ρ
   of -0.89 for avg_ensemble across configs. The mechanism: lower edge thresholds
   inflate EV via more noise trades. Within a fixed config, category-level EV↔actual
   was ρ = -0.20 (2024, n=4 active cats) and ρ = +0.93 (2025, n=7 active cats).
   One year says useless, one says great — with n=2 years we cannot trust it.

2. **Cross-year category P&L is anti-correlated (ρ = -0.65).** Categories that
   profit in 2024 tend to lose in 2025 and vice versa. Historical category P&L
   is NOT a prospective signal for allocation.

3. **All prior results used stale data.** Portfolio-kelly scripts read from
   `storage/d20260225_buy_hold_backtest/` (the old full grid, pre-refactor).
   Must switch to `storage/d20260305_config_selection_sweep/` (targeted grid,
   post-refactor).

---

## The Right Framing

### What we care about

**Portfolio-level P&L at a single entry time** = Σ(category P&L × allocation weight).
We do NOT care about per-category P&L ranking. A reallocation strategy is good if
it produces higher portfolio P&L (and acceptable risk) across entry points and years.

### The idle-bankroll decomposition

Empirical finding with new data (avg_ensemble, edge=0.20, KF=0.05):

```
Year  Active cats  Uniform P&L  Equal-Active P&L  Oracle P&L
2024  2.9 / 8      $4,847        $13,707          $37,803
2025  4.1 / 9      $24,542       $53,586          $141,793
```

At edge=0.20, ~3-4 of 8-9 categories are active. **Half the bankroll sits idle.**

- Equal-among-active (no signal, just redistribute idle bankroll to active cats)
  gives **2-3× uniform P&L**
- But captures only **25-27% of oracle uplift** — the oracle knows which active
  categories are winners, equal-among-active amplifies both winners and losers

This decomposes the allocation problem into two layers:
1. **Layer 1: Idle bankroll redistribution** — trivial, requires no signal,
   captures ~25% of oracle upside
2. **Layer 2: Within-active weighting** — requires a signal, captures the rest

### The signal problem

Layer 2 only helps if we have a signal that ranks active categories by future
P&L. Available prospective signals (known at entry time T):

| Signal | What it measures | Known issue |
|--------|-----------------|-------------|
| `ev_pnl_blend` | Model's expected P&L per category | Inflated; 1 of 2 years says useless |
| `ev_pnl_model` / `ev_pnl_market` | Component EVs | Same inflation problem |
| `capital_deployed` | How much the model bets in category | Correlated with EV |
| `n_positions` | Number of trades taken | More trades = more diversification? |
| Mean edge per trade | Avg(model_prob - market_prob) | Quality signal, not volume |
| Max nominee edge | Largest single edge in category | Measures "best opportunity" |

No signal is validated for cross-year category ranking. The deep problem:
with only 2 years and 8-9 categories, we have ~5 effective independent
observations (per the config sweep eigenvalue analysis). Not enough to
validate any prospective signal.

---

## Goals

1. Establish whether category reallocation is worth doing at all (vs uniform)
2. Find the best reallocation strategy — decomposing into Layer 1 (idle redistribution)
   vs Layer 2 (within-active weighting)
3. Test EV and alternative prospective signals for Layer 2
4. Answer: does the recommended (model, config) change when reallocation is added?
   Joint (config, allocation) optimization.
5. Produce a concrete recommendation for live trading allocation.

---

## Proposed Analyses

### Analysis 1: Oracle and equal-active decomposition

Establishes the ceiling and the easy wins.

For each (config, entry_point, year):
1. Uniform: $1000 per category → portfolio P&L
2. Equal-active: redistribute idle bankroll to active categories → portfolio P&L
3. Oracle: allocate proportional to max(actual_pnl, 0) → portfolio P&L

Report: per-config uplift distributions, fraction of oracle captured by equal-active,
sensitivity to edge threshold. This tells us: **how much room is there beyond the trivial
redistribution?**

### Analysis 2: Prospective signal comparison

For each candidate signal, at each (config, entry_point):
- Compute per-category signal values
- Allocate proportional to signal (with various aggressiveness and caps)
- Evaluate portfolio P&L
- Compare to uniform and equal-active baselines

Candidate signals:
- **EV-proportional** (current approach) — use ev_pnl_blend
- **Edge-magnitude-proportional** — use mean edge per active trade in category
- **Capital-proportional** — use capital_deployed per category
- **Equal-among-active** (Layer 1 only, no Layer 2 signal)

For each signal, test:
- Pure proportional (aggressiveness=1.0)
- Blended at 25%, 50%, 75% toward proportional
- With concentration caps (20%, 30%, 50% of total bankroll per category)

Evaluation metric: portfolio-level P&L improvement over uniform, averaged across
entry points, for each year. Also: CVaR-5%, P(outperform uniform).

### Analysis 3: Noise sensitivity (revalidation)

Same concept as before but with new data and all signals. Add noise to each
signal and measure degradation. If a signal survives 100% noise, its value
is primarily from Layer 1 (idle redistribution), not Layer 2.

### Analysis 4: Joint (config, allocation) optimization

The key interaction: **edge threshold determines which categories are active.**
- Low edge (0.02): almost all categories active → reallocation ≈ uniform
- High edge (0.20): few categories active → reallocation concentrates heavily
- Very high edge (0.25): 1-3 active → extreme concentration

This means optimal edge might differ between uniform and reallocation regimes.

For each of the 27 configs × each allocation strategy:
- Compute portfolio P&L across all entry points, both years
- Build combined Pareto frontier (PnL vs CVaR) over (config, allocation) pairs
- Compare: does the best config change? Is it the same edge=0.20 that config
  sweep recommended, or does reallocation prefer a different threshold?

### Analysis 5: Cross-year strategy stability

Same as before: rank strategies by portfolio P&L separately for 2024 and 2025.
Spearman ρ across years. But now testing all signal variants, not just EV blends.

### Analysis 6: Entry-point robustness

For the top 3-5 strategies: what fraction of entry points show positive uplift
vs uniform? Is the benefit concentrated in a few entries or spread across all?

---

## Rename Plan

### One-off directories

| Current | New |
|---------|-----|
| `d20260305_config_selection_sweep` | `d20260305_model_config_selection` |
| `d20260305_portfolio_kelly` | `d20260305_category_allocation` |

### Storage directories (shared symlink)

| Current | New |
|---------|-----|
| `storage/d20260305_config_selection_sweep/` | `storage/d20260305_model_config_selection/` |
| `storage/d20260305_portfolio_kelly/` | `storage/d20260305_category_allocation/` |

### What needs updating after rename

For `model_config_selection`:
- `git mv` one-off dir
- `mv` storage dir
- Update: EXP_DIR in 3 scripts, SWEEP_MODULE in run.sh, docstrings, image alt-text
- Update: experiment_index.md

For `category_allocation`:
- `git mv` one-off dir
- `mv` storage dir
- Update: OUTPUT_DIR in 6 scripts, module paths in docstrings
- Update: imports in robustness_analysis.py (from pareto_frontier)
- Update: experiment_index.md

---

## Script Plan

| Script | Action | Purpose |
|--------|--------|---------|
| `analyze_edge_by_category.py` | **Rewrite** | Category heterogeneity + idle bankroll analysis |
| `quantify_reallocation_value.py` | **Rewrite** | Oracle decomposition (Analysis 1) |
| `pareto_frontier.py` | **Rewrite** | Multi-signal comparison + joint config optimization (Analysis 2, 4) |
| `robustness_analysis.py` | **Rewrite** | Noise sensitivity, cross-year stability, entry-point robustness (Analysis 3, 5, 6) |
| `simulate_portfolio_kelly.py` | **Rewrite** | MC portfolio simulation with multiple signals |
| `joint_kelly_optimizer.py` | **Rewrite** | Joint (config, allocation) optimization |

All scripts:
- Switch data source to `storage/d20260305_model_config_selection/`
  via `load_entry_pnl(year, exp_dir=...)`
- Restrict to avg_ensemble model
- Test across all 27 targeted-grid configs

### Data source change

All scripts currently call:
```python
load_entry_pnl(year)  # → reads storage/d20260225_buy_hold_backtest/
```

Change to:
```python
EXP_DIR = Path("storage/d20260305_model_config_selection")
load_entry_pnl(year, exp_dir=EXP_DIR)  # → reads targeted grid data
```

---

## README Rewrite Plan

Aggressive rewrite. New structure:

1. **Motivation** — uniform allocation wastes half the bankroll; quantify idle capital
2. **Idle bankroll decomposition** — oracle, equal-active, uniform comparison
3. **Signal comparison** — EV vs edge-magnitude vs capital vs equal-active
4. **Joint (config, allocation) Pareto frontier** — does optimal config change?
5. **Robustness** — noise sensitivity, cross-year stability, entry-point spread
6. **Recommendation** — which allocation to use in live trading

Remove:
- Per-category P&L correlation analysis (wrong framing)
- Joint Kelly optimizer prose about "addresses a different problem"
- Old EV-only Pareto frontier tables

Keep (but re-generate with new data):
- Strategy ranking cross-year stability
- Noise sensitivity
- Entry-point robustness
- Config sensitivity

---

## Open Questions

### Q1: Does the edge-magnitude signal make sense?

"Mean edge per trade" = avg(model_prob - market_prob) across all trades in a
category at that entry point. This measures edge *quality* rather than edge
*quantity* (which is what EV measures). It avoids the trade-count inflation
problem.

But: with multi_outcome Kelly, positions are sized proportionally to edge
already. So capital_deployed ≈ f(n_trades × edge). The signals might be
highly correlated, in which case testing multiple signals is theater.

**Plan:** Compute pairwise correlations between all candidate signals first.
If they're all ρ > 0.8, we're really just testing one signal with slight
variants. If they diverge, genuine comparison is possible.

### Q2: Interaction between edge threshold and reallocation

Hypothesis: with reallocation, the optimal edge threshold might be HIGHER
than 0.20. Rationale: at high edge, fewer categories are active, so idle
bankroll is larger, so reallocation captures more value. The cost of high
edge (fewer trades) is offset by the benefit (more concentration on genuine
edge). Analysis 4 will test this directly.

Counter-hypothesis: with reallocation, you can LOWER the edge threshold because
the idle-bankroll redistribution compensates for having more marginal trades.
You take more trades (more diversification) and reallocation concentrates on
the ones that work.

### Q3: Is Layer 2 worth pursuing at all?

Equal-among-active captures 25-27% of oracle uplift. To capture more, we need
a signal that ranks active categories. But we only have 2 years of data and
3-7 active categories per entry point. Any signal "validated" on this data is
essentially noise.

Possible conclusion: **just use equal-among-active** and accept that we can't
do better with the data we have. The recommendation becomes:
"redistribute idle bankroll to active categories, equal weight among them."
Simple, robust, no model dependency beyond the edge threshold.

We should let the data decide: if no signal consistently beats equal-among-active
across both years and all configs, that IS the recommendation.

---

## Execution Order

1. Rename directories (one-off + storage + all references)
2. Run Analysis 1 (oracle decomposition) — establishes ceiling
3. Run Analysis 2 (signal comparison) — finds best signal
4. Run Analysis 3 (noise sensitivity) — validates robustness
5. Run Analysis 4 (joint config optimization) — checks config interaction
6. Run Analysis 5+6 (cross-year + entry-point) — final robustness
7. Write README with findings
