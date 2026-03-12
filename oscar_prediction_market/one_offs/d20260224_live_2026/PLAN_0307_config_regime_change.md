# PLAN: Config Regime Change — avg_ensemble + maxedge_100 Allocation

**Created:** 2026-03-07
**Status:** Ready for implementation
**Worktree:** `.worktrees/feature/live-2026` (branch `feature/live-2026`)

---

## Motivation

Two research studies completed since the original d20260224 live setup
fundamentally change which model, trading config, and bankroll allocation
we should use for the 2026 ceremony (March 15):

1. **[d20260305_config_selection_sweep](../d20260305_config_selection_sweep/)**
   showed that `avg_ensemble` is the best model (highest cross-year rank
   stability ρ=0.862, lowest EV inflation 1.10×, 100% both-year profitable).
   It also showed that structural params are settled: `fee_type=taker`,
   `kelly_mode=multi_outcome`, `allowed_directions=all`. Edge threshold is
   the only meaningful risk dial (KF barely matters with multi_outcome).

2. **[d20260305_portfolio_kelly](../d20260305_portfolio_kelly/)** showed that
   signal-proportional category allocation captures 2–4× the P&L of uniform.
   `maxedge_100` is the recommended strategy: best avg_rank (2.5), best
   worst_rank (4), ranked top-4 in both years.  It uses
   `max(model_prob - market_prob)` per category to allocate more bankroll
   to categories with the largest buy-side model–market disagreement.

### What changes from the old regime

| Aspect | Old (pre-Mar 7) | New (Mar 7+) |
|--------|-----------------|--------------|
| **Model** | clogit (all 3 configs) | avg_ensemble |
| **Configs** | 3 named (conservative/moderate/aggressive) | 1 config |
| **Edge threshold** | 0.05 / 0.15 / 0.15 | 0.20 |
| **Kelly fraction** | 0.35 / 0.50 / 0.15 | 0.05 |
| **Kelly mode** | independent / independent / multi_outcome | multi_outcome |
| **Fee type** | maker | taker |
| **Directions** | YES-only / all / all | all |
| **Category allocation** | Uniform $1K/category | maxedge_100 (signal-proportional) |
| **Config neighborhoods** | ~336 per config | None (dropped) |

---

## Design Decisions

### D1. Update d20260224 in-place (Option C)

Rather than creating a new one-off, we evolve d20260224 because:
- Same ceremony, same pipeline, same trained models
- 70%+ of code is reusable (913-line generate_report.py, shell scripts, etc.)
- README already uses dated subsections — "Mar 7" is natural
- Avoids cross-one-off import tangles

### D2. Post-hoc allocation scaling (not backtest-time)

The backtest still runs with uniform $1K/category. Allocation weights multiply
P&L in the report generator. This is what the portfolio_kelly study validated
and requires no backtest engine changes. Kelly sizing is approximately linear
in bankroll, so post-hoc scaling is accurate for the "what to allocate" question.

### D3. Run all 6 models, trade only avg_ensemble

All 4 base models + 2 ensembles still run through the backtest for
`model_vs_market.csv` data (needed by `analyze_model_agreement.py` and by
the model comparison report section). Only avg_ensemble is used for trading
decisions and P&L reporting.

### D4. Compute maxedge_100 signal from model_vs_market.csv

The allocation weight for each category is computed in `generate_report.py`:
1. Load `model_vs_market.csv`, filter to `model_type == "avg_ensemble"`
   and the target entry snapshot
2. Compute `max_edge = max(model_prob - market_prob)` per category
3. Feed into `compute_weights(strategy="max_edge", aggressiveness=1.0)`
   imported from `d20260305_portfolio_kelly.shared`
4. Scale per-category P&L by weight; scale deployed capital by weight

### D5. Single config, extensible design

Code assumes 1 config but uses `dict[str, BacktestConfig]` for the config
registry so extending to multiple configs later is trivial. CLI args like
`--config` still work, they just have one valid value.

### D6. Keep model comparison section in reports

The per-category model comparison (all 6 models) stays in reports as a
sanity check and risk signal, even though we only trade on avg_ensemble.

---

## Implementation Plan

### Task 1: Update `recommended_configs.py`

**Goal:** Replace the 3-config (conservative/moderate/aggressive) system with a
single recommended config.

**Changes:**
- Replace `OPTION_MODELS` dict (3 entries mapping to "clogit") with a single
  constant: `RECOMMENDED_MODEL = "avg_ensemble"`
- Replace `get_recommended_configs()` (which built 3 configs) with a function
  that builds 1 config:
  - `avg_ensemble`, edge=0.20, KF=0.05, taker fees, multi_outcome Kelly,
    all directions, spread_penalty from trade history
- Keep `RECOMMENDED_CONFIGS: dict[str, BacktestConfig]` as a dict (for
  extensibility) but with 1 entry: `{"recommended": <config>}`
- Keep backward-compatible exports: `OPTION_MODELS` can become
  `{"recommended": "avg_ensemble"}` so downstream code doesn't break
  structurally, just has 1 key instead of 3
- `DEFAULT_BANKROLL` and `DEFAULT_SPREAD_PENALTY` stay the same

**Downstream impact:** All files that import `OPTION_MODELS` and
`RECOMMENDED_CONFIGS` will work unchanged — they iterate over a dict with
1 entry instead of 3.

### Task 2: Update `__init__.py`

**Goal:** Update `_TODAY` to current date.

**Changes:**
- `_TODAY = date(2026, 3, 7)` (was `date(2026, 3, 2)`)

### Task 3: Simplify `run_buy_hold.py`

**Goal:** Remove neighborhood grid logic, simplify config dispatch.

**Changes:**
- Remove import of `config_neighborhoods.generate_neighborhood_grid`
- Remove `_MODEL_TO_OPTIONS` mapping logic
- Simplify `get_configs_for_model()`:
  - For the recommended model (`avg_ensemble`): return the single
    recommended config
  - For other models: also return the single recommended config (so model
    comparison uses identical config)
  - Remove `centers_only` parameter (no neighborhoods = always centers-only)
- Keep `--centers-only` CLI arg but make it a no-op (for backward compat
  with existing shell scripts)
- Keep `--live` mode unchanged — it already forces centers-only
- Keep all 6 models running (4 individual + 2 ensemble)
- Keep `BANKROLL = 1000.0` per category (allocation is post-hoc)

### Task 4: Add allocation to `generate_report.py`

**Goal:** Add a "Category Allocation" section showing maxedge_100 weights
and scale P&L/capital accordingly.

**Changes:**

1. **Add import** of `compute_weights` from
   `d20260305_portfolio_kelly.shared`

2. **Add allocation computation function:**
   ```python
   def compute_allocation_weights(
       model_vs_market: pd.DataFrame,
       model_type: str,
       entry_snapshot: str,
       position_summary: pd.DataFrame,
       config_label: str,
   ) -> pd.DataFrame:
       """Compute maxedge_100 weights per category.

       Returns DataFrame with columns:
           category, max_edge, is_active, raw_weight, final_weight,
           allocated_bankroll
       """
   ```
   - Filter `model_vs_market` to target model + snapshot
   - Compute `max_edge = max(model_prob - market_prob)` per category
   - Determine `is_active` from position_summary (any positions in category?)
   - Call `compute_weights(cat_data, strategy="max_edge", aggressiveness=1.0)`
   - Compute `allocated_bankroll = base_bankroll * weight`

3. **New report section** `_allocation_section()` inserted between positions
   and scenario P&L sections:
   - Strategy name and provenance (link to portfolio_kelly study)
   - Table: category | max_edge | active? | weight | bankroll
   - Total allocated vs total uniform
   - Brief explanation of the maxedge_100 formula

4. **Update `_mc_section()`:** Scale per-category P&L by allocation weight
   in the Monte Carlo simulation. Each `CategoryScenario.pnls` gets
   multiplied by its category's weight. `total_capital_deployed` also
   scales by weight.

5. **Update `_scenario_pnl_section()`:** Show both uniform and allocated
   P&L columns side by side for transparency.

6. **Update `_write_csv_summary()`:** Add allocation weight and allocated
   capital columns.

7. **Simplify multi-config iteration:**
   - `generate_reports()` still iterates over `RECOMMENDED_CONFIGS.keys()`
     but now that's just `["recommended"]`
   - Report filename: `{timestamp}_recommended.md` (was `_conservative.md`
     etc.)

### Task 5: Update `analyze_model_agreement.py`

**Goal:** Work with single config instead of 3.

**Changes:**
- Update the `--config` CLI arg default to `"recommended"` (was one of
  conservative/moderate/aggressive)
- `choices=list(RECOMMENDED_CONFIGS)` still works (now just
  `["recommended"]`)
- No logic changes needed — it already looks up `OPTION_MODELS[config_name]`
  and `RECOMMENDED_CONFIGS[config_name]`, which now return `"avg_ensemble"`
  and the single config respectively

### Task 6: Update `analyze_orderbook.py`

**Goal:** Work with single config instead of 3.

**Changes:**
- Same pattern as Task 5: update default `--config` choice
- Still imports `OPTION_MODELS` and `RECOMMENDED_CONFIGS` — works with
  1-entry dict

### Task 7: Update `run_live.sh`

**Goal:** Simplify to single-config pipeline.

**Changes:**
- Remove any `--config` flags that iterated over 3 configs
- Should already work since `generate_report` defaults to all configs
  (now just 1)
- Verify the 4-step pipeline still runs cleanly

### Task 8: Update `run.sh`

**Goal:** Remove neighborhood reference from Step 5 description.

**Changes:**
- Update comment: "3 recommended configs + neighborhoods" →
  "recommended config (avg_ensemble + maxedge_100 allocation)"
- Remove `--inferred-lag-hours` if it's just the default

### Task 9: Delete `config_neighborhoods.py`

**Goal:** Remove dead code.

**Changes:**
- Delete the file
- Verify no remaining imports (only `run_buy_hold.py` imported it, which
  Task 3 cleaned up)

### Task 10: Condense README findings

**Goal:** Collapse pre-Mar-7 findings into a `<details>` block and add a
new "Mar 7 — Config regime change" section.

**Changes:**

1. **Wrap all existing findings** (Mar 3 live orderbook, Mar 2 SAG update,
   Feb 28, Feb 27, Feb 24 sections) in a single `<details>` block:
   ```markdown
   <details>
   <summary>Pre-Mar-7 findings (old 3-config / clogit / uniform regime)</summary>
   ... existing content ...
   </details>
   ```

2. **Add new "Mar 7 — Config regime change" dated section** before the
   collapsed block:
   - Summary of what changed and why (link to config_selection_sweep and
     portfolio_kelly)
   - New config table (1 row)
   - New allocation strategy description
   - Note that ASC+WGA is pending (Mar 8) — next update

3. **Update the "Setup" section** at the top:
   - Replace the 3-config table with the single config
   - Update model description (clogit → avg_ensemble)
   - Add "Category Allocation" subsection describing maxedge_100
   - Update "Available Snapshots" table

4. **Update "How to Run" section:**
   - Remove `--config conservative/moderate/aggressive` examples
   - Simplify report generation commands
   - Update the "After a New Precursor Event" checklist

5. **Update "Code Modules" table:**
   - Remove `config_neighborhoods.py`
   - Note allocation import from portfolio_kelly

### Task 11: Rename existing PLAN files

**Goal:** Follow `PLAN_<mmdd>_<name>.md` convention.

**Changes:**
- `PLAN.md` → `PLAN_0224_buy_hold_live_setup.md` ✅ (already done)
- `PLAN_refactor_buy_hold_pipeline.md` → `PLAN_0302_refactor_buy_hold_pipeline.md` ✅ (already done)

---

## Execution Order

Tasks can be grouped into phases:

**Phase 1 — Config core (Tasks 1, 2, 9, 11):**
Update the config definitions, delete dead code, rename PLANs.
No functional behavior changes yet.

**Phase 2 — Backtest runner (Task 3):**
Simplify `run_buy_hold.py` to use single config, remove neighborhoods.
After this, `run_buy_hold.py` works with new configs but produces
uniform-allocation results (same as before, just with avg_ensemble config).

**Phase 3 — Report + allocation (Task 4):**
The main new functionality. Adds maxedge_100 allocation to
`generate_report.py`. After this, reports show allocated bankrolls and
scaled P&L.

**Phase 4 — Auxiliary scripts (Tasks 5, 6, 7, 8):**
Update supporting scripts to single-config interface.

**Phase 5 — README (Task 10):**
Condense old findings, add Mar 7 section, update top-matter.

**Verification:** After each phase, run `make dev` to ensure no regressions.
After Phase 3, generate a report to verify allocation weights appear correctly.

---

## Testing

- `make dev` (format + lint + typecheck + test) after each phase
- Manual: `uv run python -m ...generate_report --bankroll 1000` and inspect
  the markdown output for the allocation section
- Verify model agreement analysis still runs:
  `uv run python -m ...analyze_model_agreement`
- Verify live pipeline:
  `bash .../run_live.sh` (fetches current orderbook + generates report)

---

## Dependencies

- `d20260305_portfolio_kelly.shared.compute_weights` — imported at runtime
  for allocation weight calculation
- `d20260225_buy_hold_backtest.portfolio_simulation` — already imported
  for Monte Carlo simulation (no change)
- `d20260225_buy_hold_backtest.year_config` — already imported for
  `YEAR_CONFIGS[2026]` (no change)

---

## Files Changed Summary

| File | Action | Phase |
|------|--------|-------|
| `recommended_configs.py` | Rewrite — single config | 1 |
| `__init__.py` | Update `_TODAY` | 1 |
| `config_neighborhoods.py` | Delete | 1 |
| `PLAN.md` | Rename → `PLAN_0224_buy_hold_live_setup.md` | 1 |
| `PLAN_refactor_buy_hold_pipeline.md` | Rename → `PLAN_0302_refactor_buy_hold_pipeline.md` | 1 |
| `run_buy_hold.py` | Simplify — remove neighborhoods, single config | 2 |
| `generate_report.py` | Add allocation section + post-hoc P&L scaling | 3 |
| `analyze_model_agreement.py` | Update default config name | 4 |
| `analyze_orderbook.py` | Update default config name | 4 |
| `run_live.sh` | Simplify to single config | 4 |
| `run.sh` | Update Step 5 description | 4 |
| `README.md` | Condense old findings, add Mar 7 section, update setup | 5 |
