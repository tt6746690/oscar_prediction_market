# PLAN: 2026 Live Oscar Predictions — Buy-and-Hold

**One-off:** `d20260224_live_2026`
**Storage:** `storage/d20260224_live_2026/`
**Worktree:** `.worktrees/feature/live-2026` (branch `feature/live-2026`)
**Created:** 2026-02-24 | **Revised:** 2026-02-28

---

## Goal

Apply buy-and-hold strategy to the **2026 Oscar ceremony** (Mar 15) using
real-time precursor data.  Uses the 3 configs identified as robust across
both 2024 and 2025 ceremonies in the cross-year buy-hold backtest
([d20260225_buy_hold_backtest](../d20260225_buy_hold_backtest/)).

Key outputs:
1. **Scenario P&L** — for each (category, model, entry point, config), what's
   the P&L if each nominee wins?
2. **Position summary** — what positions does each config recommend right now?
3. **Config sensitivity** — how stable are results in the neighborhood of
   each recommended config?
4. **Model vs market divergence** — where does the model disagree with the
   market, and how has that evolved across snapshots?

---

## Methodology: Buy-and-Hold

The buy-and-hold approach (from d20260225) uses a single `MarketSnapshot` per
entry point.  With one moment, the `BacktestEngine` buys once and holds to
settlement.  No rebalancing, no selling.

Since the 2026 winner is unknown, we use `result.settlements` (a
`dict[str, SettlementResult]`) to compute hypothetical P&L for every possible
winner.  This replaces the single `result.settle(known_winner)` call used in
historical backtests.

---

## Recommended Configs (from Cross-Year Analysis)

The [cross-year buy-hold backtest](../d20260225_buy_hold_backtest/) tested 615
configs across both 2024 and 2025 ceremonies.  Three configs emerged as robust:

| Option | Model | Kelly Mode | Edge | Side | kf | Fee | Combined P&L | Both-Year Rate | ρ |
|:------:|-------|:----------:|:----:|:----:|:--:|:---:|-------------:|:--------------:|:-:|
| **A** | avg_ensemble | multi_outcome | 0.15 | all | 0.15 | maker | +$14,005 | 88% | +0.64 |
| **B** | clogit | independent | 0.10 | yes | 0.50 | maker | +$8,134 | 100% | +0.92 |
| **C** | clogit_cal_sgbt_ensemble | multi_outcome | 0.12 | yes | 0.50 | maker | +$9,266 | 86% | +0.62 |

- **Option A (Aggressive):** Highest combined P&L, trades both YES and NO.
- **Option B (Conservative):** 100% both-year profitable rate, highest ρ.
- **Option C (Balanced):** Best ensemble of the two strongest individual models.

Each option is tested only with its assigned model.

### Config Neighborhood (Sensitivity)

Per recommended config, ~336 neighborhood configs varying:

| Parameter | Grid |
|-----------|------|
| Edge threshold | center ± {0.01, 0.02, 0.03} → 7 values |
| Kelly fraction | center ± {0.05, 0.10, 0.15} → 6 values |
| Kelly mode | toggle (independent ↔ multi_outcome) |
| Fee type | toggle (maker ↔ taker) |
| Allowed directions | toggle (yes ↔ all) |

Total: ~336 × 3 options = **1,008 configs**.  Unassigned models (lr, gbt,
cal_sgbt) run only the 3 center configs.

---

## 2026 Calendar

Oscar ceremony: **March 15, 2026**.

| Snapshot | Event | Status (Feb 27) |
|:--------:|-------|:----------------:|
| Jan 22 | Nominations | ✅ |
| Feb 7 | DGA | ✅ |
| Feb 21 | Annie | ✅ |
| Feb 22 | BAFTA | ✅ |
| Feb 28 | PGA | Tonight |
| Mar 1 | SAG | Pending |
| Mar 8 | ASC + WGA | Pending |

4 snapshots available now.  PGA + SAG cluster this weekend (Feb 28 – Mar 1).

---

## Pipeline

### Steps 0–3: Data Prep + Model Training (already done)

Steps 0–3 were completed in the initial run (Feb 24) and are idempotent:

0. **Refresh precursors** — re-scraped Wikipedia, manually patched Annie winner
1. **Copy configs** — feature/param_grid/cv_split from d20260220
2. **Build datasets** — 36 = 9 categories × 4 snapshots
3. **Train models** — 144 = 9 categories × 4 model types × 4 snapshots

Existing artifacts: `storage/d20260224_live_2026/{shared,configs,datasets,models}/`

### Step 5: Buy-and-Hold Analysis (new)

The core analysis.  Replaces the old `run_predictions.py`.

```bash
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.run_buy_hold
```

Runs all 6 models (4 individual + 2 ensembles) × 9 categories × 4 entry points,
each with the model's assigned config set.  For models assigned to a recommended
option (avg_ensemble, clogit, clogit_cal_sgbt_ensemble), runs the full ~336
neighborhood.  Other models run only the 3 center configs.

Outputs:
- `results/scenario_pnl.csv` — P&L per (category, model, entry, config, assumed_winner)
- `results/scenario_pnl_agg.csv` — min/mean/max P&L per (category, model, entry, config)
- `results/model_vs_market.csv` — model probability vs market price per entry
- `results/position_summary.csv` — positions held per (category, model, entry, config)

For fast iteration (just center configs):
```bash
uv run python -m ...run_buy_hold --centers-only
```

---

## Related One-offs

### PGA Scenario Analysis (`d20260228_pga_scenario_analysis`)

PGA (tonight, Feb 28) affects **Best Picture** and **Animated Feature** through
`pga_bp_winner`, `pga_animated_winner`, `has_pga_dga_combo`, and
`precursor_wins_count`.  A separate one-off analyzes how different PGA outcomes
shift model predictions and trading decisions — requires retraining since the
feature set changes.

See [d20260228_pga_scenario_analysis/PLAN.md](../d20260228_pga_scenario_analysis/PLAN.md).

---

## Incremental Update Plan

| Date | Event | Action |
|------|-------|--------|
| **Feb 24** | Initial run | Steps 0–3: data + models (done) |
| **Feb 27** | Revamp to buy-hold | Step 5: run_buy_hold.py (done) |
| **Feb 28** | Centers-only analysis | Run analysis, populate README (done) |
| **Feb 28** | PGA winner | Add snapshot 5 + PGA scenario analysis |
| **Mar 1** | SAG winner | Add snapshot 6, re-train, re-analyze |
| **Mar 8** | ASC + WGA | Add snapshot 7, re-train, re-analyze |
| **Mar 15** | Oscar ceremony | Settlement: compute actual P&L |

Each update:
1. Update `__init__.py` `_TODAY` date
2. Re-fetch precursors (picks up new winners)
3. Build dataset for new snapshot (idempotent)
4. Train models for new snapshot (idempotent)
5. Re-run `run_buy_hold.py` (picks up new entry points automatically)
6. Add dated subsection to README.md

---

## Storage Layout

```
storage/d20260224_live_2026/
├── shared/                         # Refreshed precursors + metadata
├── configs/                        # Copied from d20260220
├── datasets/{cat}/{date}/          # 36 datasets (9 cats × 4 snaps)
├── models/{cat}/{model}/{date}/    # 144 models (9 × 4 × 4)
├── market_data/                    # Kalshi candles + trades
├── nominee_maps/                   # Model ↔ Kalshi name mappings
├── results/
│   ├── scenario_pnl.csv           # Scenario P&L for all possible winners
│   ├── scenario_pnl_agg.csv       # Min/mean/max P&L per config
│   ├── model_vs_market.csv        # Probability divergences
│   └── position_summary.csv       # Held positions
└── run.log
```

---

## Code Modules

| Module | Purpose |
|--------|---------|
| `__init__.py` | 2026 constants, snapshot dates, storage paths |
| `generate_configs.py` | 3 recommended configs + neighborhood grids |
| `run_buy_hold.py` | Core buy-hold backtest runner (no known winner) |
| `refresh_data.py` | Refresh precursor + metadata caches |
| `build_datasets.sh` | Build as-of-date datasets |
| `train_models.sh` | Train all models |
| `setup_configs.sh` | Copy configs from d20260220 |
| `run.sh` | Master pipeline script |

Imports from:
- `d20260225_buy_hold_backtest/` — `build_entry_moment`, data loading, year config
- `d20260220_backtest_strategies/` — `BacktestGridConfig`, data prep utilities
