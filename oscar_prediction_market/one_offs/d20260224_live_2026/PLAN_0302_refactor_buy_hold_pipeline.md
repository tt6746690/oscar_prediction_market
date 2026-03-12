# Refactoring Plan: Buy-Hold Pipeline Deduplication

**Created:** 2026-03-02
**Status:** Backlog — parked until after 2026 ceremony (March 15)

## Problem

Three one-off directories have evolved into a layered import chain acting as
a de facto shared library:

```
d20260220_backtest_strategies   ← foundational constants, data_prep, BacktestGridConfig
        ▲           ▲
        │           │
d20260225_buy_hold_backtest     ← year-parameterized wrappers, buy-hold engine
        ▲
        │
d20260224_live_2026             ← live trading reports, orderbook, model agreement
        ▲
        │
d20260228_pga_scenario_analysis ← PGA what-if analysis
```

~150 lines of accidental duplication in the core buy-hold loop, plus Monte Carlo
simulation is reimplemented in two places.

## Duplication Inventory

| Duplicated piece | ~Lines | Between |
|---|---|---|
| `_remap_person_names()` | 14 | d20260224 ↔ d20260225 (identical) |
| `_load_individual_model()` / `_load_and_remap_individual()` | 18 | d20260224 ↔ d20260225 (same logic, different name) |
| `_load_ensemble_model()` / `_load_and_remap_ensemble()` | 16 | d20260224 ↔ d20260225 (same logic, different name) |
| Config construction (KellyConfig → BacktestConfig) | 25 | d20260224 ↔ d20260225 (char-identical) |
| Capital deployed computation | 6 | d20260224 ↔ d20260225 (identical) |
| Model-vs-market divergence loop | 20 | d20260224 ↔ d20260225 (one extra `is_winner` field) |
| Entry moment loop pattern | 12 | d20260224 ↔ d20260225 |
| MC portfolio simulation core | 15 | generate_report ↔ scenario_scoring |
| `run_category_buy_hold()` overall structure | ~150 shared / 300 total | d20260224 ↔ d20260225 (~60% identical) |

## The True Historical vs Live Fork

The *only* fundamental difference between the two pipelines is **settlement handling**:

- **Historical** (`d20260225/run_backtests.py`): calls `result.settle(winner_model_name)` →
  actual P&L, worst/best/EV metrics
- **Live** (`d20260224/run_buy_hold.py`): iterates `result.settlements` → scenario matrix
  (what if nominee X wins?)

Everything else — model loading, name matching, spread estimation, entry moment
construction, config construction, engine execution — is identical.

## Import Chain Deep Dive

### What d20260224 imports from d20260225

From `data_loading.py`:
- `build_nominee_match`, `build_title_to_person_map`, `fetch_category_market_data`,
  `load_all_snapshot_predictions`, `load_ensemble_predictions`

From `run_backtests.py`:
- `ALL_ENSEMBLES`, `CLOGIT_CAL_SGBT_ENSEMBLE_LABEL`, `build_entry_moment`

From `year_config.py`:
- `YEAR_CONFIGS`, `YearConfig`

### What d20260225 imports from d20260220

From `__init__.py`:
- `BACKTEST_MODEL_TYPES`, `CATEGORY_SLUGS`, `ENSEMBLE_SHORT_NAME`,
  `MODEL_SHORT_NAMES`, `MODELED_CATEGORIES`, `get_post_nomination_snapshot_dates`

From `data_prep.py`:
- `BANKROLL`, `DEFAULT_SPREAD`, `MIN_TRADES_FOR_SPREAD`, `CategoryMarketData`,
  `build_market_prices`, `make_spread_penalties_by_model_name`,
  `remap_predictions_to_person_names`, `renormalize_predictions`

From `evaluation.py`:
- `evaluate_model_accuracy`

From `run_backtests.py`:
- `BacktestGridConfig`, `generate_trading_configs`

### Root cause

`d20260220/data_prep.py` hardcodes `CEREMONY_YEAR=2025` and `CALENDAR`. The
year-parameterized wrappers in `d20260225/data_loading.py` exist only to add a
`YearConfig` parameter around the same logic.

## Proposed Solutions

### Option A: Minimal — Deduplicate within one-offs (Small diff, low risk)

Move shared helpers up without restructuring packages:

1. Move `_remap_person_names`, `_load_and_remap_individual`, `_load_and_remap_ensemble`
   into `d20260225/data_loading.py`. Delete copies from d20260224.
2. Extract `build_backtest_config(cfg, mean_spread) → BacktestConfig` into
   d20260225 shared location. Both pipelines import it.
3. Extract `build_mvm_rows(...)` with optional `winner_model_name`.
4. Extract MC sampling core `sample_portfolio_pnl()` → shared module.

**Pros**: Small diff, directly eliminates ~100 lines.
**Cons**: The tangled import chain remains.

### Option B: Moderate — Create `trading/buy_hold.py` (Recommended)

Extract the core buy-hold loop into the proper `trading/` package:

1. Create `trading/buy_hold.py` with:
   - `build_backtest_config(cfg, mean_spread) → BacktestConfig`
   - `run_single_entry(moment, cfg, mean_spread, spread_penalties) → BacktestResult`
   - `build_mvm_rows(..., winner_model_name=None)`
   - `compute_capital_deployed(result) → float`
   - `sample_portfolio_pnl(cat_scenarios, n_samples, rng) → np.ndarray`

2. Move `BacktestGridConfig` + `generate_trading_configs()` from d20260220 → `trading/schema.py`.

3. Move model-loading + name-remapping → `trading/oscar_predictions.py` (parameterized by YearConfig).

4. Move `MODELED_CATEGORIES`, `MODEL_SHORT_NAMES`, `BACKTEST_MODEL_TYPES` from d20260220 → `trading/` or `modeling/`.

5. Both `run_buy_hold.py` and `run_backtests.py` become thin wrappers calling shared core.

**Pros**: Clean package boundaries, one-offs import from `trading/` not each other.
**Cons**: Larger refactor, need to update all importers + tests.

### Option C: Aggressive — Unified `run_buy_hold()` with `winner=None|str`

Build on B, plus: single `trading/buy_hold.py::run_category_buy_hold()` that
accepts `winner_model_name: str | None = None`. Both d20260224 and d20260225
call the same function.

**Pros**: Maximum DRY, single source of truth.
**Cons**: Most complex, risk of over-abstraction.

## Live Pricing Support

Both historical and live pipelines need a way to get market prices. Currently:

- **Historical**: Uses cached candlestick data (diskcache, no TTL). Fetches
  daily/hourly candles for the entry timestamp.
- **Live**: Same cached candle data — can be stale for the current partial day.

### Design for live pricing

The entry moment construction (`build_entry_moment`) takes `market_prices_by_date`
which maps `date → {nominee: price}`. For live pricing:

1. Fetch current orderbook mid-prices via `KalshiPublicClient.get_orderbook()`
2. Convert to the same `{nominee: price}` dict format
3. Feed into the same `build_entry_moment` → `BacktestEngine` pipeline

This could be:
- A `price_source: Literal["candle", "orderbook_mid"]` parameter on `build_entry_moment`
- Or a separate `build_live_entry_moment()` that fetches orderbook and constructs the moment

The rest of the pipeline (Kelly sizing, MC simulation, report generation) stays identical.

## Parking

Recommend: Option B after the March 15 ceremony, when we can refactor without
time pressure. For now, live pricing can be added as a thin overlay in d20260224.

## Open Questions

1. Should `YearConfig` move to `trading/` or stay in one-offs?
2. Should `RECOMMENDED_CONFIGS` move to `trading/` (d20260228 imports it)?
3. Should `BacktestGridConfig` sit in `trading/schema.py` alongside `TradingConfig`?
4. MC probability weighting: model-only (live) vs blend (historical) — parameterize or keep separate?
