# Backtest One-Off Cleanup & Refactoring Plan

**Date:** 2026-02-23
**Branch:** `feature/backtest-strategies`
**Goal:** Clean up the `d20260220_backtest_strategies` one-off — fix bugs, remove
dead code, consolidate duplicated constants, promote reusable utilities,
and prepare for multi-year backtesting.

---

## Current State

| File | Lines | Role |
|------|-------|------|
| `run_backtests.py` | 659 | Orchestration + config grid + CLI |
| `data_prep.py` | 679 | Data loading, name matching, TradingMoment construction |
| `evaluation.py` | 117 | Model accuracy + market-favorite baseline |
| `__init__.py` | 166 | Constants + snapshot/trading date helpers |
| `analyze.py` | 2,445 | ~20 plot/analysis functions (single script) |
| `compare_delay_modes.py` | 121 | Post-hoc delay comparison |
| `run_all_delay_modes.py` | 60 | 3-mode subprocess orchestrator |
| `discover_tickers.py` | 171 | Kalshi API ticker discovery (DUPLICATE) |

Upstream `trading/` module (4,776 lines) is well-structured and generic.

---

## Phase 1: Bug Fixes & Dead Code Removal

Low-risk mechanical cleanup. No design decisions needed.

### 1a. Fix bugs

| Bug | File | Fix |
|-----|------|-----|
| Broken nominee-map save — `str(model_names)` matching always yields `"unknown"` | `data_prep.py` L669–676 | Delete the entire save block. Caller (`run_backtests.py` L295) already saves correctly. |
| Dead parameter `ticker_to_nominee` in `build_market_prices()` | `data_prep.py` L233 | Remove parameter. Update the one call site. |
| Misleading docstring "weekdays only" vs code includes all days | `__init__.py` `get_trading_dates()` | Fix docstring to say "all days (Kalshi trades 7 days/week for events)" |
| Silent `except Exception: pass` | `compare_delay_modes.py` L118 | Replace with explicit error handling or `logging.warning()` |
| Baseline ignores fees (see Phase 3 for full fix) | `evaluation.py` `run_market_favorite_baseline()` | Add maker fee deduction (1.75%) to match backtest assumptions |

### 1b. Remove dead code

| Dead item | File | Action |
|-----------|------|--------|
| `MODEL_FEATURE_FAMILY` — never imported | `__init__.py` L72–78 | Delete |
| `WINNERS_2025` import — never used | `run_backtests.py` L63 | Remove from import |
| `DATASETS_DIR`, `EXP_DIR`, `MARKET_DATA_DIR` imports — only used in `data_prep.py` | `run_backtests.py` L57–63 | Remove from import |
| `NOMINEE_MAPS_DIR` import — used only for duplicate save (removed in 1a) | `run_backtests.py` | Remove from import |

### 1c. Delete duplicated file

| File | Canonical location | Action |
|------|-------------------|--------|
| `discover_tickers.py` (171 lines) | `trading/market_data/discover_tickers.py` | Delete from one-off. Nearly identical copy — canonical version lives in `trading/market_data/` and writes to `ticker_inventory.json` in that directory. |

---

## Phase 2: Constants Consolidation

### Problem

Constants are scattered across `__init__.py` and `data_prep.py`, with some
duplicating definitions elsewhere in the codebase:

| Constant | One-off location | Canonical location | Identical? |
|----------|-----------------|-------------------|------------|
| `CATEGORY_SLUGS` | `__init__.py` | `modeling.feature_engineering.groups` | Yes (identical dict, different ordering) |
| `SLUG_TO_CATEGORY` | `__init__.py` | `modeling.feature_engineering.groups.CATEGORY_BY_SLUG` | Yes (same inverse) |
| `PERSON_CATEGORIES` | `data_prep.py` (5 categories) | `data.schema` (7 categories) | **No** — schema has 7 (data-level), one-off has 5 (Kalshi ticker naming). Semantically different. |
| `MODELED_CATEGORIES` | `__init__.py` | Nowhere else | Unique to one-off |
| `BACKTEST_MODEL_TYPES` | `__init__.py` | Nowhere else | Unique to one-off |
| `MODEL_SHORT_NAMES` | `__init__.py` | Nowhere else | Unique to one-off |
| `WINNERS_2025` | `data_prep.py` | Nowhere else | Unique to one-off |

### Plan

**2a. Import `CATEGORY_SLUGS` / `CATEGORY_BY_SLUG` from canonical location**

Replace one-off definitions with imports from `modeling.feature_engineering`:
```python
from oscar_prediction_market.modeling.feature_engineering import (
    CATEGORY_SLUGS,
    CATEGORY_BY_SLUG as SLUG_TO_CATEGORY,
)
```

**2b. Rename one-off's `PERSON_CATEGORIES` to `KALSHI_PERSON_NAME_CATEGORIES`**

This avoids confusion with `data.schema.PERSON_CATEGORIES` (which is a
superset). The one-off's set specifically means "categories where **Kalshi
uses person names** as ticker labels instead of film titles."

**2c. Create `constants.py` for year-specific backtest constants**

Move from `data_prep.py` → new `constants.py`:
- `CEREMONY_YEAR`
- `WINNERS_2025` → rename to `WINNERS` (parameterized by year later)
- `BANKROLL`
- `DEFAULT_SPREAD_CENTS`, `MIN_TRADES_FOR_SPREAD`
- `KALSHI_PERSON_NAME_CATEGORIES`
- Path constants (`EXP_DIR`, `DATASETS_DIR`, `RESULTS_DIR`, etc.)

Keep in `__init__.py`:
- `MODELED_CATEGORIES`, `BACKTEST_MODEL_TYPES`, `MODEL_SHORT_NAMES` (experiment identity)
- `get_post_nomination_snapshot_dates()`, `get_trading_dates()` (derived helpers)

**2d. Extensibility for multi-year: `WINNERS` dict keyed by year**

```python
# constants.py
WINNERS: dict[int, dict[OscarCategory, str]] = {
    2025: {
        OscarCategory.BEST_PICTURE: "Anora",
        ...
    },
    # 2024: { ... },  # Add when backesting 2024
}
```

This is forward-compatible. The runner accepts `--ceremony-year` and looks
up `WINNERS[year]`.

---

## Phase 3: Fix `evaluation.py` Market-Favorite Baseline

### Problem

`run_market_favorite_baseline()` calculates raw P&L without fees, making it
incomparable with the main backtest.

### Fix

Add fee deduction using maker fees (1.75%) as the assumption. The baseline
is a passive strategy (buy once, hold to settlement), so maker fees are
reasonable:

```python
from oscar_prediction_market.trading.kalshi_client import estimate_fee_cents

# After computing raw P&L for a buy at price_cents:
fee_cents = estimate_fee_cents(price_cents, FeeType.MAKER)
entry_cost = (price_cents + fee_cents) / 100  # per contract
```

This matches the `FeeType.MAKER` assumption used in the main backtest's
default config.

Also fix the fragile winner matching that uses `str.contains()` — switch
to exact `WINNERS` lookup via `get_winner_model_name()` or direct dict
access.

---

## Phase 4: Pydantic `BacktestGridConfig` Model

### Problem

`generate_trading_configs()` returns `list[dict]` — typos in config keys
are silent.

### Plan

Create a Pydantic model for the trading parameter space:

```python
class BacktestGridConfig(BaseModel):
    """One point in the trading parameter grid."""
    model_config = {"extra": "forbid"}

    kelly_fraction: float = Field(ge=0, le=1)
    edge_threshold: float = Field(ge=0)
    sell_edge_threshold: float
    fee_type: FeeType
    kelly_mode: KellyMode
    trading_side: TradingSide
    bankroll_mode: str = Field(pattern=r"^(fixed|dynamic)$")
    max_position: float = Field(ge=0)
    max_exposure: float = Field(ge=0)
    spread_penalty: float = Field(ge=0)
```

`generate_trading_configs()` returns `list[BacktestGridConfig]`.
`config_to_label()` becomes a method on the model.

Location: new file `config.py` in the one-off, or at the top of
`run_backtests.py` (it's only ~80 lines with the model).

---

## Phase 5: Promote Reusable Utilities to `trading/`

### Candidates assessed

| Utility | Location | Promote? | Rationale |
|---------|----------|----------|-----------|
| `_get_daily_prices_with_fallback()` | `data_prep.py` | **No** | Overlaps `price_utils.get_market_prices_on_date()`, but operates on `dict[str, dict[str, float]]` vs DataFrame. Thin wrapper (10 lines), different input type. Not worth forcing a shared interface. |
| `renormalize_predictions()` | `data_prep.py` | **No** | 6 lines, filter + sum=1. Too thin to justify as a library function. |
| `_make_daily_close_timestamp()` | `data_prep.py` | **Maybe** | Creates 4 PM ET → UTC. Could live in `price_utils` if other one-offs need it. Leave for now, revisit after multi-year. |
| `build_market_prices()` | `data_prep.py` | **No** | Oscar-specific name remapping (Kalshi→model names). |
| `make_spread_penalties_by_model_name()` | `data_prep.py` | **No** | Name-mapping glue layer, specific to the one-off's data flow. |
| `build_nominee_match()` | `data_prep.py` | **No** | Orchestrates nominee matching with Oscar-specific name maps. |
| `build_trading_moments()` | `data_prep.py` | **No** | Already correctly placed. TradingMoment *type* is in `trading/backtest.py` (generic). Construction is caller's responsibility per engine design. |

### Recommendation

**No promotions in this phase.** The data_prep.py functions are
domain-specific glue; the trading/ module is already correctly generic.
Revisit if a second domain (NFL, crypto) needs backtesting — at that
point, any shared patterns would emerge naturally.

One exception worth considering for later: if multi-year Oscar backtesting
reveals that `_make_daily_close_timestamp()` and
`_get_daily_prices_with_fallback()` are copy-pasted for
each year, promote them then.

---

## Phase 6: Consolidate `analyze.py` with `analysis/` Module

### Analysis of overlap

| Category | Count | Examples |
|----------|-------|---------|
| **Redundant** (drop-in replaceable) | 0 | — |
| **Partial overlap** (shared building blocks) | 7 | Style constants, divergence heatmaps, prob trajectories, metrics-over-time, entropy |
| **Unique** (backtest-specific) | 21 | P&L decomposition, config selection, risk profiling, timing, parameter sensitivity |

### Plan

**Keep `analyze.py` as a single script** (per user preference).

**6a. Import style constants from `analysis/style.py`**

Replace local `MODEL_DISPLAY`, `CATEGORY_DISPLAY`, `SHORT_MODEL_COLORS`
definitions (lines 54–81) with imports from `analysis.style`:

```python
from oscar_prediction_market.analysis.style import (
    MODEL_COLORS,
    get_model_color,
    get_model_display,
    CATEGORY_DISPLAY,
    apply_style,
)
```

If the display names need slight differences (e.g., `"cal_sgbt"` short name
for columns), extend via a local override dict rather than re-defining from
scratch.

**6b. Reuse `metrics.compute_prob_concentration()` for entropy math**

`plot_probability_concentration()` in analyze.py computes entropy inline
with identical math ($-\sum p \log_2 p$). Import and use
`compute_prob_concentration()` from `analysis.metrics`.

**6c. Factor shared heatmap rendering to `analysis/plot_trading.py`**

Both `analyze.py::plot_divergence_heatmaps()` and
`analysis/plot_trading.py::plot_divergence_heatmaps()` render
nominee × date heatmaps with annotated cells. Extract a shared
`_render_divergence_heatmap(pivot_df, ax, cmap, ...)` helper into
`analysis/plot_trading.py` that both can call. analyze.py adds
the multi-category loop on top.

**6d. No other promotions**

The remaining ~75% of analyze.py is genuinely unique trading/backtest
analysis. Leave as-is in the script.

---

## Phase 7: Multi-Year Extensibility

### Current hardcoding

| Hardcoded item | Location | How to parameterize |
|----------------|----------|-------------------|
| `CEREMONY_YEAR = 2025` | `data_prep.py` | CLI arg `--ceremony-year` in `run_backtests.py` |
| `WINNERS_2025` | `data_prep.py` → `constants.py` | `WINNERS[year]` dict (see Phase 2d) |
| `CALENDAR` (imports `CALENDARS["2025"]`) | `data_prep.py` | `CALENDARS[str(year)]` |
| Snapshot model paths | `data_prep.py` `load_predictions()` | Already parameterized by `DATASETS_DIR` pattern — depends on how model training stores multi-year outputs |
| `"2025"` in storage paths | `data_prep.py`, shell scripts | `EXP_DIR / str(year) / ...` |

### Plan

Add `--ceremony-year` CLI arg to `run_backtests.py`. Pass `year` through
to `data_prep` functions. The main structural change:

1. `constants.py`: `WINNERS` is a `dict[int, dict[OscarCategory, str]]`
2. `data_prep.py`: All functions that use `CEREMONY_YEAR` accept it as a
   parameter instead of reading the module-level constant
3. `run_backtests.py` `main()`: Reads `--ceremony-year` and passes it
   through
4. `EXP_DIR` becomes `EXP_DIR / str(year)`

**Prerequisite**: Model snapshots for the target year must exist. The
shell scripts (`train_models.sh`, `build_datasets.sh`) handle model
training and are already parameterized by year.

---

## Execution Order

| Phase | Risk | Effort | Dependencies |
|-------|------|--------|-------------|
| **1. Bug fixes + dead code** | Low | Small | None |
| **2. Constants consolidation** | Low | Small | Phase 1 |
| **3. Fix baseline fees** | Low | Small | Phase 2 (for WINNERS lookup) |
| **4. BacktestGridConfig Pydantic** | Low | Medium | None (independent) |
| **5. Promote to trading/** | — | — | **Skip for now** (nothing to promote) |
| **6. analyze.py ↔ analysis/ consolidation** | Medium | Medium | Phase 2 (style imports) |
| **7. Multi-year extensibility** | Medium | Medium | Phases 1–3 |

Phases 1–4 are safe to execute in sequence. Phase 6 can be done in
parallel. Phase 7 should come last.

---

## Validation

After each phase:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -m oscar_prediction_market.one_offs.d20260220_backtest_strategies.run_backtests \
    --fast --categories best_picture 2>&1 | tail -20
# Should complete without errors and produce reasonable P&L numbers
```

After all phases:
```bash
make all  # format + lint + typecheck + test
```
