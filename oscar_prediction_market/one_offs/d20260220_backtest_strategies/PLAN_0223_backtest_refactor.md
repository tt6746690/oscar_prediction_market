# Plan: Backtest Refactor — Datetime Engine + Intraday Prices + Code Cleanup

**Date:** 2026-02-23 (v2 — refined with research findings + feedback)
**Branch:** `feature/backtest-strategies`

---

## Vision

Refactor the backtester to a **datetime-based engine** (`TradingMoment`),
use **hourly candle prices** at execution timestamps, refactor the monolithic
`run_backtests.py`, and re-generate all results + README using the honest
inferred-lag mode as the canonical baseline.

---

## Background: Current Architecture

```
            AwardsCalendar   ───→  snapshot dates (date)
                                        │
  run_backtests.py:                     │
    get_active_snapshot_for_date()  ────┘ (date → date comparison)
        │                                    │
        ▼                                    ▼
    predictions_by_date (str→dict)      prices_by_date (str→dict)
        │                                    │
        └──────── BacktestEngine.run() ──────┘
                          │
              iterates by date_str keys
              prices = daily close price
```

**Key limitation:** Even with "inferred+6h" mode, the effective date is
computed but execution still uses the **daily close** price for that date. If
DGA ends 10:30 PM PST (06:30 UTC Feb 9) and we add 6h → 12:30 UTC Feb 9 →
effective date = Feb 9 → we use Feb 9 close price. But the real opportunity is
to trade at the *intraday* price at 12:30 UTC, which may be very different from
the daily close (the market reprices throughout the day).

---

## Key Design Decisions

### 1. Unified datetime-based delay model

Instead of two modes (`signal_delay_days: int` vs `effective_dates: dict`),
use **one model**: every snapshot gets an `available_at: datetime` (UTC). This
subsumes both modes:

| Use case | How `available_at` is computed |
|----------|-------------------------------|
| Fixed day delay | `datetime(snap_date + N_days, market_open_hour, tzinfo=ET)` |
| Inferred lag | `max(event_end_times_on_date) + timedelta(hours=lag_hours)` |
| Custom | Caller supplies `available_at` directly |

**Pydantic model:**

```python
class FixedDayDelay(BaseModel):
    """Signal available N calendar days after snapshot date, at market open."""
    mode: Literal["fixed_day"] = "fixed_day"
    days: int = Field(..., ge=0)

class InferredLag(BaseModel):
    """Signal available lag_hours after latest event UTC end time."""
    mode: Literal["inferred"] = "inferred"
    lag_hours: float = Field(..., ge=0)

SignalDelayConfig = Annotated[
    FixedDayDelay | InferredLag,
    Discriminator("mode"),
]

def compute_snapshot_availability(
    calendar: AwardsCalendar,
    snapshot_dates: list[date],
    delay_config: SignalDelayConfig,
    strict: bool = True,
) -> dict[date, datetime]:
    """Compute available_at (UTC datetime) for each snapshot."""
```

### 2. Intraday price lookup — hourly candles only

**Data sources available:**

| Source | Granularity | Status |
|--------|-------------|--------|
| Daily candles (`get_batch_candlesticks`, 1440min) | Daily close | Complete |
| Hourly candles (`get_batch_candlesticks`, 60min) | Hourly OHLC | **Supported by API, not yet fetched** |
| Trade history (`get_trades`) | Per-trade | Complete (pagination already fetches all pages) |

**Research findings:**
- `KalshiPublicClient.get_batch_candlesticks()` already accepts
  `period_interval` of 1, 60, or 1440. Hourly candles are fully supported.
- `OscarMarket.fetch_intraday_prices()` already wraps this for any interval.
- **Trade pagination is NOT broken** — `get_trades()` already fetches all
  pages via cursor loop. The page-3 log message is informational only.
  No pagination fix needed.
- No hourly candle data exists in storage yet — only daily candles in
  `storage/.../market_data/candles/`.

**Simplified approach (per feedback):**

Keep `get_price_at_timestamp()` simple — **hourly candles only**. No fallback
chain. If hourly data isn't available, raise an error. This is fine because:
- Hourly granularity is sufficient for 6h lag (several-hour approximation OK)
- We'll fetch hourly candles for all tickers before running backtests
- Avoids complexity of multi-source fallback logic

```python
def get_price_at_timestamp(
    timestamp: datetime,              # UTC
    outcome: str,
    hourly_candles: pd.DataFrame,     # pre-loaded, required
) -> float:
    """Return hourly close price (cents) for outcome at the given UTC timestamp.

    Looks up the hourly candle containing ``timestamp`` (i.e., the candle
    whose hour bucket includes this timestamp). Returns the close price.

    Raises:
        ValueError: If no hourly candle data found for this outcome/timestamp.
    """
```

### 3. Engine → datetime-based (`TradingMoment`)

**Decision: Option B — full datetime engine refactor.** No backward compat.

The engine iterates over `TradingMoment` objects instead of date-string dicts.
This gives clean, honest semantics where timestamps in results accurately
reflect execution times. All downstream code (analyze.py, compare_delay_modes)
will be refactored to match — no attempt to preserve old interfaces.

```python
class TradingMoment(BaseModel):
    model_config = {"extra": "forbid"}

    timestamp: datetime              # UTC — when this trading decision happens
    active_snapshot: date            # which snapshot's predictions are used
    predictions: dict[str, float]    # {outcome: probability}
    prices: dict[str, float]         # {outcome: price_cents} at this timestamp

class BacktestEngine:
    def run(self, moments: list[TradingMoment], ...) -> BacktestResult:
```

**What changes:**
- `BacktestEngine.run()` takes `list[TradingMoment]` instead of two
  `dict[str, dict[str, float]]` args
- `PortfolioSnapshot.snapshot_date: str` → `snapshot_timestamp: datetime`
- `generate_signals()` gets real timestamps instead of synthetic ones
- `apply_signals()` fill dates become datetimes
- `daily_pnl.csv` dates become datetime strings (still one row per moment)
- All downstream analysis code updated accordingly

**What stays the same:**
- Signal generation logic (edge computation, Kelly sizing, etc.)
- Portfolio management (positions, fills, settlement)
- Core math doesn't care about date vs datetime

### 4. Results directory convention

**Current state:**
- `results/` — original delay=0 run (20 analysis files from analyze.py)
- `results_delay_0/` — delay=0 re-run (6 raw files from run_backtests.py)
- `results_delay_1/` — delay=1 (6 raw files)
- `results_inferred_6h/` — inferred+6h (6 raw files)

**New convention:**
- Rename `results/` → `results_delay_0/` (these ARE the delay=0 results)
- `results/` becomes a **symlink** → `results_inferred_6h/` (canonical mode)
- After refactor, canonical mode = inferred+6h with intraday prices
- `analyze.py` always reads from `results/` by default (resolves through symlink)
- `--results-dir` CLI arg overrides for comparing other modes

### 5. Backtest helper placement — options analysis

Where should extracted helpers live? Three options:

#### Option A: All in `trading/`

Move `build_backtest_inputs()`, `compute_snapshot_availability()`,
`get_active_snapshot_for_timestamp()`, `get_price_at_timestamp()` into
`trading/`.

**Pro:** Generic, reusable for future markets. Clean dependency direction.
**Con:** Some helpers (name matching, prediction remapping) are deeply
Oscar-specific and don't belong in generic `trading/`.

#### Option B: All in `one_offs/d20260220.../`

Keep everything under the one-off. Extract files within but don't promote.

**Pro:** Self-contained experiment. Easy to see all code for this experiment.
**Con:** Duplicates generic logic. Next experiment re-copies.

#### Option C: Split by generality (recommended)

| Location | What goes there | Why |
|----------|-----------------|-----|
| `trading/signal_delay.py` | `SignalDelayConfig`, `compute_snapshot_availability()`, `get_active_snapshot_for_timestamp()` | Market-agnostic delay computation |
| `trading/price_utils.py` | `get_price_at_timestamp()`, `get_market_prices_on_date()` (existing) | Market-agnostic price lookups |
| `trading/backtest.py` | `TradingMoment`, `BacktestEngine` with datetime iteration | Core engine |
| `one_offs/.../constants.py` | `MODELED_CATEGORIES`, `WINNERS_2025`, `MODEL_SHORT_NAMES`, etc. | 2025-specific constants |
| `one_offs/.../config.py` | `generate_trading_configs()`, `config_to_label()` | Experiment-specific config grid |
| `one_offs/.../data_prep.py` | `load_predictions()`, `remap_predictions_to_person_names()`, `build_trading_moments()`, name matching, etc. | Oscar + 2025-specific data wiring |
| `one_offs/.../market_fetch.py` | `_fetch_category_market_data()`, `run_market_favorite_baseline()` | Oscar + Kalshi data fetching |

**Recommendation: Option C.** Generic delay/price/engine logic goes into
`trading/` (reusable). Oscar+2025-specific wiring stays under the one-off
(experiment-scoped). This is the natural boundary — `trading/` should work
for any prediction market, one-off does the Oscar-specific plumbing.

### 6. analyze.py + README generation

- Add `--results-dir` CLI arg to `analyze.py` (easy — thread through
  `RESULTS_DIR`/`PLOTS_DIR` as function params instead of module globals)
- Add a `generate_readme_tables.py` script under the one-off that reads
  results CSVs and outputs markdown table fragments. Can be called from
  `analyze.py` or standalone. Helps regenerate README numbers after re-runs.

### 7. Predictions respect delay (already handled)

Predictions are indexed by snapshot date. The Feb 8 snapshot's predictions
(which include DGA winner features) are only used after Feb 8 snapshot
becomes active (e.g., Feb 9 at 12:30 UTC with 6h lag). Before that, the
Feb 7 snapshot's predictions (no DGA features) are used. The snapshot-based
model training pipeline inherently enforces this.

**No code change needed** — just documenting the confirmation.

---

## Phased Execution Plan

### Phase 0: Bugfixes and Small Wins (< 1 hour)

0a. **Fix month-boundary crash** in `compute_inferred_effective_dates()`.
    - `__init__.py` L305: `snap_date.day + 1` → `snap_date + timedelta(days=1)`
    - Current code constructs `datetime(year, month, day+1, ...)` which crashes
      on month boundaries (e.g., Jan 31 → day=32).

0b. **Add strict mode** for missing event times.
    - `strict: bool = True` parameter on `compute_inferred_effective_dates()`
    - Raise `ValueError` if `events_on_date` is empty in strict mode
    - Default fallback behavior preserved when `strict=False`

### Phase 1: SignalDelayConfig + Unified Available-At (small)

1a. Create `trading/signal_delay.py` with:
    - `FixedDayDelay`, `InferredLag`, `SignalDelayConfig` (discriminated union)
    - `compute_snapshot_availability()` → `dict[date, datetime]`
    - `get_active_snapshot_for_timestamp()` (new, datetime-based)
    - **Delete** old `compute_inferred_effective_dates()` +
      `get_active_snapshot_for_date()` from `__init__.py`. No backward compat.

1b. Update `SimulationConfig` to use `SignalDelayConfig` instead of
    `signal_delay_days: int`.

1c. Update `run_backtests.py` CLI to produce `SignalDelayConfig` from
    `--signal-delay-days` / `--inferred-lag-hours` args.

### Phase 2: Intraday Price Infrastructure (small-medium)

2a. **Fetch hourly candles** — add `fetch_hourly_candles()` to
    `oscar_market.py` using existing `get_batch_candlesticks(period_interval=60)`.
    Save to `storage/.../market_data/hourly_candles/`.

2b. **Build `get_price_at_timestamp()`** in `trading/price_utils.py`:
    - Hourly candles only, raise `ValueError` if not found.
    - No fallback chain — keep it simple.

2c. **Fetch hourly candle data** for all tickers in the 2025 experiment
    (Jan 23 – Mar 3 window).

**Note:** Trade pagination does NOT need fixing — research confirmed
`get_trades()` already fetches all pages. The page-3 log is informational.

### Phase 3: Datetime Engine Refactor (medium-large)

This is the big refactor. No backward compat — change everything consistently.

3a. **Refactor `BacktestEngine`** in `trading/backtest.py`:
    - Add `TradingMoment` model
    - `run()` takes `list[TradingMoment]` instead of two dict args
    - `PortfolioSnapshot.snapshot_date` → `snapshot_timestamp: datetime`
    - Update `generate_signals()` and `apply_signals()` for datetime
    - Update `BacktestResult` and CSV output to use datetime strings

3b. **Build `TradingMoment` construction** in `one_offs/.../data_prep.py`:
    - `build_trading_moments()` function that:
      1. Takes `snapshot_availability: dict[date, datetime]`, predictions,
         hourly candles, daily candles, trading date range
      2. For each trading day, determines active snapshot
      3. On snapshot activation day: uses hourly candle price at `available_at`
      4. On subsequent days: uses daily close price
      5. Returns `list[TradingMoment]`

3c. **Update `run_category_backtest()`** to use `build_trading_moments()` →
    `engine.run(moments)`.

3d. **Update downstream code:**
    - `analyze.py` — parse datetime strings in CSV, adapt all plots
    - `compare_delay_modes.py` — read datetime-indexed results
    - All CSV reading/writing updated consistently

### Phase 4: Refactor run_backtests.py (medium)

Current: 1,064 lines. Target: ~500 lines in `run_backtests.py`, rest extracted.

4a. Extract to **new files under the one-off**:
    - `constants.py` — `MODELED_CATEGORIES`, `CATEGORY_SLUGS`,
      `BACKTEST_MODEL_TYPES`, `MODEL_SHORT_NAMES`, `WINNERS_2025`
    - `config.py` — `generate_trading_configs()`, `config_to_label()`
    - `data_prep.py` — `load_predictions()`, `load_all_snapshot_predictions()`,
      `build_title_to_person_map()`, `remap_predictions_to_person_names()`,
      `renormalize_predictions()`, `build_market_prices()`,
      `make_spread_penalties_by_model_name()`, `get_winner_model_name()`,
      `evaluate_model_accuracy()`, `build_trading_moments()`
    - `market_fetch.py` — `_fetch_category_market_data()`,
      `run_market_favorite_baseline()`

4b. Move signal delay logic to `trading/signal_delay.py` (done in Phase 1).

4c. Slim `__init__.py` to:
    - `get_post_nomination_snapshot_dates()`, `get_trading_dates()`
    - Re-exports from `constants.py` for backward compat of imports

4d. Slim `run_backtests.py` to:
    - Imports + path constants (EXP_DIR, BANKROLL)
    - `run_category_backtest()` (slimmed, calls extracted helpers)
    - `main()` (CLI + orchestration)

### Phase 5: Tests (small-medium)

5a. `tests/trading/test_signal_delay.py`:
    - Known 2025 effective dates with 6h lag (the table from the README)
    - BAFTA same-day edge case (21:30 UTC + 6h = next-day UTC but same-day ET)
    - `get_active_snapshot_for_timestamp()` with various timestamps
    - Month-boundary dates (Jan 31 → Feb 1)
    - Strict mode: raises on missing event times
    - `FixedDayDelay` vs `InferredLag` produce correct `available_at`

5b. `tests/trading/test_price_utils.py` (extend existing):
    - `get_price_at_timestamp()` with synthetic hourly candle data
    - Error case: no hourly data for requested timestamp

### Phase 6: Re-run Experiments + Update README (medium-large)

6a. **Fetch hourly candle data** for all tickers.

6b. **Re-run backtests** with inferred+6h using intraday prices:
    ```bash
    uv run python -m ...run_backtests --inferred-lag-hours 6 \
        --results-dir storage/.../2025/results_inferred_6h
    ```

6c. **Set up canonical symlink**:
    ```bash
    cd storage/d20260220_backtest_strategies/2025/
    mv results results_delay_0    # rename original delay=0 results
    ln -s results_inferred_6h results
    ```

6d. **Run analyze.py** against canonical results (reads from `results/`).

6e. **Update README** with inferred+6h as canonical:
    - Keep signal delay analysis section (methodology context)
    - Rewrite all result sections using inferred+6h numbers
    - Keep delay=0 numbers only in the delay comparison section
    - Regenerate all plot assets

6f. **Run `sync_assets.sh`** to copy new plots to assets/.

---

## File Map (after refactor)

```
trading/
├── signal_delay.py          # NEW: SignalDelayConfig, compute_snapshot_availability
├── price_utils.py           # EXTEND: get_price_at_timestamp (hourly)
├── backtest.py              # REFACTOR: TradingMoment, datetime-based engine
├── oscar_market.py          # EXTEND: fetch_hourly_candles
└── ...                      # (kalshi_client.py unchanged)

one_offs/d20260220_backtest_strategies/
├── __init__.py              # SLIM: calendar utils + re-exports
├── constants.py             # NEW: MODELED_CATEGORIES, MODEL_SHORT_NAMES, WINNERS, etc.
├── config.py                # NEW: generate_trading_configs, config_to_label
├── data_prep.py             # NEW: load_predictions, build_trading_moments, name matching
├── market_fetch.py          # NEW: _fetch_category_market_data, market_favorite_baseline
├── run_backtests.py         # SLIM: ~500 lines, orchestration + CLI
├── analyze.py               # EXTEND: --results-dir CLI arg
├── generate_readme_tables.py # NEW: markdown table fragments from results
├── compare_delay_modes.py   # UPDATE: datetime-indexed results
├── README.md                # REWRITE: inferred+6h canonical
└── ...

storage/.../2025/
├── results/                 # SYMLINK → results_inferred_6h (canonical)
├── results_delay_0/         # RENAMED from old results/
├── results_delay_1/
├── results_inferred_6h/     # Canonical: inferred+6h with intraday prices
└── market_data/
    ├── candles/             # Daily candles (existing)
    └── hourly_candles/      # NEW: hourly OHLC candles
```

---

## Dependency Order

```
Phase 0 (bugfixes)
    │
Phase 1 (SignalDelayConfig)
    │
Phase 2 (hourly candles + get_price_at_timestamp)
    │
Phase 3 (datetime engine + TradingMoment) ──→ Phase 5 (tests)
    │
Phase 4 (refactor run_backtests.py)
    │
Phase 6 (re-run experiments + update README)
```

Phases 0→1→2 are sequential prerequisites. Phase 3 depends on 1+2.
Phase 4 can be interleaved with Phase 3 (extracting helpers doesn't depend
on engine changes). Phase 5 can be written alongside 3. Phase 6 depends on
everything.

---

## Open Questions

1. **Hourly candle completeness** — Need to verify that Kalshi returns hourly
   candles for the full Jan 23 – Mar 3 window across all tickers. If some
   tickers have sparse hourly data (low-volume nominees), we may need a
   fallback to daily close for those specific cases. We'll discover this during
   Phase 2c (data fetch) and adapt `get_price_at_timestamp()` if needed.

2. **Settlement handling in datetime engine** — Currently settlement uses the
   ceremony date as a date string. With datetime-based moments, settlement
   should be the last `TradingMoment` with a special flag, or handled outside
   the moment loop. Need to decide during Phase 3a implementation.
