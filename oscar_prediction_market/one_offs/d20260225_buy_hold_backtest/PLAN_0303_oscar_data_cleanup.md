# Plan: Trading Module Cleanup — Detailed Implementation

**Date:** 2026-03-03
**Scope:** `trading/` module architecture, config grid, IO isolation, name matching, DataFrame to Pydantic

---

## Table of Contents

1. [P1: Delete BacktestGridConfig — add label to BacktestConfig + factory function](#p1-delete-backtestgridconfig)
2. [P2: Move generate_trading_configs + generate_neighborhood_grid to trading/](#p2-move-config-generators)
3. [P3: Fix 3 stray .name.lower() to .slug](#p3-slug-cleanup)
4. [P4: Remove all dead backward-compat code](#p4-remove-dead-backward-compat)
5. [P5: Decompose oscar_data.py — separate IO from pure logic](#p5-decompose-oscar_data)
6. [P6: Accept only ceremony_year, derive ceremony_number](#p6-ceremony-parameter)
7. [P7: Eliminate datasets_dir threading — caller loads data upfront](#p7-eliminate-datasets_dir)
8. [P8: Inline CategoryMarketData — split fetch_category_market_data](#p8-inline-categorymarketdata)
9. [P9: Consolidate name matching + remap at load time](#p9-name-matching)
10. [P10: Move load_predictions to modeling/prediction_io.py](#p10-move-load_predictions)
11. [P11: Use Self type instead of string forward references](#p11-self-type)
12. [P12: print() to logging in library modules](#p12-print-to-logging)
13. [P13: Replace DataFrames with list[Candle] Pydantic model](#p13-candle-pydantic)
14. [P14: Clean up price_utils.py — move old functions to one-offs](#p14-price_utils-cleanup)
15. [P15: Rename actual_pnl to total_pnl at source](#p15-actual_pnl-rename)

---

## P1: Delete BacktestGridConfig

### What changes

1. **Delete** `trading/backtest_grid_config.py` entirely.
2. **Delete** `tests/trading/test_backtest_grid_config.py`.
3. **Add `label` computed field to `BacktestConfig`** in `trading/backtest.py`:

```python
@computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
@property
def label(self) -> str:
    """Short label for this config, suitable for display and filenames."""
    kc = self.trading.kelly
    side_str = (
        "yes"
        if self.trading.allowed_directions == frozenset({PositionDirection.YES})
        else "no"
        if self.trading.allowed_directions == frozenset({PositionDirection.NO})
        else "all"
    )
    return (
        f"fee={self.trading.fee_type.value}_kf={kc.kelly_fraction}"
        f"_bet={kc.buy_edge_threshold}_mp={self.trading.min_price}"
        f"_km={kc.kelly_mode.value}_bm={self.simulation.bankroll_mode.value}"
        f"_side={side_str}"
    )
```

The label extracts fields from nested sub-configs. Same format string as current `BacktestGridConfig.label`.

4. **Add factory function `make_oscar_backtest_config()`** in `trading/backtest.py`:

```python
def make_oscar_backtest_config(
    kelly_fraction: float,
    buy_edge_threshold: float,
    kelly_mode: KellyMode,
    fee_type: FeeType,
    allowed_directions: frozenset[PositionDirection],
    bankroll: float,
    mean_spread: float,
    min_price: float = 0.0,
    bankroll_mode: BankrollMode = BankrollMode.FIXED,
) -> BacktestConfig:
    """Construct BacktestConfig with Oscar-specific defaults.

    Hardcodes:
    - sell_edge_threshold = NEVER_SELL_THRESHOLD (buy-and-hold, no selling)
    - max_position_per_outcome = bankroll * 0.5
    - max_total_exposure = bankroll
    """
    return BacktestConfig(
        trading=TradingConfig(
            kelly=KellyConfig(
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                kelly_mode=kelly_mode,
                buy_edge_threshold=buy_edge_threshold,
                max_position_per_outcome=bankroll * 0.5,
                max_total_exposure=bankroll,
            ),
            sell_edge_threshold=NEVER_SELL_THRESHOLD,
            fee_type=fee_type,
            min_price=min_price,
            allowed_directions=allowed_directions,
        ),
        simulation=BacktestSimulationConfig(
            spread_penalty=mean_spread,
            bankroll_mode=bankroll_mode,
        ),
    )
```

5. **Add `with_runtime_params()` method to `BacktestConfig`**:

```python
def with_runtime_params(self, bankroll: float, mean_spread: float) -> Self:
    """Return a copy with runtime-dependent params filled in.

    Grid generators produce configs with sentinel bankroll/spread values.
    At backtest time, call this to bind the actual per-category values.
    """
    kelly = self.trading.kelly.model_copy(update={
        "bankroll": bankroll,
        "max_position_per_outcome": bankroll * 0.5,
        "max_total_exposure": bankroll,
    })
    trading = self.trading.model_copy(update={"kelly": kelly})
    simulation = self.simulation.model_copy(update={"spread_penalty": mean_spread})
    return self.model_copy(update={"trading": trading, "simulation": simulation})
```

This lets grid generators produce `list[BacktestConfig]` with placeholder `bankroll=1, mean_spread=0`. At runtime: `cfg = cfg.with_runtime_params(bankroll=1000, mean_spread=0.03)`.

6. **Update all callers** — currently 6 files import `BacktestGridConfig`:
   - `d20260220/run_backtests.py` — `generate_trading_configs` uses `to_backtest_config`
   - `d20260225/run_backtests.py` — imports from d20260220, uses `to_backtest_config`
   - `d20260224/generate_configs.py` — `RECOMMENDED_CONFIGS`, `generate_neighborhood_grid`
   - `d20260224/run_buy_hold.py` — `get_configs_for_model` returns `list[BacktestGridConfig]`
   - `d20260228/run_scenarios.py` — `_find_config_option`
   - `trading/__init__.py` — re-export (deleted in P4)

Each caller changes from `grid_cfg.to_backtest_config(bankroll, mean_spread)` to `cfg.with_runtime_params(bankroll, mean_spread)`.

### Files changed

| File | Change |
|---|---|
| `trading/backtest_grid_config.py` | **Delete** |
| `trading/backtest.py` | Add `label` computed field, `make_oscar_backtest_config()` factory, `with_runtime_params()` method |
| `trading/__init__.py` | Remove `BacktestGridConfig` re-export |
| `tests/trading/test_backtest_grid_config.py` | **Delete** (add equivalent label + with_runtime_params tests to `test_backtest.py`) |
| `d20260220/run_backtests.py` | `generate_trading_configs` returns `list[BacktestConfig]` using factory |
| `d20260225/run_backtests.py` | Same pattern |
| `d20260224/generate_configs.py` | `RECOMMENDED_CONFIGS` become `list[BacktestConfig]`; `generate_neighborhood_grid` produces `list[BacktestConfig]` |
| `d20260224/run_buy_hold.py` | Update types |
| `d20260228/run_scenarios.py` | Update `_find_config_option` |

---

## P2: Move Config Generators to trading/

### What changes

1. **Move `generate_trading_configs()`** from `d20260220/run_backtests.py` to a new `trading/config_grid.py`.

2. **Move `generate_neighborhood_grid()`** from `d20260224/generate_configs.py` to the same file.

3. **Keep the same grid dimensions for now** (7 x 7 x 2 x 2 x 3 = 588). Future work: trim to ~100 configs.

4. **Remove cross-one-off import** — d20260225 currently imports `generate_trading_configs` from d20260220. After this change, both import from `trading/`.

5. **Move `RECOMMENDED_CONFIGS`** from `d20260224/generate_configs.py` to `trading/config_grid.py` — these are the Pareto-optimal configs from cross-year analysis, reusable across experiments.

### Files changed

| File | Change |
|---|---|
| `trading/config_grid.py` | **New**: `generate_trading_configs()`, `generate_neighborhood_grid()`, `RECOMMENDED_CONFIGS` |
| `d20260220/run_backtests.py` | Remove `generate_trading_configs`, import from trading |
| `d20260225/run_backtests.py` | Import from trading instead of d20260220 |
| `d20260224/generate_configs.py` | Remove the moved functions and constants, import from trading |
| `d20260224/run_buy_hold.py` | Import from trading |

---

## P3: Fix 3 Stray .name.lower() to .slug

### Exact changes

```python
# trading/name_matching.py line 120
- slug = category.name.lower()
+ slug = category.slug

# trading/market_data/registry.py line 121
- slug = category.name.lower()
+ slug = category.slug

# data/build_dataset.py line 772
- cat_dir = base_dir / category.name.lower()
+ cat_dir = base_dir / category.slug
```

All are functionally identical (`name.lower()` == `.slug` for all `OscarCategory` values) but `.slug` is the canonical API.

---

## P4: Remove Dead Backward-Compat Code

### Exact deletions

**1. `d20260224_live_2026/__init__.py`** — delete legacy snapshot constants:
```python
# DELETE these lines:
AVAILABLE_SNAPSHOT_INFO = ...
AVAILABLE_SNAPSHOT_DATES = ...
AVAILABLE_SNAPSHOT_DATE_STRS = ...
```

Also delete unused re-exports of `OscarCategory`, `ModelType`, `get_trading_dates` (zero callers from outside the file).

**2. `d20260220_backtest_strategies/__init__.py`** — delete `WINNERS_2025` re-export (zero callers).

**3. `trading/__init__.py`** — remove ALL re-exports (zero callers). Reduce to docstring-only or empty file. Every caller already uses direct submodule imports.

**4. `modeling/generate_feature_ablation_configs.py`** — remove unused `model_type` parameter from `load_nonzero_features_from_csv()` and its single call site.

**5. `d20260224_live_2026/generate_report.py`** — remove MC metrics key remapping shim (lines ~263-291). Update downstream formatting code (lines ~549-557) to use canonical key names (`mean_pnl`, `cvar_05`, etc.) directly.

---

## P5: Decompose oscar_data.py

### Target structure after all changes

```
modeling/
  prediction_io.py          # NEW: load_predictions, load_all_snapshot_predictions,
                            #       load_ensemble_predictions (P10)

trading/
  oscar_market.py           # Unchanged scope: Kalshi API access, Candle model (P13)
  oscar_data.py             # SLIMMED: pure orchestration helpers only
    build_market_prices()           (pure)
    make_spread_penalties_by_model_name()  (pure)
    get_winner_model_name()         (pure)
    renormalize_predictions()       (pure)
    build_model_source()            (pure -- accepts pre-loaded data, P7)
    build_ensemble_source()         (pure -- accepts pre-loaded data, P7)
    build_title_to_person_map()     (pure -- accepts records, P7)
    build_screenplay_film_titles()  (pure -- extracted helper, P9)
  oscar_moments.py          # Unchanged in scope (builds MarketSnapshot)
  name_matching.py          # Enhanced: absorb remap logic from oscar_data (P9)
```

Functions that move OUT of `oscar_data.py`:
- `load_predictions`, `load_all_snapshot_predictions`, `load_ensemble_predictions` -> `modeling/prediction_io.py` (P10)
- `remap_predictions_to_person_names` -> **delete** (one-liner inlined into P10 load functions)
- `remap_to_person_names` -> **delete** (trivial wrapper)
- `build_nominee_match` -> **delete** (callers use `match_nominees` + `build_screenplay_film_titles` directly, P9)
- `load_individual_model_predictions` -> **delete** (callers compose the steps)
- `load_ensemble_model_predictions` -> **delete** (callers compose the steps)
- `fetch_category_market_data` -> split into 2 functions (P8)
- `CategoryMarketData` -> **delete** (P8)
- `SourceBuildResult` -> replaced with `tuple[TemporalModel, dict[str, str]]` (P8)
- `_resolve_dataset_path` -> **delete** (unnecessary when caller loads data, P7)

**After cleanup, oscar_data.py shrinks from ~1050 lines to ~300 lines** — all pure functions.

---

## P6: Accept Only ceremony_year

### What changes

Every function that takes `ceremony_number` loses that parameter. Internally: `ceremony_number = ceremony_year - 1928`.

Functions affected:
- `build_title_to_person_map(category, records, ceremony_year)` — filter `records` by `ceremony_year - 1928`
- `build_nominee_match(category, model_names, market, ..., ceremony_year)` — pass to `match_nominees`
- `build_model_source(...)` and `build_ensemble_source(...)` — same
- All callers drop the `ceremony_number=calendar.ceremony_number` arg

~15 call sites change. Mechanical.

---

## P7: Eliminate datasets_dir Threading

### Problem

`datasets_dir` threads through 8 functions so 2 JSON reads can happen deep in the call stack:
1. **Title-to-person map**: for acting/directing categories, reads nomination records to get `{film_title: person_name}`
2. **Screenplay film titles**: for original screenplay, reads nomination records to get `{person_name: film_title}`

Both read the same JSON file. Bug: `build_ensemble_source` calls `build_title_to_person_map` inside a loop over model types — same JSON read N times with identical result.

### Implementation

**Step 1:** Caller loads nomination records once:
```python
records = load_nomination_records(datasets_dir, ceremony_year)
```

**Step 2:** `build_title_to_person_map` becomes pure (accepts records, no IO):
```python
def build_title_to_person_map(
    category: OscarCategory,
    records: list[dict],     # raw nomination records
    ceremony_year: int,
) -> dict[str, str]:
    """Build {film_title: person_name} from nomination records.

    For KALSHI_PERSON_NAME_CATEGORIES (directing + 4 acting), Kalshi uses
    person names while our model uses film titles. This map enables the conversion.
    Returns {} for non-person-name categories.
    """
    if category not in KALSHI_PERSON_NAME_CATEGORIES:
        return {}
    ceremony_number = ceremony_year - 1928
    return {
        rec["title"]: rec["nominee_name"]
        for rec in records
        if rec["ceremony"] == ceremony_number and rec["category"] == category.value
    }
```

**Step 3:** `build_model_source` and `build_ensemble_source` accept pre-loaded data:
```python
def build_model_source(
    category: OscarCategory,
    model_type: ModelType,
    market: OscarMarket,
    predictions_by_snapshot: dict[str, dict[str, float]],  # pre-loaded by caller
    title_to_person: dict[str, str],                       # pre-loaded by caller
    ceremony_year: int,
    screenplay_film_titles: dict[str, str] | None = None,  # pre-loaded by caller
    ...
) -> tuple[TemporalModel, dict[str, str]] | None:
```

**Step 4:** Delete `_resolve_dataset_path`.

### Files changed

| File | Change |
|---|---|
| `trading/oscar_data.py` | Remove `_resolve_dataset_path`, change 6+ function signatures to accept data instead of paths |
| All 4 one-off callers | Load nomination records upfront, pass to functions |

---

## P8: Inline CategoryMarketData — Split fetch_category_market_data

### Current single function does 3 things

1. **Fetch + cache**: API calls -> parquet
2. **Spread estimation**: trades -> spread dict
3. **Bundle**: all results -> `CategoryMarketData`

### Implementation

**Split into 2 functions:**

```python
def fetch_and_cache_market_data(
    category: OscarCategory,
    ceremony_year: int,
    start_date: date,
    end_date: date,
    market_data_dir: Path,
    fetch_hourly: bool = False,
) -> tuple[OscarMarket, list[Candle], list[Candle] | None, pd.DataFrame]:
    """Fetch market data from Kalshi API and cache to disk.

    Returns:
        market: OscarMarket instance
        daily_candles: Daily OHLC candles for all nominees
        hourly_candles: Hourly OHLC candles (None if fetch_hourly=False)
        trades_df: Raw trade history DataFrame
    """
    ...

def estimate_category_spreads(
    trades_df: pd.DataFrame,
    tickers: list[str],
    default_spread: float = 0.03,
    min_trades: int = 10,
) -> tuple[dict[str, float], float]:
    """Estimate per-ticker spreads from trade history.

    Returns (spread_by_ticker, mean_spread).
    """
    ...
```

**Caller pattern (replaces CategoryMarketData destructuring):**
```python
market, daily_candles, hourly_candles, trades_df = fetch_and_cache_market_data(
    category, ceremony_year, start_date, end_date, market_data_dir, fetch_hourly=True
)
spread_by_ticker, mean_spread = estimate_category_spreads(
    trades_df, list(market.nominee_tickers.values())
)
```

**Delete `CategoryMarketData` class.** Also delete `kalshi_to_ticker` field — it was redundant alias for `market.nominee_tickers`.

**`SourceBuildResult` -> tuple**: `build_model_source` returns `tuple[TemporalModel, dict[str, str]]` or `None`. The `matched_model_names` field was just `set(nominee_map.keys())` — callers can derive it.

### Files changed

| File | Change |
|---|---|
| `trading/oscar_data.py` | Delete `CategoryMarketData`, `SourceBuildResult`, split `fetch_category_market_data` |
| `d20260220/run_backtests.py` | Replace `mkt = fetch_category_market_data(...)` with 2 calls |
| `d20260225/run_backtests.py` | Same |
| `d20260224/run_buy_hold.py` | Same |
| `d20260223/audit_uniform_lag_hours.py` | Same |

---

## P9: Consolidate Name Matching + Remap at Load Time

### Goal

Do title-to-person remapping at prediction load time so all predictions are immediately in Kalshi-compatible name space. This eliminates the separate remapping step that currently happens mid-pipeline.

### Category behavior

| Category | Model CSV "title" column | Kalshi names | Remap at load? |
|---|---|---|---|
| Best Picture | Film titles | Film titles | No |
| Animated Feature | Film titles | Film titles | No |
| Cinematography | Film titles | Film titles | No |
| Directing | Film titles | Person names | **Yes** (title to person) |
| Actor Leading | Film titles | Person names | **Yes** |
| Actress Leading | Film titles | Person names | **Yes** |
| Actor Supporting | Film titles | Person names | **Yes** |
| Actress Supporting | Film titles | Person names | **Yes** |
| Original Screenplay | **Person names** (despite column="title") | Film titles | **No** — already correct for matching, screenplay handled by `model_film_titles` fallback |

The 5 categories that need remapping are defined by `KALSHI_PERSON_NAME_CATEGORIES` in `data/oscar_winners.py:51-57`.

### Implementation

**Step 1:** In `modeling/prediction_io.py` (P10), add optional title-to-person remapping:

```python
def load_all_snapshot_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_keys: list[SnapshotInfo],
    models_dir: Path,
    title_to_person: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Load predictions across snapshots, optionally remapping titles to persons.

    For directing + 4 acting categories, Kalshi uses person names but model
    predictions use film titles. Pass title_to_person to remap at load time
    so predictions are immediately in Kalshi-compatible name space.
    """
    result = {}
    for key in snapshot_keys:
        preds = load_predictions(category, model_type, key, models_dir)
        if preds is not None:
            if title_to_person:
                preds = {title_to_person.get(k, k): v for k, v in preds.items()}
            result[str(key)] = preds
    return result
```

**Step 2:** Delete from `oscar_data.py`:
- `remap_predictions_to_person_names` — inlined as one-liner dict comprehension above
- `remap_to_person_names` — trivial wrapper
- `load_individual_model_predictions` — callers compose steps directly
- `load_ensemble_model_predictions` — callers compose steps directly
- `build_nominee_match` — callers use `match_nominees` + `build_screenplay_film_titles` directly

**Step 3:** Extract screenplay helper from `build_nominee_match`:
```python
def build_screenplay_film_titles(
    records: list[dict],
    ceremony_year: int,
) -> dict[str, str] | None:
    """For Original Screenplay: build {person_name: film_title} for matching fallback.

    Model predictions have person names in the "title" column for screenplay,
    and Kalshi uses film titles. match_nominees uses this as a fallback for fuzzy matching.
    """
    ceremony_number = ceremony_year - 1928
    titles = {
        rec["nominee_name"]: rec["title"]
        for rec in records
        if rec["ceremony"] == ceremony_number
        and rec["category"] == OscarCategory.ORIGINAL_SCREENPLAY.value
    }
    return titles if titles else None
```

**Step 4:** Delete `build_nominee_map` from `d20260228/run_scenarios.py` — use canonical `match_nominees`.

### Caller pattern after P9

```python
# 1. Load nomination records (once per ceremony)
records = load_nomination_records(datasets_dir, ceremony_year)
title_to_person = build_title_to_person_map(category, records, ceremony_year)

# 2. Load predictions (remapped at load time for person-name categories)
preds = load_all_snapshot_predictions(
    category, model_type, snapshot_keys, models_dir,
    title_to_person=title_to_person,  # {} for non-person categories, no-op
)

# 3. Name matching (predictions already in Kalshi-compatible name space)
screenplay_titles = (
    build_screenplay_film_titles(records, ceremony_year)
    if category == OscarCategory.ORIGINAL_SCREENPLAY
    else None
)
nominee_map = match_nominees(
    model_names=list(next(iter(preds.values())).keys()),
    kalshi_names=list(market.nominee_tickers.keys()),
    category=category,
    ceremony_year=ceremony_year,
    model_film_titles=screenplay_titles,
)
```

### Functions after consolidation

| Function | Location | Status |
|---|---|---|
| `normalize_person_name` | `data/utils.py` | **Keep** (used by data pipeline) |
| `normalize_name` | `name_matching.py` | **Keep** |
| `match_nominees` | `name_matching.py` | **Keep** |
| `validate_matching` | `name_matching.py` | **Keep** |
| `build_title_to_person_map` | `oscar_data.py` | **Keep** (now pure, accepts records) |
| `build_screenplay_film_titles` | `oscar_data.py` | **New** (extracted from build_nominee_match) |
| `get_winner_model_name` | `oscar_data.py` | **Keep** |
| `renormalize_predictions` | `oscar_data.py` | **Keep** |
| `remap_predictions_to_person_names` | -- | **Delete** (inlined into load) |
| `remap_to_person_names` | -- | **Delete** (wrapper) |
| `build_nominee_match` | -- | **Delete** (callers compose match_nominees + screenplay_titles) |
| `load_individual_model_predictions` | -- | **Delete** (callers compose steps) |
| `load_ensemble_model_predictions` | -- | **Delete** (callers compose steps) |
| `build_nominee_map` (d20260228) | -- | **Delete** (duplicate of match_nominees) |

---

## P10: Move load_predictions to modeling/prediction_io.py

### What moves

```python
# NEW FILE: modeling/prediction_io.py

def load_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_key: SnapshotInfo,
    models_dir: Path,
) -> dict[str, float] | None:
    """Load test predictions for one model + snapshot.

    Reads from: {models_dir}/{cat_slug}/{short_name}/{snapshot_key}/
                {run_name}/{5_final_predict or 2_final_predict}/predictions_test.csv

    Returns {model_name: probability} or None if file not found.
    """
    ...  # exact current implementation from oscar_data.py

def load_all_snapshot_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_keys: list[SnapshotInfo],
    models_dir: Path,
    title_to_person: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Load predictions across all snapshots, optionally remapping title to person.

    Returns {snapshot_key_str: {name: probability}}.
    """
    ...

def load_ensemble_predictions(
    category: OscarCategory,
    model_types: list[ModelType],
    snapshot_keys: list[SnapshotInfo],
    models_dir: Path,
    title_to_person: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Load + average predictions across model types, per snapshot.

    Returns {snapshot_key_str: {name: avg_probability}}.
    """
    ...
```

### Callers update

All callers change from:
```python
from ...trading.oscar_data import load_all_snapshot_predictions
```
to:
```python
from ...modeling.prediction_io import load_all_snapshot_predictions
```

---

## P11: Use Self Type

### Exact changes

```python
# data/schema.py
from typing import Self

class OscarCategory(StrEnum):
    @classmethod
    def from_slug(cls, slug: str) -> Self:
        return cls[slug.upper()]

# modeling/models/types.py
from typing import Self

class ModelType(StrEnum):
    @classmethod
    def from_short_name(cls, short_name: str) -> Self:
        ...
```

---

## P12: print() to logging in Library Modules

### Files to change

- `trading/oscar_data.py` — ~10 `print()` calls (status messages, warnings)
- `trading/oscar_moments.py` — ~5 `print()` calls
- `trading/oscar_market.py` — check for any

Replace with:
```python
import logging
logger = logging.getLogger(__name__)

# print(f"  Fetching market data for {cat_slug}...")
logger.info("Fetching market data for %s (%d)", cat_slug, ceremony_year)

# print(f"  WARNING: No market tickers for ...")
logger.warning("No market tickers for %s/%d: %s", category.name, ceremony_year, e)
```

One-off scripts and analysis scripts keep `print()` (they are scripts, not library code).

---

## P13: Replace DataFrames with list[Candle] Pydantic Model

### Problem

`hourly_prices_df` and `daily_prices_df` are DataFrames with implicit schemas. They're never used for vectorized computation — all access is row filtering + iteration. Pydantic models give schema clarity and validation.

### New model in `trading/oscar_market.py` (the producer)

```python
class Candle(BaseModel):
    """Single OHLC candlestick for one nominee/ticker."""
    model_config = {"extra": "forbid"}

    timestamp: datetime   # UTC, candle period end
    date: date
    ticker: str
    nominee: str
    open: int = Field(ge=0, le=100)     # cents
    high: int = Field(ge=0, le=100)
    low: int = Field(ge=0, le=100)
    close: int = Field(ge=0, le=100)
    volume: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def yes_price(self) -> int:
        """Close price expressed as yes-contract cents."""
        return self.close

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def no_price(self) -> int:
        """Implied no-contract price in cents."""
        return 100 - self.close

    @property
    def close_dollars(self) -> float:
        """Close price as a dollar fraction (0.0 to 1.0)."""
        return self.close / 100
```

Use the same `Candle` model for both daily and hourly granularity. Daily candles are just candles with a timestamp at midnight or EOD.

### Changes to oscar_market.py

`fetch_candlestick_history` returns `list[Candle]` instead of DataFrame:

```python
def fetch_candlestick_history(self, ...) -> list[Candle]:
    candles = []
    for record in raw_records:
        candles.append(Candle(
            timestamp=record["end_period_ts"],
            date=record["end_period_ts"].date(),
            ticker=ticker,
            nominee=self.nominee_tickers_inverse[ticker],
            open=record["price_open"],
            high=record["price_high"],
            low=record["price_low"],
            close=record["price_close"],
            volume=record["volume"],
        ))
    return candles
```

`get_daily_prices` -> returns `list[Candle]`.
`get_hourly_prices` -> returns `list[Candle]`.

### Caching

Use JSON via TypeAdapter (cleaner than parquet for small datasets):
```python
from pydantic import TypeAdapter
_candle_list_adapter = TypeAdapter(list[Candle])

# Save
path.write_bytes(_candle_list_adapter.dump_json(candles))
# Load
candles = _candle_list_adapter.validate_json(path.read_bytes())
```

### Downstream changes

| File | Current | After |
|---|---|---|
| `oscar_data.build_market_prices` | Iterates `daily_prices_df.iterrows()` | Iterates `list[Candle]` directly |
| `oscar_data.fetch_category_market_data` | Returns DF in CategoryMarketData | Returns `list[Candle]` (P8) |
| `oscar_moments._get_daily_prices_with_fallback` | Looks up dict from `build_market_prices()` | Same dict pattern, built from `list[Candle]` |
| `oscar_moments._get_hourly_price_at_timestamp` | Calls `price_utils.get_hourly_prices_at_timestamp(df)` | Inlined: filter `list[Candle]` by timestamp window |
| `price_utils.get_hourly_prices_at_timestamp` | DataFrame boolean masks | **Delete** — fold into oscar_moments (P14) |
| `analysis/data_loading.py` | Queries daily DF | Update to iterate `list[Candle]` |

---

## P14: Clean Up price_utils.py

### Current state

| Function | Used by current pipeline? | Used by old one-offs only? |
|---|---|---|
| `get_market_prices_on_date` | No | Yes (d20260214) |
| `build_prices_by_date` | No | Yes (d20260214, d20260219) |
| `serialize_daily_prices` | No | Yes (d20260214) |
| `deserialize_daily_prices` | No | Yes (d20260214) |
| `get_hourly_prices_at_timestamp` | Yes (oscar_moments) | No |

### Implementation

1. **Move 4 old-only functions** to the one-offs that use them:
   - `get_market_prices_on_date` + `build_prices_by_date` -> `d20260214_trade_signal_ablation/price_helpers.py` (or inline)
   - `serialize_daily_prices` + `deserialize_daily_prices` -> same
   - The old one-offs should continue to work with DataFrames since they predate the Candle model

2. **Fold `get_hourly_prices_at_timestamp`** into `oscar_moments.py` as a private `_get_hourly_price_at_timestamp()`. After P13, it operates on `list[Candle]` instead of DataFrame:

```python
def _get_hourly_price_at_timestamp(
    candles: list[Candle],
    ticker: str,
    target_ts: datetime,
    window_hours: int = 2,
) -> int | None:
    """Find the closest hourly candle for a ticker near target_ts.

    Searches within +/- window_hours. Returns close price in cents, or None.
    """
    best: Candle | None = None
    best_delta = timedelta(hours=window_hours)
    for c in candles:
        if c.ticker != ticker:
            continue
        delta = abs(c.timestamp - target_ts)
        if delta <= best_delta:
            best = c
            best_delta = delta
    return best.close if best else None
```

3. **Delete `trading/price_utils.py`** entirely.

4. **Delete `tests/trading/test_price_utils.py`** — tests cover old functions only (not the active `get_hourly_prices_at_timestamp`).

---

## P15: Rename actual_pnl to total_pnl at Source

### Problem

`run_backtests.py` writes `actual_pnl` in entry_pnl.csv. 3 downstream scripts rename it to `total_pnl`. `scenario_scoring.py` uses `actual_pnl` directly (~12 locations).

### Implementation

**Step 1:** In `d20260225/run_backtests.py`, rename the output column:
```python
# line ~433
- "actual_pnl": actual_pnl,
+ "total_pnl": actual_pnl,
```

**Step 2:** In `d20260225/run_backtests.py`, update the aggregation (line ~604):
```python
- total_pnl=("actual_pnl", "sum"),
+ total_pnl=("total_pnl", "sum"),
```

**Step 3:** Remove 3 rename shims from downstream scripts:
- `analyze.py` lines 31-33 (`.rename(columns={"actual_pnl": "total_pnl"})`)
- `analyze_plots.py` lines 121-123
- `generate_tables.py` lines 53-55

**Step 4:** In `scenario_scoring.py`, replace all `"actual_pnl"` -> `"total_pnl"` (~12 occurrences at lines 138, 149, 154, 175, 178, 192, 213, 285, 514, 542-543, 603-604).

**Step 5:** In `analyze_scenario_plots.py`, replace `"actual_pnl"` -> `"total_pnl"` (~4 occurrences at lines 305, 383, 667, 1099).

**Step 6:** In `generate_tables.py`, check remaining `"actual_pnl"` references (lines 612, 869, 901 that read scenario_scoring output).

**Step 7:** Rewrite existing CSV headers in-place (no need to re-run expensive backtests):
```bash
cd "$(git rev-parse --show-toplevel)"
for f in storage/d20260225_buy_hold_backtest/*/results/entry_pnl.csv; do
    sed -i '' 's/actual_pnl/total_pnl/g' "$f"
done
```

**Step 8:** Re-run scenario_scoring + analysis scripts to regenerate derived CSVs.

---

## Execution Order and Dependencies

```
Phase 1 -- Quick mechanical fixes (independent, do first):
  P3:  .slug cleanup (5 min)
  P11: Self type (5 min)
  P12: print to logging (20 min)
  P15: actual_pnl to total_pnl (30 min + CSV rewrite + rerun scoring)

Phase 2 -- Backward compat cleanup:
  P4:  Remove dead code (20 min)

Phase 3 -- Config grid refactor:
  P1:  Delete BacktestGridConfig (2 hrs) -- depends on P4 (trading/__init__.py cleanup)
  P2:  Move generators to trading/ (1 hr) -- depends on P1

Phase 4 -- IO isolation (the big one, do together):
  P6:  ceremony_year only (30 min)
  P10: Move load_predictions to modeling/ (1 hr)
  P7:  Eliminate datasets_dir (1 hr) -- depends on P6, P10
  P9:  Name matching consolidation + remap at load time (2 hrs) -- depends on P7
  P5:  Final oscar_data.py cleanup (1 hr) -- depends on P7, P9, P10

Phase 5 -- DataFrame to Pydantic:
  P13: Candle model (2-3 hrs) -- independent but large; touches oscar_market, oscar_data, oscar_moments
  P8:  Split fetch_category_market_data (1 hr) -- depends on P13
  P14: Clean up price_utils (30 min) -- depends on P13

Total estimated: ~12-14 hours across 5 phases
```

---

## Open Questions

**Q1 (P1):** For the grid generator, the plan proposes `BacktestConfig` with sentinel values + `with_runtime_params()`. This keeps one type everywhere but means configs temporarily have fake bankroll/spread values. Alternative: lightweight `NamedTuple` for swept dimensions. Recommendation is `with_runtime_params` approach -- confirm?

**Q2 (P13):** For caching candles, JSON (via TypeAdapter) vs keep parquet? JSON is cleaner with Pydantic; parquet is more compact but these datasets are small (~500 rows). Recommendation: JSON.

**Q3 (P9):** Screenplay special case: model CSV has person names in the "title" column, Kalshi has film titles. The remap-at-load-time approach applies to the 5 person-name categories only and leaves screenplay unchanged. `title_to_person` returns `{}` for screenplay (no remap). Screenplay matching continues to use the `model_film_titles` fallback in `match_nominees`. This is the cleanest approach -- confirm?

**Q4 (P13):** Should `analysis/data_loading.py` also be updated to use `list[Candle]`, or is that out of scope? It queries daily prices for calibration plots.
