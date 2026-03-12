# Consolidate Buy-Hold Pipeline into `trading/`

**Created:** 2026-03-03
**Updated:** 2026-03-03 (v2 — refined after discussion)
**Status:** Planning

## Problem Statement

Four one-offs have evolved into a layered import chain where `d20260220` acts as
a de facto shared library:

```
d20260220_backtest_strategies   ← constants, data_prep, BacktestGridConfig, evaluation
        ▲           ▲
        │           │
d20260225_buy_hold_backtest     ← year-parameterized data_loading, buy-hold engine
        ▲
        │
d20260224_live_2026             ← live scenario reports, orderbook pricing
        ▲
        │
d20260228_pga_scenario_analysis ← PGA what-if (skip from this refactor)
```

**32 cross-import sites** from `d20260220`, plus one-offs importing from each
other. Character-identical function duplication, hardcoded constants, and a
950-line `data_prep.py` locked to `CEREMONY_YEAR=2025` with a 394-line
year-parameterized shim on top.

## Scope

Refactor shared code into `trading/`. Update all consumers: d20260220, d20260224,
d20260225. Skip d20260228 (PGA scenario analysis) — fundamentally different data
pipeline (counterfactual feature engineering).

Don't worry about backward compatibility — update all importers to use the
refactored code consistently.

---

## Issue Inventory

### I1: `MODEL_SHORT_NAMES` dict is redundant with `ModelType.short_name`

`d20260220/__init__.py` defines `MODEL_SHORT_NAMES: dict[ModelType, str]` with 4
entries. But `ModelType` already has a `.short_name` property (defined in
`modeling/models/types.py`) with the exact same values:

```
LOGISTIC_REGRESSION  → "lr"       (both)
CONDITIONAL_LOGIT    → "clogit"   (both)
GRADIENT_BOOSTING    → "gbt"      (both)
CALIBRATED_SOFTMAX_GBT → "cal_sgbt" (both)
```

Every `MODEL_SHORT_NAMES[model_type]` can be replaced with
`model_type.short_name`.

The dict also doubles as a "valid model types for this experiment" gate, but
that's already served by `BACKTEST_MODEL_TYPES` list.

Similarly, `SHORT_NAME_TO_MODEL` can become a classmethod on `ModelType`.

### I2: `CATEGORY_SLUGS` is a dict, not a property on `OscarCategory`

`CATEGORY_SLUGS: dict[OscarCategory, str]` maps categories to slug strings (e.g.,
`BEST_PICTURE → "best_picture"`). It lives in
`modeling/feature_engineering/groups.py` and is re-exported through 3 layers.

Usage is overwhelmingly `CATEGORY_SLUGS[category]` for **path construction** and
display. This should be a `.slug` property on `OscarCategory` (like
`ModelType.short_name`).

### I3: `MODELED_CATEGORIES` / `BACKTEST_MODEL_TYPES` are experiment config, not library constants

These live in `d20260220/__init__.py`:
- `MODELED_CATEGORIES`: the 9 Oscar categories with Kalshi markets
- `BACKTEST_MODEL_TYPES`: the 4 model types used in backtests
- `ENSEMBLE_SHORT_NAME`: `"avg_ensemble"`

These are **experiment-scoped choices** — which categories to trade, which models
to run. They don't belong in `trading/` (library code). The caller should own
these decisions and pass them as parameters.

Currently re-exported by d20260224's `__init__.py` and imported across 8 files.

### I4: `BacktestGridConfig` lives in a one-off but is used by 4 one-offs

Defined in `d20260220/run_backtests.py`. Imported by d20260224, d20260225,
d20260228. It's a general-purpose parameter grid point — belongs in `trading/`.

### I5: Config construction boilerplate (20 lines) repeated 4×

The block `BacktestGridConfig → KellyConfig → TradingConfig →
BacktestSimulationConfig → BacktestConfig` is repeated in d20260220, d20260225,
d20260224 (twice), d20260228. The only variation is the spread source.

### I6: `BANKROLL`, `DEFAULT_SPREAD`, `MIN_TRADES_FOR_SPREAD` scattered

- `BANKROLL = 1000.0` in d20260220/data_prep.py and d20260225/scenario_scoring.py
- `DEFAULT_SPREAD = 0.03` in d20260220/data_prep.py and d20260228/run_scenarios.py
- `MIN_TRADES_FOR_SPREAD = 10` in d20260220/data_prep.py

These are caller concerns. `BANKROLL` is a backtest parameter (how much money),
not a library constant. `DEFAULT_SPREAD` and `MIN_TRADES_FOR_SPREAD` are spread
estimation defaults.

### I7: `data_prep.py` (954 lines) hardcoded to 2025; `data_loading.py` (394 lines) exists as year-parameterized shim

6 functions are duplicated — same logic, but `data_loading.py` takes
`models_dir`, `datasets_dir`, `ceremony_year` parameters instead of using
module-level constants. Also adds glob-fallback for date-only snapshot keys.

Pure-transform functions (`remap_predictions_to_person_names`,
`renormalize_predictions`, `build_market_prices`,
`make_spread_penalties_by_model_name`) exist only in `data_prep.py` and are
imported directly by d20260225 and d20260224.

### I8: Character-identical function duplication between d20260224 and d20260225

| Function | Lines | Status |
|----------|-------|--------|
| `_remap_person_names` | 17 | 100% identical |
| `_load_individual_model` / `_load_and_remap_individual` | 18 | 100% identical (different name) |
| `_load_ensemble_model` / `_load_and_remap_ensemble` | 16 | 100% identical (different name) |

### I9: `run_category_buy_hold` ~80% identical between d20260224 and d20260225

Both follow the same skeleton: fetch market data → load models → match nominees →
build entry moments → iterate configs → construct BacktestConfig → run engine →
collect results. The ~20% that differs: settlement handling (known winner vs
scenario matrix), output schema, live mode, config selection.

### I10: Monte Carlo simulation duplicated with divergent features

`generate_report.py` and `scenario_scoring.py` both implement the same MC loop:
```python
for _cat, (winners, probs, pnls) in cat_scenarios.items():
    indices = rng.choice(len(winners), size=n_samples, p=probs)
    total_pnls += pnls[indices]
```
But use different probability weighting (model-only vs blend), output metrics,
and aggregation.

### I11: `YearConfig` mixes data concerns with storage paths

`YearConfig` bundles: calendar, winners, categories (domain data) + models_dir,
datasets_dir, results_dir, plots_dir, market_data_dir (storage layout). It also
hardcodes two experiment-specific storage roots (`BACKTEST_EXP_DIR`,
`BUY_HOLD_EXP_DIR`).

Meanwhile d20260224's `__init__.py` maintains its own overlapping set of
constants: `CEREMONY_YEAR`, `CALENDAR`, `EXP_DIR`, `_TODAY`, snapshot filtering
logic.

### I12: Naming inconsistencies

| Concept | Names used |
|---------|-----------|
| Snapshot identifier | `snapshot_key`, `key`, `key_str`, `snap_key_str`, `snapshot_key_str`, `entry_snapshot` |
| Trading config grid point | `cfg`, `config`, `trading_config`, `grid_cfg` |
| Market data bundle | `mkt`, `market_data`, `category_market_data` |
| Model label | `short_name`, `model_type_label`, `label` |
| Categories to trade | `MODELED_CATEGORIES`, `categories`, `CATEGORIES_BY_YEAR[year]` |
| Ceremony year range | `train_years` (string `"2000-2025"`), `test_years` (string `"2026"`) |

---

## Proposed Changes

### P1: Add `.slug` property to `OscarCategory`

Add a `slug` property to `OscarCategory` in `data/schema.py`, following the same
pattern as `ModelType.short_name`:

```python
class OscarCategory(StrEnum):
    BEST_PICTURE = "BEST PICTURE"
    ...

    @property
    def slug(self) -> str:
        return _CATEGORY_SLUGS[self]

    @classmethod
    def from_slug(cls, slug: str) -> "OscarCategory":
        return _SLUG_TO_CATEGORY[slug]
```

Define slugs for **all 20 categories** (mechanical `name.lower()` transform:
`"BEST PICTURE" → "best_picture"`). No special-casing needed since the slug
pattern is uniform.

Replace all `CATEGORY_SLUGS[category]` with `category.slug` across the codebase.
Then remove `CATEGORY_SLUGS` from `feature_engineering/groups.py` and its
re-exports.

### P2: Eliminate `MODEL_SHORT_NAMES` and `SHORT_NAME_TO_MODEL`

Replace all `MODEL_SHORT_NAMES[model_type]` with `model_type.short_name`.

Add `ModelType.from_short_name(name: str) -> ModelType` classmethod to replace
`SHORT_NAME_TO_MODEL`.

### P3: Remove `MODELED_CATEGORIES`, `BACKTEST_MODEL_TYPES`, `ENSEMBLE_SHORT_NAME` from shared imports

These are **experiment configuration**, not library constants. Each one-off
defines its own list of categories and model types. The lib code (`trading/`)
never references them.

Keep them as local constants in each one-off that needs them. No wrapper model
needed — just plain lists.

### P4: Move `BacktestGridConfig` → `trading/backtest_grid_config.py`

Move `BacktestGridConfig` from `d20260220/run_backtests.py` into a new file
`trading/backtest_grid_config.py`.

**`generate_trading_configs()` stays in the one-off** — the specific grid values
(kelly fractions, edge thresholds, direction sets) are experiment-specific.
`BacktestGridConfig` is the reusable schema; the grid generation is the
experiment design.

#### Why two configs? `BacktestGridConfig` vs `BacktestConfig`

These serve different roles and should remain separate:

| Aspect | `BacktestGridConfig` | `BacktestConfig` |
|--------|---------------------|-----------------|
| **Role** | Grid search parameter point | Full engine runtime config |
| **Fields** | 7 sweepable params | 7 sweep params + 6 fixed/runtime params |
| **Lifetime** | Experiment design — persists across all runs | Per-category-run — varies by spread data |
| **Identity** | `.label` property — used as CSV column / groupby key | No label; not serialized to CSV |
| **Constructed** | Once at experiment setup | Per category × config, from grid + runtime data |

The 6 fields in `BacktestConfig` but NOT in `BacktestGridConfig`:
- `bankroll`, `max_position_per_outcome`, `max_total_exposure` — hardcoded
  constants identical across all callers (but caller-owned, not lib constants)
- `sell_edge_threshold = NEVER_SELL_THRESHOLD` — sentinel for buy-and-hold
- `spread_penalty` — **runtime data**, varies per category (from market trades)
- `max_contracts_per_day` — unused (defaults to `None`)

`BacktestGridConfig` is a **strict subset** of `BacktestConfig`'s information.
The expansion from grid point → engine config is the job of
`to_backtest_config()` (see P5). Merging them would force runtime data
(`spread_penalty`) into the grid definition, conflating experiment design with
execution.

### P5: Add `BacktestGridConfig.to_backtest_config()` method

Eliminate the 20-line construction boilerplate by adding a method:

```python
class BacktestGridConfig(BaseModel):
    ...

    def to_backtest_config(
        self,
        bankroll: float,
        mean_spread: float,
    ) -> BacktestConfig:
        """Convert grid config + runtime params → full BacktestConfig.

        Args:
            bankroll: Total bankroll per category (e.g., $1000).
            mean_spread: One-way spread penalty in dollars (from market data).
        """
        kelly_config = KellyConfig(
            bankroll=bankroll,
            kelly_fraction=self.kelly_fraction,
            kelly_mode=self.kelly_mode,
            buy_edge_threshold=self.buy_edge_threshold,
            max_position_per_outcome=bankroll * 0.5,
            max_total_exposure=bankroll,
        )
        trading_config = TradingConfig(
            kelly=kelly_config,
            sell_edge_threshold=NEVER_SELL_THRESHOLD,
            fee_type=self.fee_type,
            min_price=self.min_price,
            allowed_directions=self.allowed_directions,
        )
        sim_config = BacktestSimulationConfig(
            spread_penalty=mean_spread,
            bankroll_mode=BankrollMode(self.bankroll_mode),
        )
        return BacktestConfig(trading=trading_config, simulation=sim_config)
```

Callers reduce from 20 lines to 1:
```python
bt_config = cfg.to_backtest_config(bankroll=1000.0, mean_spread=market_data.mean_spread)
```

**Design notes:**
- `bankroll` is a parameter, not a constant — the caller owns it.
- `max_position_per_outcome = bankroll * 0.5` stays hardcoded (convention across
  all one-offs). Could be a field on `BacktestGridConfig` if we ever want to
  sweep it.
- `sell_edge_threshold = NEVER_SELL_THRESHOLD` is hardcoded for buy-and-hold.
  Could optionally be a field too.

### P6: Consolidate data loading → `trading/oscar_data.py`

Create a new module `trading/oscar_data.py` containing the year-parameterized
versions of data loading functions. This merges `d20260220/data_prep.py` and
`d20260225/data_loading.py` into one canonical location.

**Functions to move (parameterized versions):**

| Function | Current home | Notes |
|----------|-------------|-------|
| `load_predictions(category, model_type, snapshot_key, models_dir)` | d20260225 | Add glob fallback from d20260225 |
| `load_all_snapshot_predictions(category, model_type, snapshot_keys, models_dir)` | d20260225 | Thin wrapper |
| `load_ensemble_predictions(category, model_types, snapshot_keys, models_dir)` | d20260225 | |
| `build_title_to_person_map(category, snapshot_key_str, datasets_dir, ceremony_number)` | d20260225 | |
| `remap_predictions_to_person_names(preds_by_snapshot, title_to_person)` | d20260220 | Pure transform |
| `renormalize_predictions(predictions, matched_names)` | d20260220 | Pure transform |
| `build_market_prices(daily_prices_df, nominee_map)` | d20260220 | Pure transform |
| `make_spread_penalties_by_model_name(spread_by_ticker, nominee_map, ticker_map)` | d20260220 | Pure transform |
| `get_winner_model_name(category, nominee_map, winners)` | d20260225 | Takes `winners` dict |
| `fetch_category_market_data(category, ceremony_year, ...)` | d20260225 | See interface redesign in P11 |
| `build_nominee_match(category, model_names, market, ...)` | d20260225 | |
| `CategoryMarketData` | d20260220 | Pydantic model (see below) |
| `build_entry_moment(...)` | d20260225 | See below |

**`CategoryMarketData` cleanup:**

Move the model to `trading/oscar_data.py` and clean up dead fields:
- **Drop `trades_df`** — stored but never accessed by any consumer. Used only
  transiently inside `fetch_category_market_data` to compute spreads.
- **Drop `ticker_to_kalshi`** — never accessed. Only `kalshi_to_ticker` is used.
- **Drop `spread_report`** — only used in d20260220 for debugging. Move to a
  local helper there if needed.
- Keep `arbitrary_types_allowed=True` for DataFrames. Fine for a short-lived
  in-memory data bundle — never serialized, never stored.

**`build_entry_moment` in `trading/oscar_data.py`:**

Currently 77 lines in d20260225/run_backtests.py. Builds a `MarketSnapshot` from
predictions + market prices at one entry timestamp, with Oscar-specific fallback
logic (7-day forward search for daily prices, backward forward-fill for missing
nominees). Despite being Oscar-specific, it belongs in `oscar_data.py` since that
module is already Oscar-domain code. Moving it here also eliminates its dependency
on `renormalize_predictions` from d20260220.

**Functions NOT to move:**
- `build_trading_moments` — only used by d20260220's rebalancing backtest.
- `build_model_source`, `build_ensemble_source` — specific to rebalancing
  backtest's TemporalModel construction. Keep in d20260220.

### P7: Extract shared model-loading helpers

The identical `_remap_person_names`, `_load_individual_model`,
`_load_ensemble_model` should live once in `trading/oscar_data.py`:

```python
def remap_to_person_names(
    preds_by_snapshot: dict[str, dict[str, float]],
    category: OscarCategory,
    first_snapshot_key: str,
    datasets_dir: Path,
    ceremony_number: int,
) -> dict[str, dict[str, float]]:
    """Remap title-keyed predictions to person names for person-name categories."""

def load_individual_model_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_keys: list[SnapshotInfo],
    datasets_dir: Path,
    models_dir: Path,
    ceremony_number: int,
) -> tuple[str, dict[str, dict[str, float]]] | None:
    """Load predictions for one model, remap person names.

    Returns (model_label, preds_by_snapshot) or None if no predictions found.
    """

def load_ensemble_model_predictions(
    category: OscarCategory,
    label: str,
    model_types: list[ModelType],
    snapshot_keys: list[SnapshotInfo],
    datasets_dir: Path,
    models_dir: Path,
    ceremony_number: int,
) -> tuple[str, dict[str, dict[str, float]]] | None:
    """Load and average predictions from multiple models, remap person names.

    Returns (label, preds_by_snapshot) or None if no predictions found.
    """
```

### P8: Extract MC portfolio simulation → `trading/portfolio_simulation.py`

Create `trading/portfolio_simulation.py` with a core sampling function.

**Why `portfolio_simulation.py`:** The shareable code includes both MC sampling
and portfolio-level stat computation (CVaR, distribution metrics). "monte_carlo"
is too narrow (the Pareto/aggregation functions aren't MC); "portfolio_simulation"
covers the full scope.

```python
class CategoryScenario(NamedTuple):
    """One category's winner outcomes with probabilities and P&L values."""
    winners: np.ndarray   # shape (n_outcomes,) — outcome names
    probs: np.ndarray     # shape (n_outcomes,) — must sum to 1
    pnls: np.ndarray      # shape (n_outcomes,) — P&L if that outcome wins


def sample_portfolio_pnl(
    category_scenarios: dict[str, CategoryScenario],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo portfolio P&L simulation over mutually exclusive category outcomes.

    For each sample, independently draws one winner per category from the
    probability distribution, looks up the corresponding P&L, and sums across
    categories.

    Returns:
        Array of shape (n_samples,) with total portfolio P&L per draw.
    """


def compute_cvar(pnls: np.ndarray, alpha: float) -> float:
    """Conditional Value at Risk — mean of worst alpha-fraction of outcomes."""


def compute_portfolio_mc_metrics(
    pnls: np.ndarray,
    bankroll: float,
    capital_deployed: float | None = None,
) -> dict[str, float]:
    """Compute full set of MC metrics: mean, std, CVaR, loss probs, ROIC, etc."""
```

Both `generate_report.py` and `scenario_scoring.py` call `sample_portfolio_pnl`
then apply their own metric computation on top. The probability weighting
(model-only vs blend) happens at the caller level when constructing the
`CategoryScenario.probs`.

**NOT moving:** The Pareto frontier computation, cross-year scoring, and CLI
orchestration stay in `scenario_scoring.py` — these are analysis-layer
aggregations, not reusable library code.

### P9: Simplify `YearConfig` — no new `CeremonyConfig` model

**Key finding:** The domain data that a hypothetical "CeremonyConfig" would hold
is already available from existing sources:

| Data | Existing source |
|------|----------------|
| Calendar | `CALENDARS[ceremony_year]` → `AwardsCalendar` |
| Winners | `WINNERS_BY_YEAR[ceremony_year]` → `dict[OscarCategory, str]` |
| Categories with markets | `OSCAR_MARKETS.categories_for_year(ceremony_year)` |
| Ceremony number | `CALENDARS[ceremony_year].ceremony_number` |
| Snapshot keys | `get_snapshot_info(CALENDARS[ceremony_year])` |

Creating a new `CeremonyConfig` model would just duplicate these lookups into
another object. Instead:

**A. Library functions take primitives** — `ceremony_year: int`, then look up
what they need internally, or take the specific data they need (`calendar:
AwardsCalendar`, `winners: dict[OscarCategory, str]`).

**B. `YearConfig` stays in the one-off**, simplified:
- **Keep:** `ceremony_year`, `models_dir`, `datasets_dir`, `results_dir`,
  `market_data_dir` — these are experiment-specific storage paths that vary per
  year.
- **Remove as stored fields:** `calendar`, `winners`, `categories` — these become
  `@computed_field` or `@property` that look up from existing registries. Or just
  let callers look them up directly via `CALENDARS[year]`.
- **Remove:** `train_years` (CLI formatting concern, compute inline).

**C. `CATEGORIES_BY_YEAR` stays in the one-off.** It represents "categories we
chose to model" — a subset of `OSCAR_MARKETS.categories_for_year()` (which
returns ALL categories with markets: 13–17 per year). It's experiment-specific
domain knowledge with zero external importers.

**D. No `ExperimentPaths` model.** Just pass individual `Path` parameters
(`models_dir`, `datasets_dir`, etc.). The paths are only used in the
orchestration layer and there aren't enough of them to justify a model.

**E. d20260224's `__init__.py`** simplifies by using `CALENDARS` directly:

```python
CEREMONY_YEAR = 2026
CALENDAR = CALENDARS[CEREMONY_YEAR]

def available_snapshots(as_of: date | None = None) -> list[SnapshotInfo]:
    """Snapshot keys available as of a given date (default: today)."""
    cutoff = as_of or date.today()
    all_snapshots = get_snapshot_info(CALENDAR)
    return [s for s in all_snapshots if s.event_datetime_utc.date() <= cutoff]
```

The hardcoded `_TODAY = date(2026, 3, 2)` becomes a function parameter.

### P10: Standardize naming conventions

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `snapshot_key` / `key` / etc. | `entry_point` | Consistent across codebase |
| `snapshot_key_strs` / `key_str` | `[ep.dir_name for ep in ...]` inline | Stop the str/SnapshotInfo duality |
| `entry_snapshot` (CSV column) | `entry_point_name` | Consistent with `entry_point.dir_name` |
| `cfg` / `config` / `grid_cfg` | `bt_grid_config` for BacktestGridConfig | Distinguish from `BacktestConfig` (`bt_config`) |
| `mkt` | `market_data` | Spell it out |
| `short_name` (variable) | `model_label` when used as dict key / CSV column | `short_name` is fine as a property |
| `model_runs` | `model_predictions` or `loaded_models` | More descriptive |
| `preds_by_snapshot` | Keep — it's descriptive | |
| `nominee_map` | Keep — it's descriptive | |
| `train_years` (string) | `train_year_range` or compute inline | "years" suggests a list; it's a range string |

### P11: Clean up function interfaces

**Principle:** Functions in `trading/` take **primitive or well-defined model
parameters**, not fat config objects. This keeps them testable and reusable.

**`fetch_category_market_data`** — instead of `year_config: YearConfig`, take:
```python
def fetch_category_market_data(
    category: OscarCategory,
    ceremony_year: int,
    nominations_date: date,
    ceremony_date: date,
    market_data_dir: Path,
    fetch_hourly: bool = False,
    default_spread: float = 0.03,
    min_trades_for_spread: int = 10,
) -> CategoryMarketData | None:
```

Spread estimation defaults (`DEFAULT_SPREAD`, `MIN_TRADES_FOR_SPREAD`) become
**function parameters with defaults**. The library provides sensible defaults;
callers can override. No module-level constants in `trading/`.

**`build_nominee_match`** — instead of mixed params, take:
```python
def build_nominee_match(
    category: OscarCategory,
    model_names: list[str],
    market: OscarMarket,
    snapshot_names: list[str],
    datasets_dir: Path,
    ceremony_year: int,
    ceremony_number: int,
) -> dict[str, str] | None:
```

### P12: Tests

New test files to cover the extracted code:

| Test file | Covers | Key cases |
|-----------|--------|-----------|
| `tests/trading/test_backtest_grid_config.py` | `BacktestGridConfig`, `to_backtest_config()` | Label generation, config expansion with various bankroll/spread values, round-trip field mapping |
| `tests/trading/test_oscar_data.py` | Prediction loading, name matching, market data | `renormalize_predictions`, `build_market_prices`, `CategoryMarketData` construction |
| `tests/trading/test_portfolio_simulation.py` | MC sampling, CVaR, metrics | Deterministic seed tests, edge cases (single category, zero probs), CVaR math |

---

## Proposed File Layout

```
trading/
├── __init__.py                (existing — update exports)
├── schema.py                  (existing — no changes)
├── backtest.py                (existing — no changes)
├── backtest_grid_config.py    (NEW — BacktestGridConfig + to_backtest_config)
├── oscar_data.py              (NEW — data loading, name matching, CategoryMarketData)
├── portfolio_simulation.py    (NEW — MC sampling, CVaR, metrics)
├── edge.py                    (existing)
├── kelly.py                   (existing)
├── signals.py                 (existing)
├── portfolio.py               (existing)
├── kalshi_client.py           (existing)
├── oscar_market.py            (existing)
├── price_utils.py             (existing)
├── name_matching.py           (existing)
├── temporal_model.py          (existing)
├── inspect_trade.py           (existing)
└── market_data/               (existing)
```

New test files:
```
tests/trading/
├── test_backtest_grid_config.py   (NEW)
├── test_oscar_data.py             (NEW)
├── test_portfolio_simulation.py   (NEW)
```

---

## Execution Plan

### Phase 1: Schema extensions (no behavior change)

1. **Add `OscarCategory.slug` property** in `data/schema.py`. Define slugs for
   all 20 categories. Add `from_slug()` classmethod. Replace all
   `CATEGORY_SLUGS[c]` with `c.slug` across codebase. Remove `CATEGORY_SLUGS`.

2. **Add `ModelType.from_short_name()` classmethod** in
   `modeling/models/types.py`. Replace all `SHORT_NAME_TO_MODEL` / reverse-dict
   usage.

3. **Eliminate `MODEL_SHORT_NAMES` dict** — replace all
   `MODEL_SHORT_NAMES[m]` with `m.short_name`.

### Phase 2: BacktestGridConfig extraction

4. **Create `trading/backtest_grid_config.py`** — move `BacktestGridConfig` from
   d20260220/run_backtests.py. Add `to_backtest_config()` method.

5. **Update all importers** — d20260220, d20260224, d20260225 now import
   `BacktestGridConfig` from `trading/`. Replace 20-line config construction
   blocks with `cfg.to_backtest_config(bankroll=..., mean_spread=...)`.

6. **Write `test_backtest_grid_config.py`**.

### Phase 3: Data loading consolidation

7. **Create `trading/oscar_data.py`** — merge year-parameterized functions from
   d20260225/data_loading.py + pure transforms from d20260220/data_prep.py +
   shared model-loading helpers (`remap_to_person_names`,
   `load_individual_model_predictions`, `load_ensemble_model_predictions`) +
   `build_entry_moment` + cleaned-up `CategoryMarketData`.

8. **Update all importers** — d20260220, d20260224, d20260225 now import from
   `trading/oscar_data`. Delete d20260225/data_loading.py. Slim down
   d20260220/data_prep.py to only its non-shared functions.

9. **Write `test_oscar_data.py`**.

### Phase 4: MC extraction

10. **Create `trading/portfolio_simulation.py`** — extract `CategoryScenario`,
    `sample_portfolio_pnl`, `compute_cvar`, `compute_portfolio_mc_metrics`.

11. **Update `scenario_scoring.py` and `generate_report.py`** to use shared
    sampling kernel.

12. **Write `test_portfolio_simulation.py`**.

### Phase 5: Cleanup & naming

13. **Simplify `YearConfig`** — remove stored domain fields that duplicate
    existing registries. Keep storage paths only.

14. **Simplify d20260224's `__init__.py`** — use `CALENDARS` directly, replace
    `_TODAY` with function parameter.

15. **Apply naming conventions** (P10) across touched files.

16. **Delete dead code** — `MODEL_SHORT_NAMES`, `SHORT_NAME_TO_MODEL`,
    `CATEGORY_SLUGS`, d20260225/data_loading.py, orphaned helpers.

17. **Run `make all`** — format, lint, typecheck, test. Fix any breakage.

---

## Decisions Made

These were discussed and resolved during planning:

| Question | Decision | Rationale |
|----------|----------|-----------|
| `generate_trading_configs` location | Stays in one-off | Grid values (kelly fractions, edge thresholds) are experiment-specific |
| `build_entry_moment` location | Move to `oscar_data.py` | 77 lines, Oscar-specific but oscar_data is already Oscar-domain code |
| `CategoryMarketData` with DataFrames | Keep DataFrames, drop dead fields | Short-lived in-memory bundle, never serialized. Drop `trades_df`, `ticker_to_kalshi`, `spread_report` |
| `oscar_data.py` naming | Keep `oscar_data.py` | Broad but accurate — covers predictions, market data, name matching |
| New `CeremonyConfig` model | **Don't create** | All data already in `CALENDARS`, `WINNERS_BY_YEAR`, `OSCAR_MARKETS` |
| `CATEGORIES_BY_YEAR` location | Stays in one-off | Experiment-specific (subset of marketed categories); zero external importers |
| `ExperimentPaths` model | **Don't create** | Just pass individual `Path` params; not enough to justify a model |
| Spread estimation defaults | Function params with defaults | `default_spread=0.03`, `min_trades_for_spread=10` as params, not module constants |
| BacktestGridConfig vs BacktestConfig merge | Keep separate | Different roles (grid point vs engine config), `.label` heavily used in analysis CSVs |
| MC file name | `portfolio_simulation.py` | Covers both MC sampling and portfolio-level metrics, not just Monte Carlo |
| `.slug` coverage | All 20 categories | Mechanical `name.lower().replace(" ", "_")` transform, no special-casing |

---

## Open Questions

1. **`run_category_buy_hold` unification (I9)?** The two versions are ~80%
   identical but differ in settlement handling (known winner vs scenario matrix),
   output schema, and live-mode logic. Options: (a) extract the shared skeleton
   into a higher-order function with settlement as a callback, (b) extract only
   the shared sub-steps (data loading, moment building) and let each one-off
   orchestrate, (c) defer — the shared sub-steps from P6/P7 already eliminate
   most duplication. **Leaning toward (c)** — once oscar_data.py has the shared
   helpers, the remaining orchestration is thin enough to keep separate.

2. **d20260220's `data_prep.py` post-refactor shape?** After moving shared
   functions to `oscar_data.py`, what remains? Likely: `build_trading_moments`,
   `build_model_source`, `build_ensemble_source`, `get_snapshot_dates`,
   `get_trading_dates` — all specific to the rebalancing backtest. Should
   `data_prep.py` start importing from `oscar_data.py` for the moved functions,
   or should it be updated to call them directly? The rebalancing backtest hasn't
   been year-parameterized — we might not want to refactor it now.

3. **Scope creep guard — should d20260220 be fully updated?** d20260220 is the
   oldest one-off and the current de facto library. Fully updating it means
   touching its 950-line `data_prep.py`, 755-line `run_backtests.py`, and
   multiple analysis scripts. Alternative: focus on d20260224 + d20260225 as
   primary consumers of the new `trading/` modules, and leave d20260220 as a
   legacy provider that still exports its own functions for its own use. Other
   one-offs stop importing from it.
