# Plan: Post-Backtest Improvements & Refactors

**Date:** 2025-02-23
**Branch:** `feature/backtest-strategies`
**Status:** Planning — awaiting user review before implementation

---

## Overview

After completing the calendar fix (PLAN_0222) and multi-category backtests,
this plan covers the next wave of improvements: documentation fixes, code
refactors, new capabilities, and deeper design brainstorms.

Items are organized into four phases by dependency and risk:
- **Phase 1** — Surgical documentation/docstring fixes (zero risk)
- **Phase 2** — Targeted refactors (low risk, isolated changes)
- **Phase 3** — New capabilities (medium risk, new code)
- **Phase 4** — Design decisions requiring deeper brainstorm

---

## Phase 1: Documentation & Docstring Fixes

### 1.1 Fix BUY NO example in edge.py

**File:** `trading/edge.py` lines 88–103

**Problem:** The current BUY NO example uses `model_prob=0.80` and describes
it as "prob of NO winning = 1 - 0.20", making it look like
`1 - model_prob == execution_price_cents` (an equality that doesn't generally
hold). The example is also abstract — it doesn't name a specific nominee.

**Fix:**
- Use a concrete Brutalist scenario (e.g., Brutalist at 20¢ YES, model says
  12% chance of winning)
- Make model_prob and execution_price_cents clearly different values so the
  reader can't mistake one for a function of the other
- Example sketch:
  ```
  BUY NO scenario: The Brutalist has a 20% market price (YES@20¢).
  Our model gives it 12% chance of winning, so we want to BUY NO.
  - direction = NO
  - model_prob = 0.88    (prob that NO wins = 1 - 0.12)
  - execution_price_cents = 80  (NO ask = 100 - 20)
  - net_edge = 0.88 - 0.80 = +0.08  (8% edge)
  ```

**Scope:** ~10 lines in one file. Tests: none needed (docstring only).

### 1.2 Fix kelly.py docstrings

**File:** `trading/kelly.py`

Two fixes:

**(a)** `_expected_log_growth` docstring (lines 244–258): Missing the critical
point that "YES contracts on outcome j≠k expire worthless when outcome k wins".
Add a sentence like: "When outcome k wins, all YES positions on j≠k expire
worthless (lose full cost). The NO position on k also expires worthless."

**(b)** "No bet-on outcome wins" scenario in multi_outcome_kelly: The docstring
mentions `prob_none` but doesn't explain what happens to the portfolio. Add:
"If none of the bet-on outcomes wins (probability = `prob_none`), every YES
position expires worthless and every NO position pays out $1. Portfolio value
= cash + sum of NO positions × $1 — all YES positions are total losses."

**Scope:** ~15 lines across two locations. Tests: none needed.

### 1.3 Add `description=` to OutcomeAllocation.model_prob

**File:** `trading/kelly.py` lines ~91–157

**Problem:** `model_prob: float` has no `Field(description=...)` explaining
that it's direction-dependent (YES = P(outcome wins), NO = P(outcome loses)).

**Fix:** Add `Field(description=...)`:
```python
model_prob: float = Field(
    ...,
    description=(
        "Model probability in the direction of the position. "
        "YES: P(outcome wins). NO: P(outcome loses) = 1 - P(outcome wins)."
    ),
)
```

**Scope:** 1 field change.

### 1.4 Add `git rev-parse HEAD` to train_models.sh

**File:** `one_offs/d20260220_backtest_strategies/train_models.sh`

**Problem:** Experiment reproducibility — no record of which code version
produced the models.

**Fix:** Add near the top of the script:
```bash
echo "Git commit: $(git rev-parse HEAD)"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Dirty: $(git diff --shortstat)"
```

**Scope:** 3 lines.

---

## Phase 2: Targeted Refactors

### 2.1 Refactor duplicated availability filtering in data_loader.py

**Files:** `modeling/data_loader.py`

**Problem:** `prepare_features()` (lines 210–242) and
`filter_feature_set_by_availability()` (lines 316–375) both contain
near-identical availability-filtering logic: loop over feature names, look up
in `FEATURE_REGISTRY`, resolve `available_event` via calendar, partition into
available/unavailable.

**Approach:** Extract a shared helper:
```python
def _partition_features_by_availability(
    feature_names: list[str],
    as_of_date: date,
    calendar: AwardsCalendar,
) -> tuple[list[str], list[str]]:
    """Partition feature names into (available, unavailable) as of a date."""
```

Then `prepare_features()` calls this helper (replacing its inline loop), and
`filter_feature_set_by_availability()` calls it too (replacing its loop).

**Scope:** ~30 lines extracted, two call sites simplified. Tests: existing
tests cover both paths.

### 2.2 Light split of run_category_backtest

**File:** `one_offs/d20260220_backtest_strategies/run_backtests.py` lines 447–777

**Problem:** `run_category_backtest()` is ~200 lines handling market data
fetching, spread estimation, model loading, name matching, accuracy eval,
backtesting grid execution, and result aggregation. Hard to navigate.

**Approach:** Split into 3-4 functions by phase:
1. `_fetch_market_data(category, inventory) → (daily_prices_df, trades_df, spread_report)`
2. `_load_and_match_model(category, model_type, ...) → MatchedModel` (preds, nominee_map, winner)
3. `_run_backtest_grid(matched_model, prices, configs) → list[pnl_row]`
4. `run_category_backtest()` becomes a thin orchestrator calling these

Keep functions in the same file (one-off, not library code). Goal is readability,
not reuse.

**Scope:** Rearrange ~200 lines into 4 functions. Behavior unchanged. No new tests.

### 2.3 @save_plot decorator for analyze.py

**File:** `one_offs/d20260220_backtest_strategies/analyze.py`

**Problem:** 13+ `plot_*` functions all repeat:
```python
fig.tight_layout()
fig.savefig(PLOTS_DIR / f"name.png")
plt.close(fig)
print(f"  Saved name.png")
```

**Approach:** Create a decorator at the top of analyze.py:
```python
def save_plot(filename: str):
    """Decorator: tight_layout + save + close + print."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # func should return fig or (fig, axes)
            fig = result[0] if isinstance(result, tuple) else result
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {filename}")
            return result
        return wrapper
    return decorator
```

**Alternative:** Simpler context-manager style:
```python
@contextmanager
def save_figure(filename: str):
    fig = plt.gcf()
    yield fig
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename)
    plt.close(fig)
    print(f"  Saved {filename}")
```

**Recommendation:** Start with the simple functions — many `plot_*` functions
create their own `fig, ax = plt.subplots(...)` so a decorator that wraps the
return value is cleaner. Apply to 3-4 functions first, then roll out.

**Scope:** ~15 lines for decorator, ~4 lines removed per plot function × 13
functions ≈ 50 lines reduced.

### 2.4 Change _SignalContext from dataclass to Pydantic

**File:** `trading/signals.py` line 421

**Problem:** `_SignalContext` is a `@dataclass` while everything else in
`trading/` uses Pydantic `BaseModel`. Inconsistency.

**Fix:** Change `@dataclass` to `BaseModel` with `model_config = {"extra": "forbid"}`.
Since it's a private class never serialized, this is purely for consistency.
The fields are already typed; no validation changes needed.

**Scope:** ~5 lines changed.

---

## Phase 3: New Capabilities

### 3.1 Trade inspection utility

**New file:** `trading/inspect_trade.py`

**Purpose:** Given a backtest result and a specific (outcome, date) pair,
produce a detailed narrative explaining *why* the engine decided to BUY/SELL/HOLD.
Traces through: model_prob → edge calculation → Kelly sizing → signal → fill.

**Design:**
```python
def inspect_trade(
    result: BacktestResult,
    outcome: str,
    date: str,
) -> TradeInspection:
    """Explain a single trade decision from a completed backtest."""
    ...

class TradeInspection(BaseModel):
    outcome: str
    date: str
    model_prob: float
    market_price_cents: float
    direction: PositionDirection
    edge: float
    kelly_fraction: float
    target_contracts: int
    current_contracts: int
    action: TradeAction
    reason: str
    # Plus formatted narrative
    def explain(self) -> str: ...
```

**Implementation approach:** The backtest `trade_log` (list of `Fill`) and
`snapshots` already capture positions and fills. The inspection utility
reconstructs the signal generation step for that specific date by calling
`generate_signals()` with the same inputs (predictions, prices, config) and
extracting the signal for the requested outcome.

**Scope:** ~150 lines new file + CLI wrapper. Medium effort.

### 3.2 Tests for new trading/ files

**Files to test:** `price_utils.py` (82 lines), `name_matching.py` (235 lines),
`market_discovery.py`

**Approach:** Add `tests/trading/test_price_utils.py`, `tests/trading/test_name_matching.py`.
Focus on:
- `price_utils`: `get_market_prices_on_date()` with fallback behavior
- `name_matching`: `normalize_name()`, `match_nominees()` with edge cases
  (accents, "Written by" prefix, partial matches)

**Scope:** ~100 lines per test file. Low priority — these are simple utility
functions.

### 3.3 Buy YES/NO/ALL ablation in backtests

**Problem:** Current backtests only trade `TradingSide.YES`. The sell-side
(BUY NO) alpha was the motivation for the entire sell-side refactor.

**Fix:** Add `TradingSide.BOTH` and `TradingSide.NO` to the
`generate_trading_configs()` grid in `run_backtests.py`, at least for the
best-known configs:
```python
# YES/NO/BOTH ablation at 2-3 promising configs
for ts in [TradingSide.YES, TradingSide.NO, TradingSide.BOTH]:
    configs.append({
        "fee_type": FeeType.MAKER,
        "kelly_fraction": 0.10,
        "buy_edge_threshold": 0.10,
        "min_price_cents": 0,
        "kelly_mode": KellyMode.MULTI_OUTCOME,
        "bankroll_mode": "fixed",
        "trading_side": ts,
    })
```

**Scope:** ~10 lines in config grid + ensure `trading_side` is plumbed through
to `TradingConfig` (should already work since sell-side refactor).

---

## Phase 4: Design Brainstorms (Discussion Required)

### 4.1 Ceremony-Year Data Organization in oscar_market.py

#### Current State

`oscar_market.py` has three layers of hardcoded ceremony-year data:

1. **Event tickers** (lines 51–62): `OSCAR_EVENT_TICKERS: dict[OscarCategory, str]`
   — maps category → Kalshi event ticker for 2026 ceremony only.

2. **Series tickers** (lines 67–78): `OSCAR_SERIES_TICKERS: dict[OscarCategory, str]`
   — year-independent, used to discover events for any year.

3. **Nominee tickers** (lines 84–133): `BP_NOMINEE_TICKERS` (2026),
   `BP_2025_NOMINEE_TICKERS` (2025) — hardcoded per-year.
   Also `BP_2025_WINNER = "Anora"`.

Adding a new ceremony year requires:
- New `BP_20XX_NOMINEE_TICKERS` dict for each category
- New convenience factory `best_picture_20XX()`
- New `OscarMarketMetadata` constant

This doesn't scale to multi-year backtests or multiple categories × years.

#### Approach A: JSON Config Files (recommended)

Store per-ceremony data in JSON files under `trading/market_data/`:

```
trading/market_data/
├── 2025/
│   ├── best_picture.json    # {event_ticker, nominee_tickers, winner}
│   ├── directing.json
│   └── ...
├── 2026/
│   ├── best_picture.json
│   └── ...
└── series_tickers.json      # year-independent
```

Each JSON file is an `OscarMarketMetadata` + optional `winner` field:
```json
{
  "event_ticker": "KXOSCARPIC-25",
  "nominee_tickers": {"Anora": "KXOSCARPIC-25-A", ...},
  "winner": "Anora"
}
```

**Loader:**
```python
def load_market_metadata(
    category: OscarCategory, ceremony_year: int
) -> OscarMarketMetadata:
    path = MARKET_DATA_DIR / str(ceremony_year) / f"{category_slug}.json"
    return OscarMarketMetadata.model_validate_json(path.read_text())
```

**Pros:**
- Adding a new year = adding JSON files, no Python edits
- JSON files can be auto-generated by `discover_tickers.py`
- Serializable, version-controllable, diffable
- Decouples data from code

**Cons:**
- Extra file I/O at startup
- Need to keep JSON in sync if tickers change (unlikely after nomination)
- Discovery script needs to write the JSON format

#### Approach B: Python Registry with Year Parameter

Keep data in Python but organize as a nested dict:

```python
NOMINEE_TICKERS: dict[int, dict[OscarCategory, dict[str, str]]] = {
    2025: {
        OscarCategory.BEST_PICTURE: {"Anora": "KXOSCARPIC-25-A", ...},
        ...
    },
    2026: {
        OscarCategory.BEST_PICTURE: {"Sinners": "KXOSCARPIC-26-SIN", ...},
        ...
    },
}
```

**Pros:**
- No file I/O, everything in-memory
- IDE autocomplete and type checking
- Single source of truth (no JSON sync)

**Cons:**
- Adding a year still requires editing Python
- File grows linearly with years × categories
- Hard to auto-generate from discovery script

#### Approach C: API Discovery at Runtime

Use `discover_tickers.py` to query the Kalshi API and cache results:

```python
def get_market_metadata(
    category: OscarCategory, ceremony_year: int
) -> OscarMarketMetadata:
    cache_key = f"market_metadata_{ceremony_year}_{category.value}"
    # Check diskcache first, then hit API
    ...
```

**Pros:**
- Always up-to-date
- No manual data entry at all
- Works for any future year automatically

**Cons:**
- Requires API access (fails offline)
- API responses may change format
- Slower initial load
- Harder to reproduce (API state is temporal)

#### Recommendation

**Approach A (JSON configs)** is the best fit because:
1. `discover_tickers.py` already exists and fetches this data — just need to
   save as JSON instead of printing
2. JSON files become the experiment's ground truth (reproducible)
3. Aligns with the project's "configs go in storage" philosophy
4. The `market_discovery.py` module already has `load_ticker_inventory()` that
   reads JSON — this extends the same pattern

**Migration plan:**
1. Create `trading/market_data/` directory with JSON files for 2025 + 2026
2. Add `load_market_metadata()` function
3. Deprecate `BP_NOMINEE_TICKERS`, `BP_2025_NOMINEE_TICKERS` etc.
4. Update `OscarMarket` convenience factories to use JSON loader
5. Update `run_backtests.py` to use the new loader
6. Keep `OSCAR_SERIES_TICKERS` in Python (year-independent, rarely changes)

### 4.2 Cross-Category Portfolio Allocation

#### Current State

Each category gets an independent $1,000 bankroll. Kelly sizing happens
*within* each category's multi-outcome market (correctly handling the
exactly-one-winner constraint). But across categories, there's no coordination:
if Kelly says bet $200 on Best Picture and $200 on Directing, total exposure is
$400 across a $1,000 bankroll — but the engine doesn't know this.

This matters because:
- **Correlation:** Oscar categories are correlated (same films appear in
  multiple categories, precursor wins are shared signals). Over-betting
  correlated positions concentrates risk.
- **Bankroll fungibility:** In practice, you have one Kalshi account with one
  balance, not 9 separate bankrolls.

#### Approach A: Hierarchical Kelly (recommended for v1)

Two-level allocation:

1. **Category-level budget:** Divide total bankroll across categories using a
   simple heuristic (equal weight, or weighted by model confidence / number of
   categories with edge).

2. **Within-category Kelly:** Run `multi_outcome_kelly()` with the
   category-level budget as `bankroll`.

```python
# Pseudo-code
total_bankroll = 9000  # or whatever
n_categories_with_edge = count(categories where any outcome has edge > threshold)
category_budget = total_bankroll / n_categories_with_edge

for category in categories:
    kelly_config = KellyConfig(bankroll=category_budget, ...)
    signals = generate_signals(predictions, prices, config=...)
```

**Pros:**
- Simplest implementation — just change bankroll per category
- No new math needed
- Respects the fact that within-category Kelly already handles correlation
  (exactly-one-winner)
- Easy to ablate: equal-weight vs proportional

**Cons:**
- Ignores cross-category correlation entirely
- Equal weight over-allocates to categories with low edge
- Doesn't account for the fact that Best Picture correlated with Directing

#### Approach B: Risk Parity / Edge-Weighted Allocation

Weight category budgets by some measure of opportunity:

```python
# Sum of absolute edges across outcomes for each category
category_edge_mass = {cat: sum(abs(edge) for edge in edges) for cat, edges in ...}
total_edge = sum(category_edge_mass.values())
category_budget = {cat: total_bankroll * mass / total_edge for cat, mass in category_edge_mass.items()}
```

**Pros:**
- Concentrates capital where the model has stronger opinions
- Simple mental model: more edge → more money

**Cons:**
- Edge mass can be dominated by low-probability outcomes (many small edges)
- Still ignores cross-category correlation
- Overfits to model confidence (which may be miscalibrated)

#### Approach C: Global Multi-Outcome Kelly

Treat ALL outcomes across ALL categories as one big multi-outcome problem.
This would require extending `multi_outcome_kelly()` to handle multiple
independent "exactly-one-winner" constraints (one per category).

Mathematically: maximize expected log wealth where the probability space is
the product of per-category outcome spaces. If 9 categories each have ~10
outcomes, that's 10^9 joint states — but the independence between categories
(conditional on precursor results) makes this decomposable.

**Pros:**
- Theoretically optimal allocation
- Naturally handles correlation if we model it

**Cons:**
- Complex optimization: either enumerate joint states (exponential) or
  exploit independence structure (requires new math)
- Correlation modeling between categories is its own research project
- Diminishing returns: Kelly fraction is already 0.10 (conservative), so
  cross-category effects are second-order

#### Approach D: Simple Exposure Cap

Don't change Kelly at all — just add a global exposure cap:

```python
# After generating all signals across all categories
total_outlay = sum(signal.outlay_dollars for signal in all_signals)
if total_outlay > max_total_exposure:
    scale = max_total_exposure / total_outlay
    # Scale all positions down proportionally
```

**Pros:**
- Trivial to implement
- Prevents catastrophic over-exposure
- Compatible with any per-category strategy

**Cons:**
- Crude: scales everything equally regardless of edge
- Doesn't allocate to best opportunities

#### Recommendation

**Start with Approach A (hierarchical)** for the backtest, with total bankroll =
$1,000 × N_categories (so each category effectively gets $1,000 — same as now
but with the explicit framing). Then add Approach D (exposure cap) as a safety
measure.

For a more sophisticated v2, Approach B (edge-weighted) is the natural next step.
Approach C (global multi-outcome) is theoretically interesting but overkill for
the current 2025 ceremony backtest.

**Implementation:**
1. Add `portfolio_bankroll: float` and `n_active_categories: int` to the
   backtest runner
2. Compute per-category budget: `portfolio_bankroll / n_active_categories`
3. Pass as `KellyConfig.bankroll`
4. Add reporting: total portfolio P&L, total exposure, cross-category metrics
5. Ablation: equal-weight vs edge-weighted in the backtest grid

### 4.3 Config Management

#### Current State

Feature configurations have a dual-source problem:

1. **Python source of truth:** `FEATURE_REGISTRY` in `feature_engineering.py`
   defines all features with their types, transforms, availability events,
   descriptions. This is where features *exist*.

2. **JSON configs:** `modeling/configs/features/*.json` define feature *sets*
   — which features to use for a specific (category, model_type) combination.
   These are just lists of feature names + model_type.

The flow is:
```
JSON config → FeatureSet(name, features=[...], model_type)
                  ↓
FEATURE_REGISTRY[name] → FeatureDefinition(name, transform, available_event, ...)
                  ↓
data_loader.prepare_features() → filtered DataFrame
```

The problem isn't that there are two sources — that's by design (registry =
what exists, config = what to use). The pain points are:

**(a) JSON configs are hand-maintained.** Adding a new feature to the registry
doesn't automatically add it to configs. Easy to forget.

**(b) No validation that JSON feature names exist in the registry** until
runtime (`resolve_features()` raises `ValueError`).

**(c) Category-specific configs are repetitive.** 18 JSON files (9 categories ×
2 model types) that are mostly identical — same feature list, just different
`model_type` and category-specific features.

#### Approach A: Python Generates JSON (recommended)

Add a script that generates JSON configs from Python:

```python
# generate_feature_configs.py
def generate_configs():
    for category in MODELED_CATEGORIES:
        for model_type in [ModelType.LOGISTIC_REGRESSION, ModelType.GRADIENT_BOOSTING]:
            features = get_features_for_category(category, model_type)
            config = FeatureSet(
                name=f"{category_slug}_{model_type_slug}_full",
                description=f"All features for {category.name} ({model_type.value})",
                features=[f.name for f in features],
                model_type=model_type,
            )
            path = CONFIGS_DIR / f"{category_slug}_{model_type_slug}_full.json"
            path.write_text(config.model_dump_json(indent=2))
```

**Pros:**
- Python remains the single source of truth
- JSON configs are always in sync with the registry
- Can generate category-specific configs programmatically (e.g., exclude
  ANIMATED_FEATURES for non-animated categories)
- Validates at generation time

**Cons:**
- Extra step in workflow (must regenerate after adding features)
- JSON files in git are generated artifacts (some don't like committing these)

Actually, `generate_feature_ablation_configs.py` and `generate_tuning_configs.py`
already exist! This pattern is established. We just need a simpler version for
the "full" configs.

#### Approach B: Eliminate JSON, Use Python Directly

Instead of JSON configs, define feature sets in Python:

```python
# modeling/feature_sets.py
FEATURE_SETS: dict[tuple[OscarCategory, ModelType], FeatureSet] = {
    (OscarCategory.BEST_PICTURE, ModelType.LOGISTIC_REGRESSION): FeatureSet(
        name="best_picture_lr_full",
        features=get_feature_names(ModelType.LR) + BP_INTERACTION_NAMES,
        ...
    ),
    ...
}
```

CLIs would accept `--category best_picture --model-type lr` instead of
`--feature-config path/to/json`.

**Pros:**
- No JSON files to maintain
- No sync problem
- Simpler

**Cons:**
- Loses the ability to pass arbitrary feature configs via CLI (useful for
  ablation experiments where you want to test non-standard feature sets)
- build_model.py currently generates intermediate feature configs
  (3_selected_features.json) that other scripts consume — this pattern
  would need rethinking

#### Approach C: Schema Validation at Import Time

Keep JSON configs but validate them at import:

```python
# In data_loader.py or a new validation module
def validate_feature_config(config_path: Path) -> FeatureSet:
    fs = FeatureSet.model_validate_json(config_path.read_text())
    unknown = set(fs.features) - set(FEATURE_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown features in {config_path}: {unknown}")
    return fs
```

And add a `make validate-configs` target that runs this on all JSON files.

**Pros:**
- Catches errors early (CI, not runtime)
- Minimal change to existing workflow
- JSON configs remain flexible

**Cons:**
- Doesn't prevent drift — just catches it earlier
- Still need to manually update JSON when adding features

#### Recommendation

**Approach A (Python generates JSON) + Approach C (validation):**

1. Add `generate_full_feature_configs.py` that generates the 18 "full" configs
   from the registry (using `get_features_for_model()` + category-specific lists)
2. Add `validate_feature_configs.py` (or a flag on the generator) that checks
   all JSON files against the registry
3. Add `make generate-configs` and `make validate-configs` targets
4. Keep hand-edited JSON for ablation experiments (these are one-offs, not
   canonical)

This extends the existing pattern (`generate_feature_ablation_configs.py`)
without disrupting the CLI-based workflow.

---

## Implementation Order

| # | Item | Phase | Effort | Dependencies |
|---|------|-------|--------|--------------|
| 1 | Fix BUY NO example (edge.py) | 1 | S | — |
| 2 | Fix kelly.py docstrings | 1 | S | — |
| 3 | Add OutcomeAllocation description | 1 | S | — |
| 4 | Add git hash to train_models.sh | 1 | S | — |
| 5 | Refactor availability filtering | 2 | M | — |
| 6 | Change _SignalContext to Pydantic | 2 | S | — |
| 7 | @save_plot decorator | 2 | M | — |
| 8 | Light split run_category_backtest | 2 | M | — |
| 9 | Buy YES/NO/ALL ablation | 3 | S | sell-side refactor (done) |
| 10 | Trade inspection utility | 3 | L | — |
| 11 | Tests for price_utils, name_matching | 3 | M | — |
| 12 | Ceremony-year data → JSON | 4 | L | discovery script exists |
| 13 | Cross-category portfolio allocation | 4 | L | items 8, 9 |
| 14 | Config generation script | 4 | M | — |

**Suggested batching for commits:**

- **Commit A** (items 1–4): Docstring/documentation fixes
- **Commit B** (items 5–6): Availability refactor + SignalContext
- **Commit C** (items 7–8): analyze.py + run_backtests.py refactors
- **Commit D** (item 9): Trading side ablation
- **Commit E** (item 10): Trade inspection utility
- **Commit F** (item 11): New tests
- **Commit G** (item 12): Ceremony-year data reorganization
- **Commit H** (items 13–14): Portfolio allocation + config generation

---

## Decisions Deferred (Not in This Plan)

- **feature_engineering.py split:** User decision — "keep as is". Already
  well-organized with section headers.
- **Pipeline boilerplate dedup (train_predict vs evaluate_cv):** User decision
  — "ok to duplicate". Separate CLIs are intentional.
- **Modeling tests:** User decision — "no coverage is ok" for now.
- **Global multi-outcome Kelly (Approach C in §4.2):** Theoretically
  interesting but overkill for current scope. Revisit if cross-category
  correlation proves important.
