# Multi-Category Backtest Strategies

**Storage:** `storage/d20260220_backtest_strategies/`

Comprehensive backtesting across categories, models, and trading strategies.
Two goals: (1) understand whether classification performance translates to
trading edge, and (2) build confidence in model/trading configuration before
deploying real capital.

## Motivation

All prior backtesting ([d20260214_trade_signal_backtest](../d20260214_trade_signal_backtest/),
[d20260214_trade_signal_ablation](../d20260214_trade_signal_ablation/)) was limited to:
- **One category** (Best Picture only)
- **One year** (2026 ceremony season, outcome unknown)
- **One trading style** (temporal simulation with 10 snapshot dates)

We need broader evidence before committing to a strategy. This experiment:
1. Tests on **2025 ceremony** (known outcomes) across **all 9 modeled categories**
2. Trains **4 model types** (LR, clogit, GBT, cal-SGBT) at each temporal
   snapshot — evaluating both model accuracy and trading performance per model
3. Runs **daily buy-once backtests** with real Kalshi prices post-nominations
4. Uses **as-of-date feature gating** for every snapshot — model only sees features
   available at trade time
5. Answers: **does better classification (lower Brier) produce better trading
   returns?** Model comparison is a first-class analysis dimension.

**Scope:** 2025 backtest only (Phase 1–2). 2026 temporal simulation and
portfolio-level analysis deferred to follow-up.

---

## Kalshi Oscar Market Inventory

Discovered via API on 2026-02-20. Full raw inventory saved to
`storage/d20260220_backtest_strategies/ticker_inventory.json`.
Discovery script: `discover_tickers.py` in this directory.

Kalshi tickers changed naming convention between 2022–2023 (`OSCARPIC-22`)
and 2025–2026 (`KXOSCARPIC-25`).

### Categories × Years — All 9 Modeled Categories Have Markets

| Category | Series Ticker | 22 | 23 | 24 | 25 | 26 |
|----------|--------------|:--:|:--:|:--:|:--:|:--:|
| Best Picture | KXOSCARPIC | 10 | 10 | 14 | 17 | 19 |
| Best Director | KXOSCARDIR | 5 | 5 | 12 | 14 | 16 |
| Best Actor | KXOSCARACTO | 5 | 5 | 10 | 10 | 16 |
| Best Actress | KXOSCARACTR | 5 | 5 | 10 | 13 | 16 |
| Best Supporting Actor | KXOSCARSUPACTO | 5 | 5 | 10 | 13 | 17 |
| Best Supporting Actress | KXOSCARSUPACTR | 5 | 5 | 11 | 13 | 17 |
| Best Original Screenplay | KXOSCARSPLAY | 5 | 5 | 10 | 11 | 17 |
| Best Animated Feature | KXOSCARANIMATED | — | — | 5 | 9 | 7 |
| Best Cinematography | KXOSCARCINE | — | — | — | 10 | 16 |

Nominee counts include pre-nomination speculative markets. We use **only
official nominees** since these are post-nomination settled markets — speculative
picks have 0% probability. Filter to nominees present in both model predictions
and Kalshi markets, renormalize model probabilities over the matched set.

### Event Tickers by Category × Year

| Category | -22 | -23 | -24 | -25 | -26 |
|----------|-----|-----|-----|-----|-----|
| Best Picture | OSCARPIC-22 | OSCARPIC-23 | OSCARPIC-24 | KXOSCARPIC-25 | KXOSCARPIC-26 |
| Director | OSCARDIR-22 | OSCARDIR-23 | OSCARDIR-24 | KXOSCARDIR-25 | KXOSCARDIR-26 |
| Actor | OSCARACTO-22 | OSCARACTO-23 | OSCARACTO-24 | KXOSCARACTO-25 | KXOSCARACTO-26 |
| Actress | OSCARACTR-22 | OSCARACTR-23 | OSCARACTR-24 | KXOSCARACTR-25 | KXOSCARACTR-26 |
| Supp. Actor | OSCARSUPACTO-22 | OSCARSUPACTO-23 | OSCARSUPACTO-24 | KXOSCARSUPACTO-25 | KXOSCARSUPACTO-26 |
| Supp. Actress | OSCARSUPACTR-22 | OSCARSUPACTR-23 | OSCARSUPACTR-24 | KXOSCARSUPACTR-25 | KXOSCARSUPACTR-26 |
| Orig. Screenplay | OSCARSPLAY-22 | OSCARSPLAY-23 | OSCARSPLAY-24 | KXOSCARSPLAY-25 | KXOSCARSPLAY-26 |
| Animated | — | — | OSCARANIMATED-24 | KXOSCARANIMATED-25 | KXOSCARANIMATED-26B |
| Cinematography | — | — | — | KXOSCARCINE-25 | KXOSCARCINE-26 |

### Ceremony Reference

| Ceremony | Year | Suffix | BP Winner | Oscar Date |
|----------|------|:------:|-----------|------------|
| 94 | 2022 | -22 | CODA | 2022-03-27 |
| 95 | 2023 | -23 | Everything Everywhere All at Once | 2023-03-12 |
| 96 | 2024 | -24 | Oppenheimer | 2024-03-10 |
| 97 | 2025 | -25 | Anora | 2025-03-02 |
| 98 | 2026 | -26 | TBD | 2026-03-15 |

---

## Model & Feature Configs

From [d20260220_feature_ablation](../d20260220_feature_ablation/) Phase 3
recommendations:

### Feature pipeline (universal)

- **Feature set**: `lr_full` (for LR/clogit) / `gbt_full` (for GBT/cal-SGBT)
  — all features, let feature selection prune
- **Feature selection**: ON, importance threshold **t=0.90**
- **No max_features cap**
- **VIF filter**: OFF

Feature configs per category live in `modeling/configs/features/`. Copy into
experiment storage at setup time for reproducibility. Feature counts differ
per category (43–57) because each category includes different conditional
feature groups (person features for acting/directing, animated-specific, etc.).

### 4 model types (all categories)

We train **all 4 models for every category** — not pre-selecting by category.
This is a deliberate choice: we want to compare classification performance
(Brier, accuracy) with trading performance (P&L, Sharpe) across model types.
The key question: **does the best classifier also produce the best trading
returns?**

| Model | Param grid size | Key hyperparams |
|-------|:---------------:|-----------------|
| LR | 30 | C x l1_ratio (pure L2 historically wins) |
| Conditional Logit | 30 | alpha x L1_wt (expanded grid: alpha in [0.001-0.1]) |
| GBT | 36 | lr x depth x n_estimators x subsample |
| Cal-SGBT | 60 | lr x depth x n_estimators x subsample x temp |

All grids generated from `generate_tuning_configs.py` at experiment setup
(single source of truth). Full CV grid search for hyperparameter selection
— no fixed hyperparams.

### Reference: feature ablation best configs (for context, not used to pre-select)

| Category | Best Model | Brier | Acc % |
|----------|-----------|-------|-------|
| best_picture | Clogit | 0.052 | 76.9 |
| directing | Clogit | 0.028 | 92.3 |
| actor_leading | Clogit | 0.067 | 80.8 |
| actress_leading | Clogit | 0.011 | 96.2 |
| actor_supporting | Clogit | 0.031 | 92.3 |
| actress_supporting | Clogit | 0.007 | 100.0 |
| original_screenplay | Clogit | 0.018 | 92.3 |
| cinematography | Clogit | 0.015 | 100.0 |
| animated_feature | Cal-SGBT | 0.022 | 95.8 |

*These are LOYO CV results at t=0.90 from Phase 3. Clogit wins 8/9 but we
train all 4 models to test whether classification advantage translates to
trading edge.*

---

## 1. 2025 Historical Backtest (9 Categories, Known Outcomes)

### Goal

Unified backtest combining **temporal model accuracy** and **daily buy-once
trading** across all 9 categories x 4 model types for the 2025 ceremony
(settled, known winners). One model is trained per (model_type, category,
snapshot_date) — evaluating both predictive quality and trading returns.

This answers:
- "How does model accuracy evolve as precursor winners are announced?"
- "Would trading on model vs market disagreement have been profitable?"
- "Which trading config works best across all 9 categories?"
- **"Does the best classifier also produce the best trading returns?"**

### Temporal Model Snapshots (Post-Nomination Only)

Each snapshot corresponds to a precursor winner event after Oscar nominations.
We only model post-nomination since (a) that's when we'd actually trade, and
(b) the nominee set is known. Derived from `CALENDAR_2025`:

| # | As-of Date | Event(s) | New Features |
|:-:|:----------:|----------|--------------|
| 1 | 2025-01-23 | **Oscar nominations** | Oscar nom features (baseline) |
| 2 | 2025-02-07 | Critics Choice winner | CC winner features |
| 3 | 2025-02-08 | DGA + Annie + PGA winners | DGA/Annie/PGA winner features |
| 4 | 2025-02-15 | WGA winner | WGA winner features |
| 5 | 2025-02-16 | BAFTA winner | BAFTA winner features |
| 6 | 2025-02-23 | SAG + ASC winners | SAG/ASC winner features |

6 snapshots x 9 categories x 4 models = **216 model trainings** (~2-3 min each,
~7-10 hours total, parallelizable per model/category).

**Calendar asymmetry note:** Precursor timelines differ across years relative
to Oscar nominations. For 2025, Golden Globe winner (Jan 5) is pre-nomination
so it's baked into snapshot #1, while Critics Choice winner (Feb 7) is
post-nomination and gets its own snapshot. For 2026, both CC and GG winners
are pre-nomination. The `get_post_nomination_snapshot_dates()` function
handles this automatically — analysis should note which precursor events are
testable in each year.

### Model Accuracy at Each Snapshot

For each (model_type, category, snapshot), evaluate:
- Winner rank (1 = model's top pick won)
- Winner probability
- Brier score
- Accuracy (1 if rank=1)

Report as heatmap: rows = snapshots, columns = categories, cells = Brier.
One heatmap per model type. Track how model accuracy improves as precursor
winners are announced.

### Model vs Market Baseline

Two baselines to contextualize model performance:

**(a) "Buy the market favorite" baseline:** Each trading day, buy the nominee
with the highest Kalshi price (the market consensus pick). Settle against known
winner. This shows what a no-model trader gets — pure market consensus returns.
Compare model P&L to this baseline to quantify the value of having a model.

**(b) Model-market divergence tracking:** At each snapshot x trading day,
report both model and market probabilities for every nominee. Track
**divergence** (model prob - market prob) alongside P&L. This separates "the
model picks winners well" from "the model disagrees with the market AND is
right" — the latter is what generates trading edge.

Key metrics per snapshot:
- Model prob vs market prob for the eventual winner
- Mean absolute divergence across nominees
- Correlation between divergence and P&L
- Market-favorite baseline P&L vs model P&L

### Daily Buy-Once Backtest

**Trading period:** Oscar nominations (2025-01-23) through the day before
the ceremony (2025-03-01). ~37 trading days.

For each trading day, the model predictions come from the **most recent
snapshot whose as-of-date <= that day**. So on Feb 10, we use the Feb 8
snapshot (DGA/Annie/PGA winners known). This is exactly what a real trader
would have — predictions based only on information available at that time.

**Buy-once means one purchase per nominee per season.** If the model signals
BUY on Film A on day 1 and still signals BUY on day 5, we do NOT buy more.
One purchase per (nominee, category) — once a position is open, it is held to
settlement. **No sell logic** — all positions held until Oscar ceremony settles.
This prevents over-concentration and simplifies capital tracking.

Procedure for each `(model_type, category, trading_day)`:
1. Look up the active model snapshot for that day
2. Feed single-date predictions + Kalshi close prices into `BacktestEngine`
3. Skip nominees with existing open positions (buy-once constraint)
4. Settle against known winner
5. Record P&L, positions, fees

### Spread Estimation

Estimate spreads from Kalshi trade history using the existing
`estimate_spread_from_trades()` (consecutive buy/sell transition method).
Run per event ticker and report alongside backtest results:
- Median spread by category
- Spread by time-in-season (early vs late)
- Spread by nominee (frontrunner vs longshot)
- Trade count / daily volume per category (liquidity / category viability)

Use estimated per-outcome spreads as `spread_penalties` in the BacktestEngine.

**Max safe position size:** For each nominee market, compute a "max safe
position" based on daily volume (e.g., cap at 10% of trailing-5-day average
volume). Wire through `max_position_per_outcome` in `KellyConfig`. Report
alongside spread stats.

### Nominee Name Matching

Use `normalize_person_name()` from `data/utils.py` as the base normalizer
(handles accent stripping, lowercase, whitespace). For film title matching,
apply similar normalization + a lightweight fuzzy matcher (e.g.,
`rapidfuzz.fuzz.ratio` or `thefuzz`). Strategy:

1. Normalize both sides (model names from `predictions_test.csv`, Kalshi
   names from `custom_strike`)
2. Exact match after normalization
3. Fuzzy match (ratio > 85) for remaining unmatched
4. Manual override dict for known mismatches
5. Exclude "Tie" markets

Validation step: print all matches for human review before running backtests.
Filter to **official nominees only** (exclude pre-nomination speculative
markets). Renormalize model probabilities over the matched set.

### Trading Parameter Grid

| Parameter | Values |
|-----------|--------|
| fee_type | maker (1.75%), taker (7%) |
| kelly_fraction | 0.10, 0.25 |
| buy_edge_threshold | 0.05, 0.10, 0.15 |
| min_price_cents | 0, 5 |
| kelly_mode | independent, multi_outcome |

2 x 2 x 3 x 2 x 2 = **48 configs** per (model_type, category).

Prior ablation (d20260214) found top configs converge to maker + multi_outcome
+ min_price=0, but we keep the full grid since (a) it's fast to evaluate and
(b) we now test across 9 categories where different configs might win.

### Per-Precursor P&L Analysis

Aggregate P&L by individual precursor event, not coarse windows. Each row
shows the incremental value of that precursor's information:

| Event | Active Dates | Snapshot # | Days |
|-------|-------------|:----------:|:----:|
| Post-Oscar-noms | Jan 23 - Feb 6 | 1 | 15 |
| Post-CC winner | Feb 7 | 2 | 1 |
| Post-DGA/Annie/PGA winners | Feb 8 - Feb 14 | 3 | 7 |
| Post-WGA winner | Feb 15 | 4 | 1 |
| Post-BAFTA winner | Feb 16 - Feb 22 | 5 | 7 |
| Post-SAG/ASC winners | Feb 23 - Mar 1 | 6 | 7 |

"Post-DGA P&L" = P&L from all trades made while the DGA snapshot was active
(Feb 8-14), summed across all 9 categories. This measures the incremental
value of that precursor's information for trading.

Also report aggregate P&L views:
- **Fixed daily investment**: equal dollar amount each trading day, noms to ceremony
- **Per-precursor entry**: buy once after each precursor, hold to settlement
- **Cumulative by window**: running total showing when edge materializes

### Return Decomposition

Decompose net P&L into three sources to understand *where* edge comes from:

1. **Selection alpha**: P&L from correctly picking the winner vs an
   equal-weight-all-nominees baseline (buy equal $ on every nominee each day).
   Measures whether the model's probability ranking adds value.
2. **Timing alpha**: P&L from buying later in the season (when model has more
   precursor info) vs buying everything on nomination day snapshot #1.
   Measures the incremental value of waiting for precursor data.
3. **Sizing alpha**: P&L from Kelly sizing vs equal-weight positions (same
   buy signals, just different position sizes). Measures whether Kelly
   concentration helps or hurts.

This is pure arithmetic on top of existing backtest results — no additional
model training. Report as a subsection of the 2025 analysis.

### Data Quality Validation

Automated checks run before backtesting (fail-fast on data errors):

- Kalshi prices sum to ~100% per event per day (sanity check on market data)
- Model probabilities sum to 100% per snapshot (renormalization worked)
- Every model nominee maps to a Kalshi market (no orphans in either direction)
- No large gaps in price data (detect stale/illiquid markets: flag if >3 days
  with zero trades for any nominee)
- For 2025: verified winner is in the matched nominee set

### Config Selection (Pre-Commitment for 2026)

After running all 48 configs x 4 models across 9 categories on 2025:

1. Rank configs by aggregate P&L across all categories
2. Select the **best (model_type, trading_config) pair** from 2025 results
3. This pair is locked for 2026 — no further optimization on 2026 data
4. Also identify 2-3 runner-up configs for robustness comparison

The selection function should be reusable: when expanding to more years, the
same code runs over the expanded year set.

### Classification vs Trading Performance Analysis

A top-level analysis answering: **does the best classifier trade best?**

For each model type, compute:
- Mean Brier score across categories x snapshots (classification quality)
- Mean P&L across categories x configs (trading quality)
- Rank correlation between Brier ranking and P&L ranking
- Scatter plot: x = Brier score, y = P&L, one point per (model, category)

This determines whether we should pick models by Brier (cheap to evaluate via
CV) or need to run full backtests for model selection.

### Scope

9 categories x 6 snapshots x 4 models = **216 model trainings** (~7-10 hrs).
9 categories x 4 models x ~37 trading days x 48 configs = **~64,000 backtest
evals** (seconds total — pure arithmetic).

The model trainings are the bottleneck.

### Key Questions

- How does model Brier score evolve across post-nomination snapshots?
- Where does model vs market divergence generate the most edge?
- Does the model generate positive P&L across categories?
- How sensitive is P&L to buy timing?
- What trading parameters work best across categories?
- Do maker vs taker fees change viability?
- Which categories offer best risk/reward?
- Which precursor events provide the most incremental value?
- **Does the best classifier (by Brier) also produce the best P&L?**

---

## Deferred Work

### Permutation Test (Deferred)

With only 9 categories x 1 year, N is small. Planned permutation test:

1. For each (category, config), hold the model's trades fixed but randomly
   permute which nominee "wins" (sample from the actual nominees)
2. Repeat 1000 times -> null distribution of P&L
3. Compare actual P&L to the null -> p-value

This tests "Could this P&L have happened by chance?" Defer to follow-up
after validating core results. Implementation is straightforward — no
additional model training, just resampling settlement logic.

9 categories x 48 configs x 1000 permutations = **~432,000 permutation evals**
(seconds — pure arithmetic).

### Model Weighting / Market Blend (Deferred)

The prior ablation (d20260214) found GBT + alpha=0.15 market blend was the best
single-category config. We defer blending to a follow-up — first establish
which base model performs best across all 9 categories, then decide what to
blend. The 4-model comparison in this experiment provides the foundation for
informed blending decisions.

### 2026 Temporal Simulation (Deferred)

Same pipeline as Section 1, applied to the current 2026 season. Train temporal
model snapshots at each precursor event date. Since the 2026 ceremony hasn't
happened, results are conditional. Deferred until 2025 backtest validates the
pipeline.

### Portfolio-Level Cross-Category Backtest (Deferred)

Single bankroll across all categories simultaneously. Tests correlation of
bets (same film in multiple categories), capital allocation, portfolio-level
Sharpe. Deferred until per-category results are understood.

### 2026 Trading Recommendation (Deferred)

Concrete buy/sell/hold per nominee based on pre-committed config from 2025.
Regenerated as new precursor data arrives. Deferred until 2025 backtest is
complete.

### Future Expansion (2022-2024)

The pipeline supports easy year expansion. All ticker inventory already
discovered. When adding years, use both unweighted and exponential-decay
(lambda=0.7) weighting for aggregate analysis. Deferred.

---

## Infrastructure

### Market Discovery Module

New file: `trading/market_discovery.py`

```python
def discover_event_markets(event_ticker: str) -> list[KalshiMarket]:
    """Fetch all nominee markets for a given event ticker."""

def build_nominee_ticker_map(event_ticker: str) -> dict[str, str]:
    """Map nominee names to tickers. Excludes 'Tie' markets."""

def build_historical_market(event_ticker: str) -> OscarMarket:
    """Auto-discover tickers and construct OscarMarket for any event."""

def get_active_date_range(event_ticker: str) -> tuple[date, date]:
    """Determine date range with trading activity from candlestick data."""
```

### Nominee Name Matching

New file: `trading/name_matching.py`

```python
def normalize_name(name: str) -> str:
    """Normalize name for matching. Reuses normalize_person_name from data/utils.py
    for accent stripping. Also handles title variations (strip articles, etc.)."""

def match_nominees(
    model_names: list[str],
    kalshi_names: list[str],
    category: OscarCategory,
) -> dict[str, str]:
    """Match model nominee names to Kalshi nominee names.
    Returns {model_name: kalshi_name}. Uses:
    1. Exact match after normalization
    2. Fuzzy match (rapidfuzz, ratio > 85)
    3. Manual overrides
    """

MANUAL_OVERRIDES: dict[tuple[str, int], dict[str, str]] = {}
```

### Historical Event Registry

Extension of `oscar_market.py` — map all (category, ceremony_year) to event ticker:

```python
HISTORICAL_EVENT_TICKERS: dict[tuple[OscarCategory, int], str] = {
    (OscarCategory.BEST_PICTURE, 2022): "OSCARPIC-22",
    (OscarCategory.BEST_PICTURE, 2023): "OSCARPIC-23",
    ...  # All entries from ticker_inventory.json
}
```

### Snapshot Date Derivation

Derive post-nomination snapshot dates programmatically from `AwardsCalendar`:

```python
def get_post_nomination_snapshot_dates(
    calendar: AwardsCalendar,
) -> list[tuple[date, list[str]]]:
    """Derive post-nomination snapshot dates from a calendar.

    Returns list of (as_of_date, [event_labels]) sorted chronologically.
    First snapshot = Oscar nominations date. Subsequent snapshots = each
    precursor winner event that falls after nominations (events on the same
    day grouped into one snapshot).
    """
```

### Config Generation

Copy production configs into experiment storage for reproducibility. A setup
script generates:

1. **Feature configs**: Copy `modeling/configs/features/{category}_{lr,gbt}_full.json`
   into `storage/d20260220_backtest_strategies/configs/features/`
2. **Param grids**: Run `generate_tuning_configs.py` to create all 4 model
   param grids in `storage/d20260220_backtest_strategies/configs/param_grids/`
3. **CV splits**: Copy `leave_one_year_out.json`

This ensures the experiment is self-contained even if source configs change later.

### Using the BacktestEngine

A single buy date is a backtest with one snapshot:

```python
engine = BacktestEngine(config)
result = engine.run(
    predictions_by_date={buy_date_str: model_predictions},
    prices_by_date={buy_date_str: kalshi_close_prices},
    spread_penalties=estimated_spreads,
)
settlement = result.settle(actual_winner)
```

This reuses all signal generation, Kelly sizing, fee calculation, and
settlement logic from the existing engine.

### Dataset Building

Uses shared intermediate files from `storage/d20260218_build_all_datasets/`:
- `shared/film_metadata.json` — not affected by as-of-date
- `shared/precursor_awards.json` — not affected by as-of-date
- `{category}/oscar_nominations.json` — per-category, not affected by as-of-date

Only the **merge step** (`build_dataset --mode merge --as-of-date {date}`) is
re-run per snapshot date, which applies feature gating based on the calendar.

---

## Implementation Plan

### Phase 1: Infrastructure & Setup

1. **Setup script** (`setup_configs.sh`):
   - Copy feature configs from `modeling/configs/features/` into experiment storage
   - Generate param grids via `generate_tuning_configs.py` into experiment storage
   - Copy CV split
2. **Market discovery** (`trading/market_discovery.py`):
   - Fetch event markets from Kalshi API
   - Build nominee ticker maps for all 9 categories x 2025
3. **Name matching** (`trading/name_matching.py`):
   - `normalize_name()` reusing `data/utils.py`
   - `match_nominees()` with fuzzy matching (rapidfuzz)
   - Manual overrides dict
4. **Snapshot date derivation** (`get_post_nomination_snapshot_dates()`)
5. **Historical event registry** extension in `oscar_market.py`
6. **Dataset build script** (`build_datasets.sh`):
   - 6 snapshots x 9 categories = 54 merge runs
   - Reuses shared intermediates from `d20260218_build_all_datasets`
   - Idempotent (skips if output exists)

### Phase 2: 2025 Backtest (End-to-End)

1. **Build as-of-date datasets**: 54 merge runs
2. **Train temporal model snapshots**: 216 `build_model` runs
   - All with `--feature-selection --importance-threshold 0.90`
   - Parallelizable: 4 models x 9 categories can run independently
3. **Fetch Kalshi candles + trades** for all 9 2025 event tickers
4. **Match nominees**: model names <-> Kalshi names (validate manually)
5. **Run data quality validation** (prices sum, probs sum, no orphans, etc.)
6. **Estimate spreads** from trade history (report spread/liquidity stats)
7. **Run daily buy-once backtests**: ~64K evals total (seconds)
8. **Analyze**:
   a. Model accuracy evolution across snapshots (heatmaps per model type)
   b. Classification vs trading performance (scatter, rank correlation)
   c. Model vs market divergence + market-favorite baseline
   d. Per-precursor P&L breakdown
   e. Return decomposition (selection / timing / sizing alpha)
   f. Trading parameter grid analysis (best config selection)
   g. Spread + liquidity report
9. **Select best config** for 2026 (pre-commitment)
10. **Write README** with tables, plots, findings

---

## Output Structure

```
storage/d20260220_backtest_strategies/
+-- ticker_inventory.json              # Kalshi market discovery (exists)
+-- configs/                           # Copied at setup (frozen)
|   +-- features/                      # {category}_{lr,gbt}_full.json
|   +-- param_grids/                   # lr_grid.json, clogit_grid.json, etc.
|   +-- cv_splits/                     # leave_one_year_out.json
+-- market_data/                       # Shared price + trade data
|   +-- candles/                       # {event_ticker}_candles.parquet
|   +-- trades/                        # {event_ticker}_trades.parquet
|   +-- spreads/                       # {event_ticker}_spreads.json
+-- nominee_maps/                      # Name matching results
|   +-- {category}_2025.json
|
+-- 2025/                              # Historical backtest
|   +-- datasets/
|   |   +-- {category}/{as_of_date}/   # Per-snapshot merged datasets
|   +-- models/
|   |   +-- {category}/{model_type}/{as_of_date}/  # Temporal model snapshots
|   +-- results/
|   |   +-- model_accuracy.csv         # Per-snapshot model evaluation
|   |   +-- model_vs_market.csv        # Model-market divergence tracking
|   |   +-- daily_pnl.csv             # Per-day x per-config x per-model P&L
|   |   +-- per_precursor_pnl.csv      # Per-event P&L breakdown
|   |   +-- return_decomposition.csv   # Selection / timing / sizing alpha
|   |   +-- market_favorite_baseline.csv # Buy-the-favorite baseline P&L
|   |   +-- spread_report.csv          # Spread + liquidity + max safe size
|   |   +-- data_quality_report.csv    # Validation check results
|   |   +-- classification_vs_trading.csv  # Brier vs P&L per model x category
|   |   +-- best_config.json           # Pre-committed config for 2026
|   |   +-- runner_up_configs.json
|   +-- plots/
|       +-- model_accuracy_heatmap_{model_type}.png  # 4 heatmaps
|       +-- brier_vs_pnl_scatter.png   # Classification vs trading
|       +-- model_vs_market_divergence.png
|       +-- pnl_by_timing.png
|       +-- pnl_by_category.png
|       +-- per_precursor_pnl.png
|       +-- return_decomposition.png
|       +-- config_robustness_heatmap.png
|       +-- spread_liquidity.png
|
+-- analysis/                          # Cross-year synthesis (future)
    +-- aggregate_results.csv
    +-- plots/
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Nominee name mismatch | Wrong positions | Fuzzy matching + overrides + validation printout |
| Small N per category (5 noms) | Noisy per-category P&L | Aggregate across categories; permutation test (deferred) |
| Extra pre-nom nominees on Kalshi | Silent errors | Filter to official nominees only; renormalize |
| Fee structure may have changed | Wrong P&L | Test both maker/taker in grid |
| 2025 markets less liquid | Unreliable spreads | Estimate from trades; report liquidity stats; max position cap |
| Config overfit to 2025 | Bad 2026 perf | Pre-commit protocol; expand to 2022-2024 later |
| 216 model trainings is slow | ~7-10 hours | Parallelize across models/categories; idempotent scripts |

---

## Dependencies

- **d20260220_feature_ablation**: recommended configs and param grids (Phase 3 complete).
- **d20260218_build_all_datasets**: shared intermediate files for all 9 categories.
- **Kalshi public API**: price + trade data for settled markets.
- **CALENDAR_2025**: all precursor dates in `schema.py`.
- **rapidfuzz** (or thefuzz): fuzzy string matching for nominee names.

---

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"

# Step 0: Setup configs (copy features, generate param grids)
bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/setup_configs.sh

# Step 1: Build datasets for all snapshots
bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/build_datasets.sh 2>&1 \
  | tee storage/d20260220_backtest_strategies/build_datasets.log

# Step 2: Train models for all snapshots (4 models x 9 cats x 6 snapshots)
bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/train_models.sh 2>&1 \
  | tee storage/d20260220_backtest_strategies/train_models.log

# Step 3: Fetch market data + match nominees + run backtests
bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/run_backtests.sh 2>&1 \
  | tee storage/d20260220_backtest_strategies/run_backtests.log

# Step 4: Analysis + plots
uv run python -m oscar_prediction_market.one_offs.d20260220_backtest_strategies.analyze 2>&1 \
  | tee storage/d20260220_backtest_strategies/analyze.log
```
