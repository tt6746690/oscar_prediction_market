# PLAN: Buy-and-Hold Backtest Experiment

**Date:** 2025-02-25
**One-off:** `d20260225_buy_hold_backtest`
**Storage:** `storage/d20260225_buy_hold_backtest/`
**Related:** `d20260220_backtest_strategies` (rebalancing-mode backtest)

---

## Motivation

The `d20260220_backtest_strategies` experiment used a **rebalancing backtest
engine** that re-evaluates positions at every trading day.  Even though
`NEVER_SELL_THRESHOLD = -1.0` was set to prevent edge-triggered sells, the
Kelly rebalancing path (`_decide_action`'s "reducing position" branch) still
fires when the Kelly target decreases due to market price changes.  This means:

- Positions are opened, adjusted, and sometimes fully closed *before*
  settlement — not the buy-once-hold-to-resolution strategy intended.
- Categories showing "$0 P&L with non-zero trades" had all positions closed
  via Kelly rebalancing, incurring round-trip fees but reporting $0 P&L.
- The effective strategy was **daily portfolio management**, not the
  **single-entry, hold-to-settlement** strategy we want to evaluate.

This experiment implements the correct mental model:

> At each precursor event (6h after inferred datetime), evaluate model
> predictions vs. market prices.  If edge > threshold, buy once and hold
> until ceremony resolution.  Observe P&L per entry point, and compute
> aggregate performance across the Oscar season.

---

## Investigation: How the Rebalancing Engine Works

### The rebalancing backtest engine loop

```
for each TradingMoment (one per trading day, ~30 total):
    1. Read model predictions from active snapshot
    2. Read market prices at this timestamp
    3. Compute edges → Kelly targets → deltas vs current positions
    4. Execute BUY/SELL signals
    5. Record portfolio snapshot
then: settle remaining positions against known winner
```

### Why positions get closed before settlement

Even with `sell_edge_threshold = NEVER_SELL_THRESHOLD (-1.0)`, three paths
can close positions:

1. **Kelly rebalancing** (`_decide_action`, signals.py L403-407): when Kelly
   targets fewer contracts than held (`delta < 0`), emits SELL.  This fires
   when market price moves toward model price (edge shrinks → Kelly wants
   less exposure).  This is the main culprit.

2. **Direction flips** (`_process_existing`): if Kelly now wants NO instead
   of YES, it sells all YES then buys NO.

3. **Orphan liquidation**: if a nominee drops out of predictions across
   snapshot changes.

### Concrete example of the problem

For "Original Screenplay" with the recommended config:
- Feb 8: model assigns Anora 65%, market at 30¢ → big edge → Kelly buys
- Feb 15: market has moved to 45¢ → edge shrinks → Kelly target decreases
  → engine sells some contracts
- Feb 20: market at 55¢ → edge further shrinks → Kelly sells remaining
- Settlement: no positions → P&L = $0 (minus ~$15 round-trip fees in cash)

The 21 trades reported are ~10 buy + ~10 sell trades across snapshots, not
21 independent entries.

---

## Design: Buy-Once-Hold-to-Settlement Backtest

### Core concept

Each **precursor event** defines an **entry point**.  At each entry point:

1. Compute model predictions (from the snapshot triggered by this event)
2. Fetch market prices at entry timestamp (6h after event end)
3. Compute edge for each nominee
4. If edge > threshold, buy YES/NO contracts (Kelly-sized against bankroll)
5. **Hold all positions until ceremony resolution** (no rebalancing)
6. Settle: winner pays $1, losers pay $0

This produces one `EntryResult` per (entry_point, category, model, config).

### Key differences from rebalancing engine

| Aspect | Rebalancing (d20260220) | Buy-and-Hold (this) |
|--------|------------------------|---------------------|
| Trading frequency | Every day (~30 moments) | Once per snapshot (~6 entry points) |
| Position management | Kelly rebalances daily | Buy once, freeze, hold |
| Sell logic | Kelly reduction + edge flip | None until settlement |
| P&L driver | Edge at entry + rebalancing P&L | Edge at entry only |
| Interpretability | Complex (many transactions) | Simple (one entry → one P&L) |
| Aggregation | Sum across all moments | Per-entry P&L, then aggregate |

### What we want to learn

1. **Per-entry-point P&L**: if I enter after DGA (Feb 8), what's my P&L per
   category?  What about after BAFTA (Feb 16)?
2. **Temporal evolution**: how does edge and expected P&L change as the Oscar
   season progresses?  Is there a clear "best window" to enter?
3. **Risk profile**: what's the worst-case loss for a single entry?
4. **Aggregate strategy**: if I enter at every precursor event (buying
   additional contracts each time), what's the combined P&L?
5. **Model comparison**: which model is best for buy-and-hold at different
   entry points?
6. **Ensemble vs individual**: does the ensemble still dominate when we
   remove rebalancing complexity?

---

## Approach Options

### Option A: Standalone buy-and-hold engine (skip BacktestEngine)

Build a simple function that takes predictions + prices → computes edge →
Kelly sizes → settles against winner.  No engine loop, no position tracking.

```python
def buy_hold_entry(
    predictions: dict[str, float],    # {nominee: model_prob}
    prices: dict[str, float],         # {nominee: price_cents}
    winner: str,
    config: TradingConfig,
    bankroll: float,
    spread_penalties: dict[str, float] | None,
) -> EntryResult:
    """Evaluate a single buy-and-hold entry point."""
    # 1. Compute execution prices with spread
    exec_prices = ExecutionPrices.from_close_prices(prices, spread)
    # 2. Generate signals (edges + Kelly) — but only BUY signals
    report = generate_signals(predictions, exec_prices, {}, config)
    # 3. Execute buys only
    positions, cash = execute_buys(report, bankroll)
    # 4. Settle immediately against winner
    settlement = settle_positions(positions, cash, winner, bankroll)
    return EntryResult(...)
```

**Pro:**
- Dead simple — no stateful engine, no rebalancing ambiguity.
- Each entry is independent — easy to parallelize and understand.
- Reuses existing `generate_signals` + `settle_positions` functions.
- Result is a clean table: (entry_date, category, model, config) → P&L.

**Con:**
- Doesn't model the "cumulative" strategy (enter at every event).
- Can't capture intra-season dynamics (mark-to-market evolution).

**Mitigation:** The cumulative strategy can be computed by summing independent
entries (since each entry is sized against fresh bankroll).  MtM evolution can
be derived from market price time series without needing an engine.

### Option B: Add buy-once mode to BacktestEngine

Add a flag to `SimulationConfig`:

```python
class SimulationConfig(BaseModel):
    position_mode: Literal["rebalance", "buy_once"] = "rebalance"
```

When `position_mode == "buy_once"`, the engine skips signal generation for
outcomes that already have positions.  After the first BUY, the position is
frozen until settlement.

**Pro:**
- Reuses existing engine infrastructure (snapshots, fills, MtM tracking).
- Can produce intra-season MtM snapshots for wealth evolution plots.

**Con:**
- Adds complexity to an already complex engine.
- The engine still processes ~30 moments, but most become no-ops.
- Mixing two execution modes in one engine increases maintenance burden.

### Option C: Multi-entry positions list (no engine)

For each entry point, compute the buy-and-hold entry independently (like
Option A).  Then aggregate by layering entries:

```python
entries: list[EntryResult] = []
for snap_date, predictions, prices in entry_points:
    entry = buy_hold_entry(predictions, prices, winner, config, bankroll)
    entries.append(entry)

# Aggregate: total P&L = sum of individual entry P&Ls
# Or: cumulative deployment = sum of bankrolls committed
```

**Pro:**
- Cleanest separation: each entry is completely independent.
- Natural "temporal P&L" view: can see P&L contributed by each entry point.
- Easy to answer "if I only entered after DGA, what happens?"

**Con:**
- Same as Option A — no intra-season MtM tracking.

### Recommendation: Option A/C (standalone, no engine)

The buy-and-hold evaluation doesn't need an engine.  It's a **single-step
computation**: predictions + prices + winner → P&L.  Using the engine adds
complexity without benefit.

Implement Option A as the core function, then use Option C's aggregation
pattern to compute cumulative/temporal views.  If we later want MtM
evolution plots, we can derive them from market price time series (which
we already have) without needing the engine to track positions.

---

## Implementation Plan

### Phase 1: Core buy-and-hold evaluation function

**File:** `oscar_prediction_market/trading/buy_hold.py`

Place this in the trading module (not the one-off) since it's reusable
infrastructure — any experiment wanting buy-and-hold evaluation will use it.

```python
class EntryResult(BaseModel):
    """Result of a single buy-and-hold entry at one point in time."""
    entry_timestamp: datetime
    snapshot_date: date
    category: str
    model_type: str
    config_label: str
    bankroll: float
    positions: dict[str, Position]  # what was bought
    cash_after_buy: float           # cash remaining after purchases
    total_cost: float               # cost of all positions (including fees)
    settlement: SettlementResult    # P&L from holding to resolution
    n_buys: int                     # number of outcomes purchased
    signals: list[TradeSignal]      # the signals that were generated (for audit)

class BuyHoldEvaluation(BaseModel):
    """Full buy-and-hold evaluation across multiple entry points."""
    entries: list[EntryResult]
    # Aggregation helpers
    total_pnl: float
    total_bankroll_deployed: float
    return_pct: float
```

Functions:
- `evaluate_single_entry(predictions, prices, winner, config, bankroll, spread) → EntryResult`
- `evaluate_across_entries(entry_points, winner, config, bankroll) → BuyHoldEvaluation`

### Phase 2: ProbabilitySource protocol (see PLAN_refactor.md)

The ProbabilitySource interface lives in the trading module and is used by
both this experiment and future live trading.  The buy-and-hold evaluation
function consumes predictions via this protocol:

```python
class ProbabilitySource(Protocol):
    def get_predictions(self, timestamp: datetime) -> dict[str, float]: ...
    @property
    def name(self) -> str: ...
```

### Phase 3: One-off runner script

**File:** `one_offs/d20260225_buy_hold_backtest/run_backtests.py`

```python
def main():
    for category in MODELED_CATEGORIES:
        for model in models + [ensemble]:
            source = make_source(model, category, snapshots)
            for entry_point in entry_points:
                preds = source.get_predictions(entry_point.timestamp)
                prices = get_prices_at(entry_point.timestamp)
                result = evaluate_single_entry(preds, prices, winner, config, bankroll)
                rows.append(result.to_row())
    save_csv(rows)
```

### Phase 4: Analysis and README

- Per-entry-point P&L tables
- Temporal evolution of edge and P&L
- Comparison with rebalancing results (d20260220)
- Risk profile (per-entry and aggregate)
- Model comparison
- Recommended config for 2026 deployment

---

## Entry Points (2025 ceremony)

Using inferred+6h timing from `AwardsCalendar`:

| # | Snapshot Date | Event(s) | Available (ET) | Entry Date |
|:-:|:-------------|:---------|:--------------:|:----------:|
| 1 | 2025-01-23 | Oscar nominations | Jan 23 08:30 | Jan 23 |
| 2 | 2025-02-07 | Critics Choice | Feb 8 04:00 | Feb 8 |
| 3 | 2025-02-08 | DGA, PGA, Annie | Feb 9 01:30 | Feb 9 |
| 4 | 2025-02-15 | WGA | Feb 16 01:00 | Feb 16 |
| 5 | 2025-02-16 | BAFTA | Feb 16 16:30 | Feb 16 |
| 6 | 2025-02-23 | SAG, ASC | Feb 24 01:00 | Feb 24 |

Each entry uses the snapshot triggered by that event, with prices fetched
at the available-at timestamp.

---

## Trading Config Grid

Reuse the same parameter grid from d20260220:
- kelly_fraction: [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
- buy_edge_threshold: [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
- kelly_mode: [independent, multi_outcome]
- fee_type: [maker, taker]
- trading_side: [yes, no, all]
- bankroll_mode: fixed only (no dynamic — no rebalancing means no compounding)

Total: 7 × 7 × 2 × 2 × 3 = 588 configs per (model, category, entry_point).
With 5 models × 9 categories × 6 entry points = 270 (model, cat, entry) combos.
Grand total: 270 × 588 = 158,760 scenarios.

---

## Expected Output

```
storage/d20260225_buy_hold_backtest/
├── 2025/
│   ├── results/
│   │   ├── entry_pnl.csv           # One row per (entry, category, model, config)
│   │   ├── aggregate_pnl.csv       # Summed across entries per (category, model, config)
│   │   ├── model_accuracy.csv      # Per-entry model accuracy
│   │   └── model_vs_market.csv     # Per-entry divergence
│   └── plots/
│       ├── temporal_edge.png        # Edge over time per category
│       ├── entry_pnl_heatmap.png   # P&L by entry point × category
│       ├── aggregate_comparison.png # Buy-hold vs rebalancing comparison
│       └── ...
├── configs/                         # Copied from d20260220 or regenerated
└── run.log
```

---

## Dependencies

- `oscar_prediction_market.trading.buy_hold` (new module)
- `oscar_prediction_market.trading.prediction_source` (new, from refactor)
- Existing: `signals.py`, `portfolio.py`, `kelly.py`, `edge.py`, `signal_delay.py`
- Data: reuse market data from `storage/d20260220_backtest_strategies/market_data/`
- Models: reuse trained models from `storage/d20260220_backtest_strategies/2025/models/`

---

## Success Criteria

1. Each entry point produces a clear, interpretable P&L per (category, model).
2. No position is ever sold before settlement — only BUY fills in the trade log.
3. Aggregate P&L numbers are comparable to (but different from) d20260220.
4. The temporal view answers: "what's the best entry window?"
5. README contains full analysis with tables, plots, and takeaways.

---

## Open Questions

- Should each entry point get fresh bankroll, or should later entries use
  remaining cash from earlier entries?  **Recommendation:** fresh bankroll
  per entry — keeps entries independent and analysable.  The "cumulative"
  view is then a simple sum.

- Should we cap total bankroll across all entries (e.g., $6,000 for 6 entries
  at $1,000 each)?  **Recommendation:** yes, for the aggregate view.  Report
  both per-entry and aggregate returns.

- Should the `evaluate_single_entry` function live in `trading/buy_hold.py` or
  in the one-off?  **Recommendation:** `trading/buy_hold.py` — it's reusable
  infrastructure.  The one-off wires it to Oscar-specific data loading.
