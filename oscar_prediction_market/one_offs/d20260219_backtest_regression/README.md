# Backtest Refactor Regression Test

**Storage:** `storage/d20260219_backtest_regression/`

Regression test for the `d20260219_backtest_refactor` branch. Tests whether
the refactored code produces expected results compared to the pre-refactor
baseline.

## Motivation

The refactor made structural and behavioral changes to `trading/backtest/`:

### Structural changes (should not affect numerics)

1. **Flattened `backtest/` sub-package** into `trading/` — `backtest.py`,
   `portfolio.py`, `price_utils.py`, `config.py` are now top-level modules.

2. **Split `BacktestConfig`** into `TradingConfig` (shared with live trading)
   + `SimulationConfig` (backtest-only), nested in `BacktestConfig`.

3. **`KellyConfig` embedded** in `TradingConfig` instead of duplicating fields.
   Added `KellyConfig.with_bankroll()` for dynamic bankroll mode.

4. **Added `Fill` model** and `BacktestResult.trade_log` for granular trade
   recording.

5. **Added market-consensus benchmark** to `BacktestResult`.

6. **Moved `model_loader.py`** to `one_offs/d20260214_trade_signal_ablation/`.

7. **`SettlementResult.return_pct`** fixed from broken computed field to stored
   float.

### Behavioral changes (cause expected regressions)

8. **Directional spread handling**: Spread moved from `get_edge()` to the
   caller (`generate_signals()`). Buy signals use `close + spread`; sell
   signals use `close - spread`. Previously sells used `close + spread`
   (unrealistically favorable for seller).

9. **No-prediction sell price fix**: Outcomes with no model prediction that
   trigger a SELL previously used `execution_price_cents=0` (zero revenue).
   Now correctly uses `close - spread`.

## Regression Results (2025-02-19)

**12/24 configs passed, 12 failed.** All failures are from behavioral changes
#8 and #9 above. The 12 passing configs are those with zero trades (no sells
occurred).

### Failure pattern

Two effects cause regressions:

1. **Regular sells**: Revenue decreased (from `close + spread` to
   `close - spread`, a `2 × spread` difference per contract). Configs where
   this dominated show **lower** final_wealth.

2. **No-prediction sells**: Revenue increased (from $0 to `(close - spread)`).
   Configs where this dominated show **higher** final_wealth.

Example diffs:

| Config | Old wealth | New wealth | Δ | Dominant effect |
|--------|-----------|------------|---|-----------------|
| avg_kelly0.10_edge0.05_maker_indep | $1,004.47 | $962.84 | -$41.63 | Regular sell fix |
| avg_kelly0.25_edge0.08_floor10_maker | $957.97 | $968.07 | +$10.10 | No-prediction sell fix |

Both changes are **correctness improvements**: sells should execute at a worse
price for the seller (not better), and zero-revenue sells were a bug.

### Concrete example: regular sell (change #8)

Sell 10 contracts of "Anora" at market close = 30¢, spread = 2¢.

In a limit order book, the spread is the cost of immediacy. The buyer
pays the ask (above midpoint) and the seller receives the bid (below
midpoint). The close price approximates the midpoint.

```
Old code path (get_edge with spread_penalty_cents):
  get_edge(execution_price_cents=30, spread_penalty_cents=2, side=BUY)
    → adjusted_price = 30 + 2 = 32  (BUY side: close + spread)
  sell signal reuses edge.execution_price_cents = 32
    → revenue = 10 × 32¢ = $3.20  ← seller gets MORE than close (wrong!)

New code path (caller adjusts price by direction):
  sell signal: execution_price = max(0, 30 - 2) = 28  (close - spread)
    → revenue = 10 × 28¢ = $2.80  ← seller gets LESS than close (correct)

Δ = -$0.40  (= 10 contracts × 2 × 2¢ spread)
```

The bug: old code always computed `edge.execution_price_cents` for the BUY
side (`close + spread`), then reused that same price for SELL signals. A
seller hitting the bid should receive `close - spread`, not `close + spread`.

### Concrete example: no-prediction sell (change #9)

Hold 5 contracts of "The Brutalist" but model has no prediction for it
(e.g., it dropped off the model's feature set). Market close = 15¢,
spread = 2¢.

```
Old code path:
  edge_or_none is None → SELL with execution_price_cents=0
    → revenue = 5 × 0¢ = $0.00
    → fee = $0.00  (fee formula yields 0 at price=0)
    → cash impact: $0.00  ← gave away contracts for free!

New code path:
  edge_or_none is None → SELL with execution_price = max(0, 15 - 2) = 13
    → revenue = 5 × 13¢ = $0.65
    → fee = 5 × ceil(0.07 × 13 × 87 / 100) / 100 = 5 × $0.01 = $0.05
    → cash impact: +$0.60  ← correctly liquidates at market price

Δ = +$0.60
```

## How the Golden Fixture Was Captured

24 diverse configs were sampled from the pre-refactor ablation run
(`storage/d20260214_trade_signal_ablation/results/ablation_results.json`, 878 configs),
covering all combinations of:
- **model_type**: `lr`, `gbt`, `avg`, `market_blend` (alpha=0.0 and 0.15)
- **fee_type**: `maker`, `taker`
- **kelly_fraction**: 0.10, 0.15, 0.25
- **bankroll_mode**: `dynamic`, `fixed`

For each config, the fixture stores:
- `engine_config`: the parameters used (before field rename)
- `expected`: summary metrics (`final_wealth`, `total_return_pct`, `total_fees_paid`,
  `total_trades`, `n_snapshots`)
- `expected_snapshots`: per-date portfolio state (`cash`, `total_wealth`, `total_trades`,
  `n_positions`)

## Spread Estimation

The original ablation used `spread_penalty_mode = "trade_data"`: it called
`estimate_spread_penalties()` against Kalshi trade history for 2025-12-01 → 2026-02-15.
Historical trade data for closed markets is deterministic, so re-running the same
call returns the same estimates. The compare script does this automatically.

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"
bash oscar_prediction_market/one_offs/d20260219_backtest_regression/run.sh \
    2>&1 | tee storage/d20260219_backtest_regression/run.log
```

Or directly:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -m oscar_prediction_market.one_offs.d20260219_backtest_regression.compare
```

## Tolerances

| Metric | Tolerance | Reason |
|---|---|---|
| Dollar amounts | $0.015 | Golden values are rounded to 2 decimal places |
| Return pct | 0.05 pp | Golden values are rounded to 1 decimal place |
| Trade counts | exact (0) | Integer — must match exactly |

## Output Structure

```
storage/d20260219_backtest_regression/
├── golden_fixture.json   # 24 pre-refactor reference configs + expected values
└── run.log               # stdout/stderr from most recent run
```
