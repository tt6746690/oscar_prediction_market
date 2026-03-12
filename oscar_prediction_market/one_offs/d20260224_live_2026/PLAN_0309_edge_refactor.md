# Refactor Edge Detail & Limit Order Sections

**Date:** 2026-03-09
**Status:** Phase 0 complete — maxedge_100 confirmed. Phase 1-2 next.

## Motivation

The current `_compute_edge_rows()` / `_edge_detail_section()` / `_limit_order_section()`
functions in `generate_report.py` were written quickly and have several issues:
duplicated edge math, simplified Kelly, missing direction consistency, and a fallback
to a hardcoded spread. This plan refactors them to use the canonical trading modules.

---

## Decisions (all settled)

| # | Decision | Resolution |
|---|----------|------------|
| A | Maker execution price | `bid + limit_price_offset` (default 1¢). New field on `TradingConfig`. |
| B | Kelly bankroll for maker | Per-category bankroll (`bankroll / n_cats × weight`). |
| C | Cross-category Kelly | Per-category. One `multi_outcome_kelly()` call per category. |
| D | Which nominees in Kelly | ALL nominees — optimizer filters internally by threshold. |
| E | Sell limit orders | Deferred. Edge table = buy only. Sells via Position Adjustments. |
| F | `compute_allocation_weights()` edge direction | **Keep YES-only (maxedge_100).** Investigation in Phase 0 confirmed: maxedge_100 wins avg_rank 3.5 vs maxabsedge_100's 6.0, and wins on all bootstrap/pairwise/noise metrics. |
| G | Report structure: taker vs maker | Separate reports — one taker, one maker. No consolidated table. See rationale below. |
| H | `limit_price_offset` on TradingConfig | Add the field. `_compute_edges()` uses `bid + offset` for buys. |

---

## Design: `limit_price_offset` on TradingConfig (Decision H)

### Problem

Maker limit orders need an execution price that differs from the ask. Currently
`_compute_edges()` always uses `yes_ask` / `no_ask` (taker execution). For maker
configs, we want `bid + offset` instead — typically bid+1¢ for better fill probability
at a small cost to edge.

### Plan

Add `limit_price_offset` to `TradingConfig`:

```python
class TradingConfig(BaseModel):
    ...
    limit_price_offset: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Offset from bid for limit order pricing, in dollars. "
            "For buys: execution_price = bid + offset. "
            "0.0 = use ask (taker behavior). 0.01 = bid+1¢ (aggressive limit). "
            "Only meaningful when fee_type=maker."
        ),
    )
```

Update `_compute_edges()` in `signals.py`:

```python
if config.limit_price_offset > 0:
    # Limit order: price relative to bid
    buy_price = execution_prices.yes_bid.get(outcome, 0) + config.limit_price_offset
else:
    # Market order: price at ask (current behavior)
    buy_price = execution_prices.yes_ask.get(outcome)
```

This keeps the existing taker behavior unchanged (`offset=0.0` → use ask) and adds
clean maker support. The backtest won't simulate limit fills (fill is never guaranteed),
but the edge/Kelly math is correct for the assumed fill price.

### Impact on existing code

- `TradingConfig` gains one optional field with `default=0.0` → no breakage
- All existing configs pass `offset=0.0` implicitly → taker behavior unchanged
- Maker configs explicitly set `offset=0.01`

---

## Design: Separate reports for taker vs maker (Decision G)

### Rationale

Rather than building a consolidated table with both taker and maker columns, we
generate **two separate reports** from two separate `BacktestConfig`s:

1. **Taker report** (current `edge_20`): `fee_type=TAKER, limit_price_offset=0.0`
   — market orders, immediate execution, backtested
2. **Maker report** (new `edge_20_maker`): `fee_type=MAKER, limit_price_offset=0.01`
   — limit orders at bid+1¢, advisory, not backtested

### Why separate is better

- **Same code path**: both reports use the exact same `_compute_edge_rows()` →
  `Edge` → `multi_outcome_kelly()` pipeline. No special "maker column" logic.
- **Cleaner tables**: each table has one net edge, one Kelly sizing, one action.
  No doubled columns or taker/maker cross-referencing.
- **Independent reading**: the taker report is the canonical one (backtested).
  The maker report is advisory — "if you're willing to use limit orders, here are
  additional opportunities." Reader processes them independently.
- **Config-driven**: just flip `fee_type` and `limit_price_offset`. The
  `RECOMMENDED_CONFIGS` dict grows from 3 entries to 6 (3 taker + 3 maker), but
  the report generation code doesn't change at all.

### What changes

- `recommended_configs.py`: add `edge_10_maker`, `edge_15_maker`, `edge_20_maker`
- `generate_report.py`: no structural changes — just runs for maker configs too
- Assembly: generate both taker and maker reports, or combine into one file with
  clear section headers

### Open question (for later)

Should the maker report's Position Adjustments section show "current positions vs
maker-Kelly targets"? Probably yes — it tells the reader what limit orders to place.
But the "delta" might be misleading since current positions were built with taker
execution. Defer this detail to implementation.

---

## Phase 0: Category allocation investigation — COMPLETE

**Result:** `maxedge_100` (YES-only) confirmed as the best allocation signal.

Full analysis ran in the portfolio-kelly one-off (compare_signals + strategy_robustness):
- maxedge_100 avg_rank **3.5** vs maxabsedge_100's 6.0
- maxedge wins on combined P&L ($98,445 vs $96,680), bootstrap rank-1 (5.8% vs 1.4%),
  pairwise win rate (85.5% vs 83.0%), and entries-positive (94% vs 88%)
- maxabsedge has slightly better 2025 rank (4 vs 6) and cross-year ρ (0.91 vs 0.88),
  but these don't overcome maxedge's 2024 advantage (rank 1 vs 8)
- See portfolio-kelly README for full comparison table

**No changes needed to `compute_allocation_weights()`** — it already uses YES-only
`max(model_prob - market_prob)`, which is the winning strategy.

---

## Phase 1: Add `limit_price_offset` to TradingConfig

1. Add `limit_price_offset` field to `TradingConfig` (default 0.0)
2. Update `_compute_edges()` in `signals.py` to use offset when > 0
3. Add maker configs to `recommended_configs.py`
4. Run `make dev` — no existing tests should break (default=0.0)

---

## Phase 2: Refactor report edge sections

1. **Fail if `half_spread` missing** — raise `ValueError`, no fallback to 0.02
2. **Refactor `_compute_edge_rows()` to use `Edge` model** — both directions,
   single fee type per config (taker OR maker, not both)
3. **Replace simplified Kelly with `multi_outcome_kelly()`** — per-category
   optimization with category-allocated bankroll
4. **Remove `_limit_order_section()`** — replaced by maker report
5. **Update assembly** — generate report for both taker and maker configs
6. **Run pipeline, verify reports** — regenerate and visually inspect
7. **Skip tests** per user request

### Scope

~200 lines modified in `generate_report.py`, ~20 lines in `schema.py`,
~15 lines in `signals.py`, ~10 lines in `recommended_configs.py`.
