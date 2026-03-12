# Plan: Sell-Side (BUY NO) Alpha

**Date:** 2025-02-22
**Status:** Brainstorm / Future Work

## Context

The current backtest system is **BUY-YES-only** (`sell_edge_threshold=-1.0` in
`run_backtests.py`). The trading module already has SELL signal infrastructure:
`signals.py` generates SELL signals when edge flips below `sell_edge_threshold`,
`edge.py` computes edges for both sides, and `portfolio.py` handles negative
deltas. But there is no **BUY NO** logic — the system only considers buying YES
contracts to go long on a nominee winning.

**BUY NO** = betting a nominee will lose. On Kalshi, a YES contract at 80¢
implies the NO contract at 20¢. If our model says the nominee's true prob is
60% (not the market's implied 80%), we have edge on the NO side.

## Why Sell-Side Matters

In a multi-outcome Oscar category with ~10 nominees, only 1 wins. That means
~9 nominees are overpriced if you aggregate their implied probabilities > 100%.
The market's inefficiency may be larger on the NO side (fading overhyped
nominees) than the YES side (finding the winner).

Concrete alpha sources:

1. **Fading precursor momentum**: Nominee wins a precursor → price spikes →
   market overreacts beyond model's assessment. We BUY NO.
2. **Multi-outcome constraint arbitrage**: Sum of market probs > 100% (overround).
   If our model's probabilities sum to ~100% and the market's sum to 115%,
   there's systematic positive edge on NO positions.
3. **Late-season reversals**: A nominee that was hot early (e.g., led Golden Globe
   odds) but didn't win subsequent precursors. Market may be slow to re-price.
4. **Category-specific patterns**: In some categories, the frontrunner almost
   always wins (Animated Feature) — so all non-frontrunners are overpriced on
   YES. In others (Best Picture under IRV), upsets are more common.

## Current Architecture Audit

### What already works

| Component | Sell-Side Support | Notes |
|---|---|---|
| `types.py` | `Side.SELL`, `TradeAction.SELL` | Enums exist |
| `edge.py` | `get_edge(side=Side.SELL)` | Computes sell-side edge: `implied_prob - model_prob - fees` |
| `signals.py` | SELL signal generation | `_decide_action()` generates SELL when edge < `sell_edge_threshold` |
| `signals.py` | `ExecutionPrices.sell` | Bid-side prices for selling YES |
| `portfolio.py` | `apply_signals()` | Handles negative deltas (reducing position) |
| `kelly.py` | `independent_kelly`, `multi_outcome_kelly` | Only size BUY positions |
| `backtest.py` | `BacktestEngine.run()` | Only tracks YES positions |
| `run_backtests.py` | `sell_edge_threshold=-1.0` | Hard-coded never-sell |

### What's missing

The existing SELL logic is only for **closing existing YES positions** (take
profit or cut losses). It does NOT support:

1. **Opening NO positions from scratch** (no YES position to sell)
2. **Kelly sizing for NO exposure** (different payoff: pay $X, receive $1 if
   nominee loses)
3. **Portfolio risk with mixed YES/NO** (YES profits when nominee wins, NO
   profits when nominee loses — these can hedge each other)
4. **Settlement for NO contracts** (currently `settle_positions` assumes all
   positions are YES)

## Design Options

### Option 1: Symmetric YES/NO in existing framework

Add a `Side` field to positions. Kelly sizes both YES and NO opportunities in
the same pass. Portfolio tracks YES and NO independently per outcome.

```python
class Position(BaseModel):
    side: Side  # NEW: YES or NO
    contracts: int
    avg_cost_cents: float

# Kelly sees both YES and NO opportunities:
# For nominee A at 80¢ market, 60% model:
#   YES edge: 0.60 - 0.80 - fees = negative → don't buy YES
#   NO edge:  0.40 - 0.20 - fees = positive → buy NO
```

**Pros:**
- Unified framework, single signal generation pass
- Portfolio naturally sees both sides
- Kelly can balance YES/NO exposure

**Cons:**
- Major refactor of Position, portfolio, settlement
- Kelly optimization over YES+NO is a different problem (not independent binary
  bets — they're correlated)
- Need to think carefully about buying YES on nominee A AND NO on nominee B
  in the same category

### Option 2: Separate NO-only analysis pipeline

Keep existing YES pipeline unchanged. Build a parallel analysis for NO:

```python
# For each nominee with model_prob < market_prob (enough edge on NO side):
no_edge = (1 - model_prob) - (1 - market_prob / 100) - fees
# Simplifies to: market_prob/100 - model_prob - fees
```

Run independent Kelly on NO opportunities. Track NO positions separately.
Analyze separately.

**Pros:**
- No changes to existing (working) YES pipeline
- Simpler to implement and validate
- Can compare YES-only vs YES+NO in analysis

**Cons:**
- Doesn't capture YES/NO hedging benefits
- Duplicates some infrastructure
- Doesn't compose well for live trading (two separate signal streams)

### Option 3: Virtual "anti-nominee" framing

Treat BUY NO on nominee A as BUY YES on "not-A". The existing multi-outcome
Kelly already handles mutually exclusive outcomes. We just add synthetic
outcomes.

This gets messy fast — "not-A" is perfectly correlated with the set of all
other nominees. Not a clean abstraction.

**Not recommended.**

### Recommendation

**Start with Option 2** for analysis (cheap, informative). If NO-side edge is
significant, invest in Option 1 for production.

## Implementation Plan (Option 2: Parallel NO Analysis)

### Phase 1: Analysis-only (in backtest one-off)

1. **NO edge computation**: For each nominee on each trading day, compute
   `no_edge = market_implied_prob - model_prob - fees`. When `no_edge >
   buy_edge_threshold`, that's a NO-side opportunity.

2. **Sizing**: Use independent Kelly with `p = 1 - model_prob` (prob of winning
   the NO bet) and `execution_price = 100 - market_yes_price` cents.

3. **Tracking**: For each NO position, cost = `contracts * no_price_cents / 100`.
   Settlement: if nominee loses, profit = `contracts * (100 - no_price_cents) /
   100 - fees`. If nominee wins, loss = cost.

4. **Report**: For each category, show:
   - How many NO-side opportunities existed (edge > threshold)
   - Total NO-side P&L if acted on all
   - Comparison: YES-only P&L vs YES+NO P&L
   - Which nominees were most overpriced (biggest NO alpha)

### Phase 2: Integration (if Phase 1 shows promise)

1. Add `Side` field to `Position` in `portfolio.py`
2. Update `generate_signals` to consider both BUY YES and BUY NO
3. Update `settle_positions` for NO contracts
4. Update Kelly to handle mixed YES/NO portfolio
5. Update `BacktestEngine` for combined simulation

## Key Design Questions

1. **Can you hold YES and NO on the same nominee?** On Kalshi, YES + NO = risk-free
   $1 return. So the net position is just the difference. We should probably
   net positions (long YES cancels long NO).

2. **Fee asymmetry**: Are NO fees the same as YES fees on Kalshi? Need to verify
   the fee schedule applies symmetrically.

3. **Spread asymmetry**: NO contracts may have wider spreads (less liquidity)
   than YES contracts for popular nominees. The spread estimation from
   `estimate_spread_from_trades` may need adjustment.

4. **Portfolio correlation**: In multi-outcome Kelly, buying YES on A and NO on
   B are not independent — they're both "not B wins" bets. The correlation
   structure matters for optimal sizing. Multi-outcome Kelly might handle this
   if we frame it right, but need to think through it.

5. **Bankroll allocation**: With YES+NO, you could deploy much more capital
   (buying NO on all non-favorites). Need tighter position limits to avoid
   over-concentration. What fraction of bankroll for NO positions?

## Expected Edge Profile

Rough intuition for where NO alpha lives:

| Scenario | YES Edge | NO Edge | Notes |
|---|---|---|---|
| Clear frontrunner (model agrees with market) | ~0% | ~0% | No alpha |
| Model sees winner the market doesn't | Positive | Negative | Current alpha source |
| Market overhypes a non-winner | Negative | **Positive** | **NO alpha** |
| Market overround (sum > 100%) | Slight positive | **Slight positive** | Both sides have edge |
| Post-precursor momentum overshoot | Negative on winner | **Positive on non-winners** | Fading hype |

The **market overround** case is interesting: Kalshi's Oscar markets often have
implied prob sums >105%. This means there's systematic positive edge on at least
one side. If our model sums to 100%, and we can extract the overround
efficiently, that's structural (not alpha-dependent) profit.

## Priority

This is a **post-bug-fix, post-analysis** project. The YES-only backtest needs
to be correct first. Then we can evaluate whether NO-side alpha is material
enough to warrant the engineering investment.

Rough estimate: Phase 1 (analysis only) = 1–2 days. Phase 2 (integration) =
3–5 days.
