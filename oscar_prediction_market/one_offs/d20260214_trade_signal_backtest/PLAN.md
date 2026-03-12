# Trade Signal Pipeline — Plan

**Branch:** `feature/trade-signal-pipeline`
**Worktree:** `.worktrees/feature/trade-signal-pipeline`
**Date:** 2026-02-14

---

## Goal

Build a model → trade signal pipeline that:
1. Combines model predictions with live Kalshi orderbook data
2. Computes depth-weighted execution prices and edges net of fees
3. Uses Kelly criterion (single + multi-outcome) for position sizing
4. Supports active management: position deltas, sell/exit signals
5. Backtests over existing temporal snapshots to validate before live use

---

## Architecture

```
trading/                                    ← reusable, no hardcoded config
├── kalshi_client.py                        ← exists
├── price_history.py                        ← exists
├── event_impact.py                         ← exists
├── edge.py                                 ← NEW: execution price + edge estimation
├── kelly.py                                ← NEW: Kelly sizing (single + multi-outcome)
└── signals.py                              ← NEW: edge + sizing → trade signals

one_offs/d20260214_trade_signal_backtest/    ← specific application
├── run.sh                                  ← orchestrate backtest
├── config.json                             ← bankroll, kelly_fraction, thresholds
├── generate_signals.py                     ← calls trading/ with config
├── analyze_signals.py                      ← P&L curves, drawdown, edge accuracy
└── README.md
```

---

## Component 1: `trading/edge.py`

### Responsibilities

- Depth-weighted execution price from orderbook
- Edge computation net of fees
- Both buy-side and sell-side execution prices

### Key Functions

```
get_execution_price(orderbook, side: "buy"|"sell", n_contracts: int) -> float
    Walk ask (for buy) or bid (for sell) side, VWAP across consumed levels.
    Returns average fill price in cents.

get_edge(model_prob, orderbook, side, n_contracts) -> EdgeResult
    execution_price = get_execution_price(...)
    implied_prob = execution_price / 100
    gross_edge = model_prob - implied_prob  (for buy YES)
    fee_per_contract = estimate_fee(execution_price)
    net_edge = gross_edge - fee_adjustment
    Returns: EdgeResult(execution_price, implied_prob, gross_edge, net_edge, fee)
```

### Fee Model

**Finding from Kalshi docs:** "Kalshi makes money by charging a transaction fee on
the expected earnings on the contract." The fee is charged at order execution time
(per fill), not at settlement.

**Current code:** `estimate_taker_fee_cents(price_cents) = 7% of price_cents`,
clamped to [7¢, $1.75]. This appears to be: fee = 7% × contract_price.

**Nuance to verify:** Some sources suggest the fee is 7% × min(yes_price, no_price),
i.e., fee is on the cheaper side (which bounds the expected profit). The current
code uses `price_cents` directly without the `min()`. This matters for contracts
priced above 50¢.

**Decision:** Start with the existing formula. Compare against actual `fee_cost`
values from the API fills data to validate. The authenticated client's
`get_fills()` returns exact `fee_cost` per fill — we can calibrate against real
data.

### Orderbook Input Format

The existing `fetch_current_orderbook()` returns:
`{ticker, nominee, best_yes_bid, best_yes_ask, spread, midpoint, bid_depth, ask_depth}`

This only has the top-of-book. For depth-weighted pricing we need the full
orderbook levels. The `KalshiPublicClient.get_orderbook(ticker, depth)` returns
raw levels — use that directly.

### Design Decision: Handling Thin Orderbooks

If the orderbook doesn't have enough depth to fill the position, options:
1. Fill what's available, report max fillable quantity
2. Use last available level for remaining contracts (pessimistic)
3. Reject the signal entirely

**Decision:** Option 1. Report `max_fillable_at_depth` alongside the VWAP. Let the
sizing layer decide what to do with partial fills.

---

## Component 2: `trading/kelly.py`

### Mode A: Independent Kelly

For each nominee independently:

```
f* = (model_prob - implied_prob) / (1 - implied_prob)
```

Where `implied_prob` = depth-weighted execution price / 100.

Apply `kelly_fraction` multiplier (0.25 = quarter Kelly, 0.5 = half, 1.0 = full).

Convert fraction to contracts: `n_contracts = floor(f* × kelly_fraction × bankroll / execution_price_dollars)`

If total allocation across all nominees exceeds a cap, scale all positions down
proportionally.

### Mode B: Multi-Outcome Kelly (Optimization)

Maximize expected log-wealth over all mutually exclusive outcomes:

$$E[\log(W)] = \sum_{i \in S} q_i \cdot \log\!\left(W_0 - \sum_{j} c_j + (100 - p_i) \cdot n_i\right) + \left(1 - \sum_{i \in S} q_i\right) \cdot \log\!\left(W_0 - \sum_{j} c_j\right)$$

Where:
- $q_i$ = model probability of nominee $i$ winning
- $p_i$ = execution price (cents) for nominee $i$
- $n_i$ = contracts on nominee $i$ (decision variable)
- $c_j = p_j \cdot n_j / 100$ = cost of position on nominee $j$ (dollars)
- $S$ = set of nominees we're betting on
- The last term = scenario where none of our picks win

Subject to:
- $n_i \geq 0$ for all $i$
- $\sum_j c_j \leq W_0$ (can't bet more than bankroll)
- $n_i \leq \text{max\_position}$ (per-nominee cap)
- Only consider nominees where independent Kelly edge > min_edge threshold

Use `scipy.optimize.minimize` with `SLSQP` method.

The execution price is a function of position size (from `edge.py`), so the
optimizer needs to call `get_execution_price(orderbook, n_contracts)` at each
iteration. This couples edge and kelly — the interface should support this.

### Key Parameters (all configurable)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `bankroll` | Total capital for this market (USD) | 1000 |
| `kelly_fraction` | Multiplier on Kelly optimal (1.0=full) | 0.25 |
| `min_edge` | Minimum net edge to consider (fraction) | 0.05 |
| `max_position_per_nominee` | Cap per nominee (USD) | 250 |
| `max_total_exposure` | Cap total outlay (USD) | 500 |

### Output

```python
@dataclass
class KellyResult:
    mode: str  # "independent" or "multi_outcome"
    nominees: list[NomineeAllocation]
    total_outlay_dollars: float
    expected_log_growth: float

@dataclass
class NomineeAllocation:
    nominee: str
    model_prob: float
    execution_price_cents: float
    net_edge: float
    kelly_fraction_raw: float   # before kelly_fraction multiplier
    kelly_fraction_applied: float
    recommended_contracts: int
    outlay_dollars: float
    max_profit_dollars: float   # if this nominee wins
    loss_if_wrong_dollars: float  # outlay (you lose it all)
```

---

## Component 3: `trading/signals.py`

### Responsibilities

- Orchestrate edge + kelly → trade recommendations
- Compute position deltas given current holdings
- Generate sell/exit signals when edge flips

### Key Data Structures

```python
class TradeSignal(BaseModel):
    timestamp: datetime
    nominee: str
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    model_prob: float
    execution_price_cents: float
    net_edge: float
    current_contracts: int
    target_contracts: int
    delta_contracts: int           # positive = buy, negative = sell
    outlay_dollars: float          # for this delta
    reason: str                    # "positive edge", "edge flipped", etc.

class SignalReport(BaseModel):
    timestamp: datetime
    bankroll_dollars: float
    kelly_mode: str
    kelly_fraction: float
    min_edge: float
    signals: list[TradeSignal]
    total_current_outlay: float
    total_target_outlay: float
    total_delta_outlay: float
    # Summary metrics
    expected_portfolio_profit: float
    max_portfolio_loss: float
```

### Sell/Exit Signals

A position should be sold when:
1. **Edge flips negative:** `model_prob < implied_prob` (model now thinks market is right or overpricing)
2. **Edge shrinks below threshold:** edge was 10% but is now 2% — not worth the capital
3. **Better opportunity:** rebalancing capital to a nominee with higher edge

Selling has transaction costs (spread + fee on the sell side). So the sell
threshold should account for round-trip costs:

```
sell_threshold = -(spread_cost + buy_fee + sell_fee) / contract_value
```

Only sell if the negative edge exceeds the transaction cost of closing.

### Stateless Design

`generate_signals(model_predictions, orderbooks, current_positions, config) -> SignalReport`

All state (current positions, bankroll) is passed in. The function is pure.
The one-off script handles persistence and position tracking.

---

## Component 4: One-Off `d20260214_trade_signal_backtest/`

### Goal

Backtest the trade signal pipeline over the 11 existing temporal snapshots
(Dec 2025 – Feb 2026) using historical Kalshi prices. Answer:
- Were the model's edges real and tradeable?
- What would the P&L have been with Kelly sizing?
- How does independent vs. multi-outcome Kelly compare?
- What's the optimal kelly_fraction for this market?

### Data Sources

- **Model predictions:** `storage/d20260211_temporal_model_snapshots/` — 11 snapshot
  dates × {lr, gbt} models → per-nominee probabilities
- **Market prices:** Kalshi daily candlesticks (via `price_history.py`) — daily
  close prices for each nominee
- **Orderbook:** NOT available historically. Use daily close as proxy for
  execution price (no depth weighting in backtest — only live signals get that).

### Backtest Logic

For each snapshot date (chronologically):
1. Load model predictions for that date
2. Get market close prices for that date
3. Compute edges (using close price as execution price, no spread in backtest)
4. Run Kelly sizing (both modes) with configurable parameters
5. Compute position deltas vs. previous snapshot's positions
6. Track simulated portfolio: positions held, cash, mark-to-market

At ceremony (final date): settle all positions. Compute total P&L.

### Key Outputs

- **Simulated P&L curve** over 11 snapshots
- **Per-nominee P&L breakdown**
- **Comparison table:** independent Kelly vs. multi-outcome Kelly
- **Sensitivity analysis:** P&L at kelly_fraction = {0.1, 0.25, 0.5, 1.0}
- **Edge accuracy:** what fraction of "positive edge" signals were correct
  (model prob closer to outcome than market price)
- **Max drawdown** during the period

### Config (`config.json`)

```json
{
    "bankroll_dollars": 1000,
    "kelly_fraction": 0.25,
    "min_edge": 0.05,
    "max_position_per_nominee_dollars": 250,
    "max_total_exposure_dollars": 500,
    "sell_threshold_edge": -0.03,
    "model_types": ["lr", "gbt"],
    "snapshots_dir": "storage/d20260211_temporal_model_snapshots",
    "price_start_date": "2025-12-01",
    "price_end_date": "2026-02-14"
}
```

---

## Future Work (Not in Scope Now)

### Live Signal Mode

Once backtest validates the pipeline:
- Run `build_model.py` with current `as_of_date` to get fresh predictions
- Fetch live orderbook for depth-weighted pricing
- Generate signals with position deltas vs. current Kalshi positions
- Display actionable trade table

**Trigger:** Re-run after major precursor events (don't need to run daily since
features update slowly). Key remaining dates before ceremony (March 2, 2026):
- BAFTA (Feb 16), SAG (Feb 23), other late precursors.

### Order Execution

`KalshiClient` is currently read-only. Would need a `KalshiTradingClient` with
`place_order()`, `cancel_order()` methods. Not needed for signal generation —
trades can be placed manually based on signal output.

---

## Implementation Order

| Step | What | Depends On |
|------|------|------------|
| 1 | `trading/edge.py` | Existing orderbook/fee code |
| 2 | `trading/kelly.py` | `edge.py` |
| 3 | `trading/signals.py` | `edge.py`, `kelly.py` |
| 4 | One-off backtest scripts | All of the above |
| 5 | Analyze backtest results | Step 4 output |

---

## Open Questions

### Q1: Fee formula verification

The existing code uses `fee = 7% × price_cents`. Kalshi docs say "fee on expected
earnings." Some interpretations suggest `fee = 7% × min(yes_price, no_price)`.

**Action:** Compare `estimate_taker_fee_cents()` output against actual `fee_cost`
from API fills. If we have fills data (from the authenticated client), we can
calibrate. If not, use current formula and note the uncertainty.

Does the user have actual fill data from Kalshi trades we can use to validate the
fee formula?

for now keep a note in code and we could modify once we get some trade.

### Q2: Backtest execution price realism

The backtest uses daily close price as a proxy for execution price (no orderbook
depth data for historical dates). This makes the backtest optimistic:
- Ignores bid-ask spread
- Ignores slippage from orderbook depth
- Ignores the fact that placing a large order moves the price

**Mitigation options:**
1. Add a fixed spread penalty (e.g., 2¢ each way) to simulate realistic fills
2. Use trade data (from `fetch_trade_history`) to estimate historical spreads
3. Accept the optimism and note it as a caveat

**Recommendation:** Option 2 if trade data has enough granularity, otherwise
option 1.

Which approach does the user prefer?

### Q3: Model type for backtest

The temporal snapshots have both LR and GBT predictions. Options:
1. Backtest each model type independently → compare which model generates better
   trading P&L
2. Use a simple average of LR + GBT as the "model probability"
3. Run all three (LR, GBT, average) and compare

**Recommendation:** Option 3. The reusable components don't care about model type —
they just take probabilities. The one-off can loop over model types.

### Q4: Position tracking granularity

For active management across snapshots, need to track:
- Contracts held per nominee (bought at what price)
- Average cost basis
- Unrealized P&L (mark-to-market at current price)
- Whether to use FIFO or average cost for partial sells

**Recommendation:** Average cost basis (simpler, standard for Kalshi). Track as
a simple `{nominee: {contracts: int, avg_cost_cents: float}}` dict.

### Q5: Bankroll evolution

As the backtest progresses and positions change value, does the bankroll grow/shrink?
Options:
1. **Fixed bankroll:** Always size off the initial $1000, ignore paper gains/losses
2. **Dynamic bankroll:** Bankroll = cash + mark-to-market value of positions.
   Kelly sizes off current wealth.

**Recommendation:** Option 2 (dynamic). This is how Kelly is designed to work —
the fraction is of current wealth, not initial wealth. But option 1 is simpler
and avoids the "Kelly ruin" problem where a string of losses shrinks the bankroll
to near-zero.

Which does the user prefer? Could also implement both and compare.

### Q6: Pydantic vs dataclass for trading models

The existing trading code uses `@dataclass` (see `event_impact.py`). The modeling
code uses Pydantic `BaseModel`. For new trading models (`EdgeResult`, `KellyResult`,
`TradeSignal`, `SignalReport`):
- Pydantic: validation, JSON serialization, consistency with modeling code
- Dataclass: simpler, consistent with existing trading code

**Recommendation:** Pydantic. The signal reports will be serialized to JSON for
logging. Validation is valuable (e.g., `model_prob` must be 0-1).



answers:
- fee validation: not sure you can try if you can get it 
- backtest spread simulation: lets have option to do estimate historical spreads from trade data or fix 2 cent penalty each way
- model types P: yeah lets do all 3.
- position tracking: ok
- backroll evolution : yeah lets just do average cost basis for simplicity
- pydantic vs dataclass: lets prefer pydantic overall (and lets refactor event impact to use pydantic as well)

---

## Decisions (2026-02-14)

All open questions resolved:

| Question | Decision |
|----------|----------|
| Settlement | Mark-to-market only + hypothetical "what-if X wins" table (ceremony is March 2) |
| Fee formula | Keep existing `7% × price_cents`. Note in code; calibrate against real fills later |
| Spread estimation | Trade-data-first with fixed 2¢ fallback if insufficient data |
| Model types | All three: LR, GBT, average |
| Position tracking | Average cost basis |
| Bankroll evolution | Both fixed and dynamic — run both, compare |
| Kelly + orderbook coupling | Decoupled — Kelly takes fixed prices, no live orderbook iteration |
| Pydantic vs dataclass | Pydantic everywhere. Refactored `event_impact.py` dataclasses too |
| Prediction loading | Read individual snapshot directories |
| Tests | Skipped for now |

## Implementation Status

All components implemented and passing `make all`:

- [x] `trading/edge.py` — `ExecutionResult`, `EdgeResult`, `get_execution_price()`, `get_edge()`, `estimate_spread_from_trades()`
- [x] `trading/kelly.py` — `KellyConfig`, `NomineeAllocation`, `KellyResult`, `independent_kelly()`, `multi_outcome_kelly()`
- [x] `trading/signals.py` — `SignalConfig`, `Position`, `TradeSignal`, `SignalReport`, `generate_signals()`, `print_signal_report()`
- [x] `trading/event_impact.py` — refactored `PriceChange`, `Fill`, `PositionSummary` from dataclass to Pydantic
- [x] `one_offs/d20260214_trade_signal_backtest/generate_signals.py` — full backtest with portfolio tracking and hypothetical settlement
- [x] `one_offs/d20260214_trade_signal_backtest/analyze_signals.py` — wealth curves, settlement heatmap, position evolution plots
- [x] `one_offs/d20260214_trade_signal_backtest/run.sh` — orchestration script
- [x] `storage/d20260214_trade_signal_backtest/config.json` — experiment config