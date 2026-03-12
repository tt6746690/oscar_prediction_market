# Trading Module

Pipeline for converting model predictions into trade signals on Kalshi
prediction markets.

## Architecture

```
model predictions + market prices
        │
        ▼
    ┌──────────────────┐
    │ temporal_model   │  Resolve snapshot predictions at a point in time
    └────────┬─────────┘
           │ predictions at timestamp
           ▼
    ┌─────────┐
    │  edge   │  Compare model prob vs. market implied prob, net of fees
    └────┬────┘
         │ Edge per nominee
         ▼
    ┌─────────┐
    │  kelly  │  Size positions via Kelly criterion (independent or joint)
    └────┬────┘
         │ KellyResult with contract allocations
         ▼
    ┌─────────┐
    │ signals │  Generate BUY/SELL/HOLD given current positions
    └─────────┘
         │ list[TradeSignal] with trade recommendations
         ▼
    execution / backtest
```

## Modules

| Module | Purpose |
|--------|---------|
| `schema.py` | Shared `StrEnum` types, config models (`TradingConfig`, `KellyConfig`), and core data models (`Position`, `MarketQuotes`, `KellyAllocation`, `Fill`, `SettlementResult`) |
| `kalshi_client.py` | Generic Kalshi API client (public + authenticated), fee estimation |
| `oscar_market.py` | Oscar-specific: nominee/ticker mappings, price/candle/trade fetching |
| `oscar_data.py` | Oscar market-data helpers (market side only — prices, candles, daily snapshots) |
| `oscar_prediction_source.py` | Oscar-specific prediction loading, namespace remapping, and `SnapshotModel`/`EnsembleModel` construction |
| `oscar_moments.py` | Build `MarketSnapshot` objects from prediction + market price data (bridges data loading → backtest engine) |
| `edge.py` | Execution price from orderbook depth, edge = model_prob − implied_prob − fees |
| `kelly.py` | Kelly criterion position sizing — independent and multi-outcome (SLSQP) |
| `signals.py` | Orchestrate edge + Kelly into actionable trade signals with position deltas |
| `portfolio.py` | `PortfolioSnapshot`, `ExecutionBatch` + portfolio operations (apply signals, settle, mark-to-market) |
| `temporal_model.py` | Temporal model abstraction (`SnapshotModel`, `EnsembleModel`) for time-varying predictions |
| `name_matching.py` | Fuzzy matching between model nominee names and Kalshi ticker names |
| `inspect_trade.py` | Post-hoc trade analysis: reconstruct edge/Kelly for a single historical fill |
| `backtest.py` | `BacktestEngine`, `BacktestSimulationConfig`, `BacktestConfig`, `BacktestResult` |
| `backtest_configs.py` | Shared trading config grids for Oscar backtest parameter sweeps |
| `market_data/` | Ticker discovery and registry for mapping categories to Kalshi event/market slugs |

## Key concepts

**Binary option on Kalshi.** A contract that pays $1 if the outcome occurs, $0
otherwise. Buying a YES contract at 25¢ means you pay $0.25 and receive $1.00 if
the nominee wins — an implied probability of 25%.

**Edge.** The difference between your model's estimated probability and the
market-implied probability, net of fees. A positive edge means the model thinks
the market underprices the outcome.

**Kelly criterion.** A formula for optimal bet sizing that maximizes long-run
growth of capital. The fraction of bankroll to wager is `f* = (p - q) / (1 - q)`
where `p` is the model probability and `q` is the break-even probability
(implied prob + fees). In practice, a fractional Kelly (e.g., 0.25×) is used to
reduce variance.

**Multi-outcome Kelly.** When outcomes are mutually exclusive (exactly one nominee
wins), independent Kelly oversizes because it ignores the constraint that betting
on nominee A implicitly bets against nominee B. The multi-outcome optimizer
maximizes expected log-wealth jointly across all nominees using scipy SLSQP.

**Spread penalty.** In backtesting, we don't have live orderbooks, so we estimate
the bid-ask spread from historical trade data. Spread is applied directionally:
buys execute at `close + spread` (worse for buyer), sells at `close - spread`
(worse for seller). In `edge.py`, the caller is responsible for adjusting the
execution price before computing edge.

## Data flow

1. **kalshi_client** provides generic API access (markets, candlesticks, trades,
   orderbook)
2. **oscar_market** wraps the client with Oscar BP nominee mappings and
   convenience methods (daily prices, trade history, orderbook snapshots)
3. **edge** takes a model probability + execution price -> `Edge` with net edge
4. **kelly** takes a list of `Edge` -> `KellyResult` with contract allocations
5. **signals** combines edge + Kelly + current positions -> `list[TradeSignal]` with
   BUY/SELL/HOLD actions and position deltas
6. **portfolio** applies signals to positions and records fills
7. **backtest** orchestrates the full simulation loop over temporal snapshots

## Configuration

- `TradingConfig` (in `schema.py`): strategy parameters shared between live
  trading and backtesting — Kelly params, edge thresholds, fee type.
- `BacktestSimulationConfig` (in `backtest.py`): backtest-only mechanics — spread
  penalty, bankroll mode, max contracts per day.
- `BacktestConfig` = `TradingConfig` + `BacktestSimulationConfig`.

## Backtest

The backtest engine (`backtest.py`) simulates trading strategies over historical
data. See `one_offs/d20260214_trade_signal_backtest/` for the harness that
replays model predictions over temporal snapshots.
