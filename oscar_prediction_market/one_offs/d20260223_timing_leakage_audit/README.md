# Timing Leakage Audit for Backtest Strategies

**Storage:** `storage/d20260223_timing_leakage_audit/`

This one-off isolates timing leakage analysis for the multi-category 2025 Oscar
backtest.  The core question: are headline gains overstated because model
features become available on event-date while execution prices can still reflect
pre-announcement states (or ambiguous intraday timing in a 24/7 market)?

## Motivation

The parent backtest (`d20260220_backtest_strategies`) showed outsized gains in
the Feb 7-8 precursor window (Critics Choice / DGA / PGA).  We needed to test
whether that edge persists after introducing explicit execution lag.

## Setup

- Categories: Best Picture, Directing, Actor Leading, Actress Leading,
  Actor Supporting, Actress Supporting
- Models: lr, clogit, gbt, cal_sgbt
- Lag grid: 1h, 6h, 12h, 24h
- Trading config fixed for comparability: maker, independent Kelly,
  `kf=0.25`, buy edge `>= 0.08`, YES-only, fixed bankroll

Two event-time assumptions:

1. **Inferred-time run** (`lag_audit_inferred/`): infer likely event release
   timestamp from largest cross-ticker hourly move around each snapshot date.
2. **Fixed-time run** (`lag_audit_fixed_2100/`): assume release at 21:00 ET and
   apply the same lag grid.

## Findings

### How concentrated is baseline P&L?

- Baseline synthetic snapshot P&L total: **+$13,327.23**
- Feb 7 + Feb 8 combined: **+$11,997.39** (**90.0%** of baseline)

Most apparent edge is concentrated in the same two precursor dates.

### Lag sensitivity (all category × model × snapshot scenarios)

| Lag | Inferred-time total P&L | Fixed 21:00 ET total P&L |
|-----|:-----------------------:|:------------------------:|
| 1h  | +$12,294.30 | +$7,995.85 |
| 6h  | +$5,342.56  | +$2,103.87 |
| 12h | +$9,951.46  | **−$236.42** |
| 24h | +$2,288.14  | +$637.77 |

Interpretation:

- Edge decays sharply with lag in both setups.
- Under conservative fixed-time assumptions, 12h lag can eliminate total edge.
- Inferred-time is best interpreted as an optimistic upper bound.

### Event-level lag impact (all availability events)

This section expands beyond Feb 7-8 and reports every snapshot availability
event between nominations and final precursors.

**Inferred-time run (all events):**

| Snapshot Date | Event(s) | Baseline | Lag 1h | Lag 6h | Lag 12h | Lag 24h | Baseline Share |
|---------------|----------|---------:|-------:|-------:|--------:|--------:|---------------:|
| 2025-01-23 | Oscar nominations | &minus;$392.27 | &minus;$806.44 | &minus;$869.07 | &minus;$616.74 | &minus;$904.61 | &minus;2.9% |
| 2025-02-07 | Critics Choice winner | +$2,330.20 | +$1,782.89 | +$1,865.72 | +$1,062.71 | +$955.39 | +17.5% |
| 2025-02-08 | Annie + DGA + PGA winners | +$9,667.19 | +$9,135.54 | +$1,616.11 | +$8,436.69 | +$361.81 | +72.5% |
| 2025-02-15 | WGA winner | +$200.33 | +$644.02 | &minus;$95.86 | +$166.53 | +$660.56 | +1.5% |
| 2025-02-16 | BAFTA winner | +$1,927.22 | +$1,763.23 | +$2,602.71 | +$659.23 | +$860.51 | +14.5% |
| 2025-02-23 | ASC + SAG winners | &minus;$405.44 | &minus;$224.94 | +$222.95 | +$243.04 | +$354.48 | &minus;3.0% |

**Fixed 21:00 ET run (all events):**

| Snapshot Date | Event(s) | Baseline | Lag 1h | Lag 6h | Lag 12h | Lag 24h | Baseline Share |
|---------------|----------|---------:|-------:|-------:|--------:|--------:|---------------:|
| 2025-01-23 | Oscar nominations | &minus;$392.27 | &minus;$531.35 | &minus;$897.52 | &minus;$779.72 | &minus;$933.28 | &minus;2.9% |
| 2025-02-07 | Critics Choice winner | +$2,330.20 | +$1,746.47 | +$776.44 | &minus;$127.74 | +$552.83 | +17.5% |
| 2025-02-08 | Annie + DGA + PGA winners | +$9,667.19 | +$5,711.88 | +$398.74 | +$176.55 | +$752.90 | +72.5% |
| 2025-02-15 | WGA winner | +$200.33 | +$262.22 | +$586.84 | &minus;$9.37 | &minus;$743.37 | +1.5% |
| 2025-02-16 | BAFTA winner | +$1,927.22 | +$948.65 | +$871.56 | +$265.81 | +$859.18 | +14.5% |
| 2025-02-23 | ASC + SAG winners | &minus;$405.44 | &minus;$142.02 | +$367.81 | +$238.05 | +$149.51 | &minus;3.0% |

Event-level takeaways:

- The **Feb 8 (Annie/DGA/PGA)** event remains dominant in all runs, but lagged
   outcomes vary substantially by event-time assumption.
- **BAFTA (Feb 16)** still contributes meaningful lagged edge in both setups,
   suggesting some persistence beyond the immediate DGA/PGA window.
- **WGA and SAG/ASC** windows are unstable and can flip sign depending on lag.
- Negative baseline share on Jan 23 and Feb 23 means those snapshots were net
   drag in the baseline synthetic decomposition.

### Category concentration (inferred-time run)

| Category | 1h | 6h | 12h | 24h |
|----------|---:|---:|----:|----:|
| Best Picture | +$4,082.04 | +$3,806.29 | +$3,214.11 | +$1,799.15 |
| Directing | +$9,772.88 | +$2,383.45 | +$8,642.75 | +$2,267.09 |
| Actor Leading | −$968.16 | −$734.25 | −$1,037.66 | −$1,173.70 |
| Actress Leading | −$608.20 | −$39.05 | −$930.11 | −$529.82 |
| Actor Supporting | +$8.49 | −$83.80 | −$139.67 | −$95.56 |
| Actress Supporting | +$7.25 | +$9.92 | +$202.04 | +$20.98 |

The lagged edge is still mostly Best Picture + Directing.

### Inferred event-time table

| Snapshot Date | Events | Mode Time (ET) | Min | Max | Unique Times |
|---------------|--------|:--------------:|:---:|:---:|:------------:|
| 2025-01-23 | Oscar nominations | 03:00 | 02:00 | 13:00 | 5 |
| 2025-02-07 | Critics Choice winner | 14:00 | 00:00 | 23:00 | 7 |
| 2025-02-08 | Annie winner, DGA winner, PGA winner | 22:00 | 00:00 | 22:00 | 6 |
| 2025-02-15 | WGA winner | 01:00 | 01:00 | 22:00 | 6 |
| 2025-02-16 | BAFTA winner | 04:00 | 00:00 | 23:00 | 6 |
| 2025-02-23 | ASC winner, SAG winner | 21:00 | 04:00 | 21:00 | 7 |

Wide min/max ranges show this inference is noisy and should not be treated as
true release-time ground truth.

## Recommended Backtest Framework Fixes

1. Add explicit execution timestamp policy (`event_time_basis`,
   `default_event_time_et`, `execution_lag_hours`) in configs.
2. Replace date-only daily-close joins with timestamped candle lookup at/after
   `target_ts = event_ts + lag`.
3. Separate signal timestamp from fill timestamp in the simulator.
4. Keep same-day daily-close as legacy benchmark only (not deployment default).
5. Require lag-grid sensitivity (1/6/12/24h) in every headline report.
6. Persist diagnostics: selected execution timestamp and fallback-before-target
   frequency.

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"

bash oscar_prediction_market/one_offs/d20260223_timing_leakage_audit/run.sh \
  2>&1 | tee storage/d20260223_timing_leakage_audit/run.log

uv run python -m oscar_prediction_market.one_offs.d20260223_timing_leakage_audit.analyze_event_level_lag \
   --year 2025
```

## Output Structure

```
storage/d20260223_timing_leakage_audit/
└── 2025/
   ├── analysis/
   │   ├── event_level_lag_inferred.csv
   │   ├── event_level_lag_inferred.md
   │   ├── event_level_lag_fixed_2100.csv
   │   └── event_level_lag_fixed_2100.md
    ├── lag_audit_inferred/
    │   ├── uniform_lag_audit.csv
    │   ├── uniform_lag_audit_summary.json
    │   ├── inferred_event_times.csv
    │   └── inputs_snapshot_event_times.json
    ├── lag_audit_fixed_2100/
    │   ├── uniform_lag_audit.csv
    │   ├── uniform_lag_audit_summary.json
    │   ├── inferred_event_times.csv
    │   └── inputs_snapshot_event_times.json
    ├── run_inferred.log
    └── run_fixed_2100.log
```
