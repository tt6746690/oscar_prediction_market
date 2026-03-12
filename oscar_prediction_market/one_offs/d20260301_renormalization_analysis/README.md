# Renormalization Impact on Buy-Hold Backtest

**Storage:** `storage/d20260301_renormalization_analysis/`

Does rescaling model predictions to sum to 1.0 (after name-matching filters
nominees) meaningfully change buy-hold backtest PnL?

## Motivation

When predictions are loaded for trading, name matching filters them to the
subset of nominees that exist on Kalshi. After filtering, probabilities may no
longer sum to 1.0. The refactored `oscar_prediction_source.py` removed the
explicit renormalization step that previously existed in the legacy loading path.
This analysis checks whether that removal matters.

## Setup

- **Years:** 2024, 2025 (all categories with available market data)
- **Models:** lr, clogit, gbt, cal_sgbt, avg_ensemble, clogit_cal_sgbt_ensemble
- **Config:** Single representative — Kelly f=0.20, edge threshold=0.05, maker fees, YES-only
- **Method:** For each (year, category, model, snapshot entry point), run
  buy-hold backtest twice: once with raw predictions, once with predictions
  renormalized to sum=1 after filtering. Compare aggregate PnL.

## Findings

### Multinomial models (clogit, cal_sgbt) and ensembles already sum to 1.0

| Model | Mean prob_sum | Range |
|-------|--------------|-------|
| clogit | 1.0000 | [1.0000, 1.0000] |
| cal_sgbt | 1.0000 | [1.0000, 1.0000] |
| avg_ensemble | 1.0000 | [1.0000, 1.0000] |
| clogit_cal_sgbt_ensemble | 1.0000 | [1.0000, 1.0000] |
| **gbt** | **0.8654** | [0.4719, 1.3183] |
| **lr** | **0.8651** | [0.5127, 1.2125] |

Conditional logit and calibrated softmax GBT are multinomial models that
produce properly normalized probabilities by construction. Ensembles average
these and renormalize during loading. Only LR and GBT deviate, sometimes
substantially (GBT directing 2025 averages 0.47 — nearly half the probability
mass is "missing").

### Renormalization helps GBT but hurts LR

| Model | Raw PnL | Renorm PnL | Delta | Delta % |
|-------|---------|-----------|-------|---------|
| avg_ensemble | $5,183 | $5,183 | $0 | 0.0% |
| cal_sgbt | $5,469 | $5,469 | $0 | 0.0% |
| clogit | $4,385 | $4,385 | $0 | 0.0% |
| clogit_cal_sgbt_ensemble | $5,014 | $5,014 | $0 | 0.0% |
| **gbt** | **$2,771** | **$4,487** | **+$1,716** | **+61.9%** |
| **lr** | **$4,919** | **$4,150** | **-$769** | **-15.6%** |

GBT benefits substantially from renormalization because its raw predictions
systematically underestimate total probability (mean 0.87). Without
renormalization, edges appear smaller than they are, leading to fewer trades
(75 raw vs 92 renormalized) and missed opportunities.

LR moves in the opposite direction: renormalization slightly *hurts*. This is
likely because LR's probability deviations happen to cancel out favorably
in the raw case (some categories sum > 1, inflating edges on correct picks).

### Net effect is small relative to model choice

Total PnL across all models: $27,741 (raw) vs $28,687 (renorm) — a +3.4%
difference. The gap between the best model (cal_sgbt at $5,469) and worst
(gbt-raw at $2,771) dwarfs the renormalization effect.

### Trade count diagnostic

| Model | Raw Trades | Renorm Trades |
|-------|-----------|---------------|
| lr | 64 | 109 |
| gbt | 75 | 92 |
| Others | unchanged | unchanged |

Renormalization inflates the perceived model confidence for underpredicting
models, pushing more edges above the 5% buy threshold.

## Takeaway

**Renormalization is a no-op for the models we actually use in production
(clogit, cal_sgbt, ensembles).** It only matters for LR and GBT which have
non-normalized probability outputs. Since those models are not used standalone
in the live trading pipeline, no code change is needed.

If LR or GBT were ever used standalone, renormalization after name-matching
would be advisable — but it should be applied consistently at the
`build_entry_moment` level rather than ad-hoc in callers.

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -m oscar_prediction_market.one_offs.\
d20260301_renormalization_analysis.analyze_renorm 2>&1 | tee storage/d20260301_renormalization_analysis/run.log
```

## Output Structure

```
storage/d20260301_renormalization_analysis/
├── results.csv    # Per-(year, category, model, renorm) PnL and prob_sum stats
└── run.log        # Full stdout/stderr from the analysis run
```
