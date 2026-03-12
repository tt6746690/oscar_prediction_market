# Plan: ProbabilitySource Protocol + Run Script Consolidation + README Tooling

**Date:** 2026-02-25
**Branch:** `feature/backtest-strategies`
**Status:** Planning

---

## Motivation

Three independent but related problems converge:

1. **Ensemble duplication** — `run_ensemble_only.py` (341 lines) is ~90%
   copy-paste of the ensemble block in `run_backtests.py` (lines 468–634).
   Any change to the backtest pipeline must be made in two places.

2. **No prediction abstraction** — predictions enter `BacktestEngine` as
   `dict[str, float]` inside `TradingMoment`. The caller (one-off script)
   is responsible for loading CSV files, averaging across models (ensemble),
   doing name matching, and injecting predictions into moments. This logic
   is duplicated across run scripts and makes it impossible to swap
   prediction sources (e.g., single model vs ensemble vs external API)
   without rewriting the entire data flow.

3. **README maintenance** — result tables in the README are manually crafted
   from CSV files. After re-running backtests with new configs, the README
   becomes stale. A `generate_readme_tables.py` script would automate this.

The fix: introduce a `ProbabilitySource` protocol at the **trading module
level**, then refactor callers to use it. This is an investment in the core
library that pays off in every future one-off.

---

## Design: ProbabilitySource Protocol

### Core Abstraction

```python
# trading/probability_source.py (NEW)

from typing import Protocol


class ProbabilitySource(Protocol):
    """Provides model probabilities for a set of outcomes at a point in time.

    Why a protocol: the backtest engine shouldn't care whether predictions
    come from a single model's CSV, an equal-weight ensemble, a time-weighted
    ensemble, or a live API call. The protocol captures the minimal contract:
    given a snapshot date, return {outcome: probability}.
    """

    @property
    def short_name(self) -> str:
        """Human-readable identifier (e.g., 'lgbm', 'ensemble_4')."""
        ...

    def get_predictions(self, snapshot_date: date) -> dict[str, float]:
        """Return {outcome_name: model_probability} for the given snapshot.

        Raises KeyError if no predictions available for this date.
        """
        ...

    def available_dates(self) -> list[date]:
        """Return all dates for which predictions exist, sorted ascending."""
        ...
```

### Concrete Implementations

```python
# trading/probability_source.py (continued)

class SingleModelSource:
    """Loads predictions from a single model's snapshot CSVs.

    Example::

        source = SingleModelSource(
            predictions_by_date={"2026-02-01": {"Anora": 0.35, ...}},
            model_name="lgbm",
        )
        source.get_predictions(date(2026, 2, 1))
        # => {"Anora": 0.35, "The Brutalist": 0.20, ...}
    """

    def __init__(
        self,
        predictions_by_date: dict[str, dict[str, float]],
        model_name: str,
    ) -> None: ...


class EnsembleSource:
    """Equal-weight average across multiple ProbabilitySources.

    Example::

        source = EnsembleSource(
            sources=[lgbm_source, xgb_source, rf_source, logreg_source],
            name="ensemble_4",
        )
        # get_predictions averages across all sources for each outcome
    """

    def __init__(
        self,
        sources: list[ProbabilitySource],
        name: str = "ensemble",
    ) -> None: ...

    def get_predictions(self, snapshot_date: date) -> dict[str, float]:
        """Average predictions across all sources, only for dates all share."""
        ...
```

### Integration Point: build_trading_moments

Currently `build_trading_moments()` in the experiment's `data_prep.py` takes
raw `dict[str, dict[str, float]]` (date_str → outcome → prob). It should
accept a `ProbabilitySource` instead:

```python
def build_trading_moments(
    source: ProbabilitySource,
    prices_by_date: dict[str, dict[str, float]],
    signal_delay: SignalDelayConfig,
    ...
) -> list[TradingMoment]:
    """Build TradingMoments by querying source for each available date."""
```

This means `TradingMoment.predictions` stays as `dict[str, float]` — the
protocol resolves predictions *before* they enter the engine. The engine
itself doesn't change.

---

## Phase 1: ProbabilitySource Protocol (trading module)

### 1.1 Create `trading/probability_source.py`

- `ProbabilitySource` protocol (as above)
- `SingleModelSource` concrete class (wraps `dict[str, dict[str, float]]`)
- `EnsembleSource` concrete class (aggregates multiple sources)
- Full docstrings with Example:: blocks
- Tests in `tests/trading/test_probability_source.py`

### 1.2 Update `build_trading_moments()` signature

The *library-level* `build_trading_moments` in `trading/backtest.py` (if it
exists there) or create a new function that accepts `ProbabilitySource`.
The experiment-level `data_prep.py` can keep its current interface as a
thin wrapper that constructs a `SingleModelSource` internally.

### 1.3 Export from `trading/__init__.py`

Add `ProbabilitySource`, `SingleModelSource`, `EnsembleSource` to exports.

**Tests:** Unit tests for `SingleModelSource`, `EnsembleSource` (averaging
correctness, missing-date handling, available_dates intersection).

---

## Phase 2: Consolidate Run Scripts

### 2.1 Extract `run_single_backtest()` helper

Factor out the shared core from both the individual-model loop body and the
ensemble block in `run_backtests.py`:

```python
def run_single_backtest(
    source: ProbabilitySource,
    category: str,
    prices_by_date: dict[str, dict[str, float]],
    winner: str,
    nominee_map: dict[str, str],
    spread_penalties: dict[str, float],
    signal_delay: SignalDelayConfig,
    config_grid: list[TradingConfig],
    sim_config: SimulationConfig,
) -> list[dict]:
    """Run backtest for one source across the full config grid.

    Returns list of result row dicts (one per config in grid).
    """
```

This eliminates ~80 lines of duplication per call site.

### 2.2 Refactor `run_backtests.py` to use `run_single_backtest`

The individual model loop becomes:

```python
for model_type in model_types:
    source = SingleModelSource(
        predictions_by_date=load_all_snapshot_predictions(category, model_type, ...),
        model_name=model_type,
    )
    rows += run_single_backtest(source, category, ...)
```

The ensemble block becomes:

```python
if run_ensemble:
    ensemble_source = EnsembleSource(
        sources=[SingleModelSource(...) for mt in model_types],
        name=ENSEMBLE_SHORT_NAME,
    )
    rows += run_single_backtest(ensemble_source, category, ...)
```

### 2.3 Simplify or remove `run_ensemble_only.py`

After the refactor, `run_ensemble_only.py` becomes a thin script that:
1. Constructs an `EnsembleSource`
2. Calls `run_single_backtest()`
3. Does CSV append logic

From ~341 lines to ~80 lines. Or remove entirely if `run_backtests.py`
already handles ensemble via `--ensemble` flag.

### 2.4 Move shared display dicts to experiment `__init__.py`

`MODEL_DISPLAY` and `CATEGORY_DISPLAY` in `analyze.py` are also needed by
`analyze_robustness.py` and potentially `generate_readme_tables.py`. Move to
the experiment's `__init__.py` or a dedicated `constants.py`.

Note: the one-off's display dicts are intentionally different from
`analysis/style.py` (which covers the full project). Keep them separate
but in a shared location within the one-off.

---

## Phase 3: README Tooling

### 3.1 Create `generate_readme_tables.py`

Script that reads the backtest CSV and generates markdown tables for the
README. Outputs to stdout so it can be piped or copy-pasted.

```bash
uv run python -m oscar_prediction_market.one_offs.\
    d20260220_backtest_strategies.generate_readme_tables \
    --csv storage/d20260220_backtest_strategies/results_inferred_6h.csv
```

Features:
- Group by category, then model
- Highlight best P&L per category
- Include config columns (kelly_fraction, buy_edge_threshold, etc.)
- Format currency values, percentages
- Output markdown directly pasteable into README.md

### 3.2 Add context/methodology section to README

Currently the README jumps straight into results. Add a section explaining:
- What the backtest simulates (rebalancing mode, daily decisions)
- Config grid dimensions and what they mean
- How to interpret P&L, ROI, max drawdown columns
- Signal delay modes and why inferred+6h is canonical

---

## Phase 4: Apply analyze_robustness.py mypy fixes

The same mypy fixes applied on main (commit `ea064d7`) need to be applied
here. Options:

- **Cherry-pick** from main: `git cherry-pick ea064d7` — but this also brings
  the `d20260225_buy_hold_backtest` one-off which doesn't belong here.
- **Manual replay**: Apply the same typed-constant pattern. The file on this
  branch may have diverged slightly from main.
- **Merge main**: If the branch is ready to absorb main's changes.

Recommendation: check if the file has diverged (`git diff main -- <path>`),
then either cherry-pick with `--no-commit` + selective staging, or manually
apply the same fixes.

---

## Dependency Graph

```
Phase 1.1 (protocol)
    │
    ├─→ Phase 1.2 (build_trading_moments update)
    │       │
    │       └─→ Phase 2.1 (extract run_single_backtest)
    │               │
    │               ├─→ Phase 2.2 (refactor run_backtests.py)
    │               └─→ Phase 2.3 (simplify run_ensemble_only.py)
    │
    └─→ Phase 1.3 (exports)

Phase 2.4 (display dicts) — independent
Phase 3.1 (generate_readme_tables) — depends on 2.4 for display dicts
Phase 3.2 (README context) — independent
Phase 4 (mypy fixes) — independent
```

---

## Entry Points Summary

| Phase | Files Created/Modified | Risk |
|-------|----------------------|------|
| 1.1 | `trading/probability_source.py` (NEW), `tests/trading/test_probability_source.py` (NEW) | Low — additive |
| 1.2 | `trading/backtest.py` or `data_prep.py` | Medium — interface change |
| 1.3 | `trading/__init__.py` | Low |
| 2.1 | `run_backtests.py` (extract helper) | Medium — refactor |
| 2.2 | `run_backtests.py` (use helper) | Medium |
| 2.3 | `run_ensemble_only.py` (simplify/remove) | Low |
| 2.4 | `__init__.py` or `constants.py`, `analyze.py`, `analyze_robustness.py` | Low |
| 3.1 | `generate_readme_tables.py` (NEW) | Low — additive |
| 3.2 | `README.md` | Low |
| 4 | `analyze_robustness.py` | Low |

---

## Success Criteria

- [ ] `ProbabilitySource` protocol exists with tests
- [ ] `SingleModelSource` and `EnsembleSource` pass unit tests
- [ ] `run_backtests.py` uses `run_single_backtest()` for both individual and ensemble
- [ ] `run_ensemble_only.py` reduced to <100 lines or removed
- [ ] `generate_readme_tables.py` produces correct markdown from CSV
- [ ] `make dev` passes (format, lint, typecheck, test)
- [ ] Backtest results are identical before/after refactor (CSV diff)
