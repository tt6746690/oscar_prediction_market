# Refactoring Plan: Shared Analysis Modules + Calibrated Softmax GBT

**Date:** 2026-02-18
**Context:** Two tasks on the multinomial modeling worktree:
1. Add **Calibrated Softmax GBT** as a new baseline model, rerun experiment
2. Extract reusable **analysis modules** from one-offs, backport

---

## Part 1: Calibrated Softmax GBT

### Motivation

Softmax GBT uses K×F features (one feature set per nominee slot) — with K=5 and
F=17, that's 85 input dimensions for ~25 training rows. The experiment confirmed
massive overfitting: 50.4% accuracy, 2× Brier score, and rankings that are nearly
uncorrelated with other models (Spearman ρ=0.142 minimum).

Conditional Logit solved this for LR: single shared β vector, F parameters, sum-to-1
by construction. The GBT equivalent is:

**Calibrated Softmax GBT = binary GBT (existing) + temperature-scaled softmax.**

Train a standard binary GBT on stacked nominee-level data (F features, ~250 rows).
At prediction time, convert probabilities to log-odds, then apply per-ceremony
softmax with learned temperature T:

$$\hat{p}_i = \frac{\exp(\text{logit}(p_i) / T)}{\sum_{j \in \text{ceremony}} \exp(\text{logit}(p_j) / T)}$$

This is different from naive normalization ($p_i / \sum p_j$), which the ablation
showed hurts GBT by -19.9pp. Temperature-scaled softmax operates on log-odds
(preserving the score ratios the model learned) and has a tunable parameter T
that controls sharpness.

### What's new vs. what exists

- **Binary GBT model**: already exists (`GradientBoostingConfig` / `GradientBoostingModel`)
- **Calibrated softmax post-processing**: NOT implemented. PLAN.md described it
  (Approach 1) but the experiment only tested Conditional Logit and Softmax GBT.
- **Temperature tuning**: NOT implemented.

### Implementation plan

1. **Add `CalibratedSoftmaxGBTConfig` and `CalibratedSoftmaxGBTModel` to `models.py`:**
   - Wraps an existing `GradientBoostingModel` (composition, not inheritance)
   - Config: all GBT hyperparams + `temperature: float = 1.0`
   - `fit(X, y, groups)`: fits the inner binary GBT on (X, y), then tunes T on
     training data via LOYO-CV within the training set, minimizing Brier score
   - `predict_proba(X, groups)`: gets binary GBT probabilities, converts to
     log-odds, applies per-group softmax with T
   - `get_feature_importance()`: delegates to inner GBT

   **Alternative (simpler):** Don't tune T inside `fit()` — just use a fixed T from
   config (tuned externally via the existing hyperparameter tuning infrastructure).
   This avoids nested CV inside `fit()` and keeps the model stateless.

   **Decision needed:** tune T inside fit() vs. externally? Recommend externally for
   simplicity — T becomes a config parameter like `n_estimators` or `max_depth`.

2. **Add config JSON:**
   - `modeling/configs/calibrated_softmax_gbt_standard.json` with GBT hyperparams +
     `temperature` (start with T=1.0, tune later)

3. **Add to `build_models.sh`:**
   - Build calibrated_softmax_gbt alongside the other 4 model types

4. **Update `analyze_cv.py` and `analyze_deep_dive.py`:**
   - Add `"calibrated_softmax_gbt"` to `MODEL_TYPES` list
   - All plots already iterate over model types — should work with minimal changes

5. **Rerun experiment:**
   - `build_models.sh` for the new model type (10 snapshots)
   - `analyze_cv.py` and `analyze_deep_dive.py` for updated comparison
   - Update README.md with 5-model comparison

### Open questions for Part 1

- **Temperature tuning strategy:** Should T be tuned per-snapshot (T varies over the
  season as the model's calibration changes) or a single T across all snapshots?
  Recommend single T for simplicity — one more hyperparameter to sweep.

- **Model naming:** `calibrated_softmax_gbt` is descriptive but long.
  Alternatives: `cal_gbt`, `softmax_cal_gbt`, `gbt_calibrated`. Recommend
  `calibrated_softmax_gbt` for clarity since `softmax_gbt` already exists and means
  something different (K×F multi-class XGBoost).

- **Should this be a wrapper in `models.py` or a post-processing step in
  `train_predict.py`?** Making it a proper model class is cleaner — callers don't
  need to know about calibration. The model exports probabilities that sum to ~1
  (not exactly 1 due to temperature, but close). Post-processing in train_predict
  would couple the calibration to the training pipeline.

  **Recommendation:** Model class in `models.py`. This also means the multi-category
  expansion gets it for free.

---

## Part 2: Shared Analysis Modules

### Current state — duplicated code

Three one-offs share heavily duplicated analysis code:

| One-off | Analysis files | Total lines |
|---------|---------------|-------------|
| `d20260211_temporal_model_snapshots` | `analysis.py` (11 analyses) | ~800 |
| `d20260214_trade_signal_ablation` | `analyze_ablation.py` + `analyze_deep_dive.py` | ~1500 |
| `d20260217_multinomial_modeling` | `analyze_cv.py` + `analyze_deep_dive.py` | ~2000 |

Plus `modeling/analyze_run.py` (557 lines) with single-experiment analysis.

The duplication falls into three clusters (see §Cluster detail below).

### Module location: `d20260201_oscar/analysis/` (Option A)

```
d20260201_oscar/
├── analysis/
│   ├── __init__.py
│   ├── temporal_snapshots.py   # Cluster 1: data loading + joining
│   ├── model_evaluation.py     # Cluster 2: CV metrics + calibration plots
│   └── backtest_analysis.py    # Cluster 3: trading backtest plots
├── modeling/
│   ├── analyze_run.py          # REMOVE — absorbed into analysis/model_evaluation.py
│   └── ...
├── trading/
│   └── ...
└── one_offs/
    └── ...
```

### Naming convention: category-agnostic

Per the multi-category expansion plan, all analysis code should work for any Oscar
category (Best Picture, Best Actor, Best Director, etc.) — and ideally for any
prediction market with temporal snapshots.

**Domain-agnostic terms:**

| BP-specific | Generic |
|-------------|---------|
| nominee | outcome |
| ceremony / ceremony year | group / group_id |
| awards season snapshot | snapshot_date |
| Kalshi market price | market_price |
| model probability | model_prob |
| film title | outcome_name / outcome_label |

However, fully generic naming may hurt readability for the 90% case (Oscar analysis).
**Proposed compromise:** Use `outcome` / `group` in function signatures and data
schemas, but allow Oscar-specific convenience wrappers where helpful.

**Question:** How generic do you want to go? Options:

**(A) Oscar-generic, category-agnostic:** Functions take `outcome_name`, `group_id`
(ceremony year), `snapshot_date`. Works for any Oscar category. Still has
Oscar-specific assumptions like `OscarMarket` for price loading.

**(B) Fully market-agnostic:** Functions take abstract DataFrames with columns
`outcome`, `group`, `date`, `model_prob`, `market_prob`. No Oscar imports. Could
work for any prediction market (elections, sports, etc.).

**(C) Oscar-specific but parameterized:** Functions take `category: OscarCategory`
and handle loading internally. Simplest for callers but tightly coupled.

**Recommendation:** **(A)** — Oscar-generic. The analysis functions take DataFrames
with columns like `outcome_name`, `group_id`, `snapshot_date`, `model_prob`,
`market_prob`. Loading functions know about Oscar storage layout. This aligns with the
multi-category expansion plan (same functions work for BP, Actor, Director, etc.)
without over-abstracting for markets we don't have.

### Cluster 1: `temporal_snapshots.py` — Data loading & joining

**What it does:** Loads model predictions from storage, loads market prices, joins
them into an analysis-ready DataFrame.

**Duplicated across:**
- `d20260211/analysis.py`: `load_predictions()`, `load_market_prices()`,
  `get_market_price_at_date()`, `build_model_market_df()`, `load_feature_importances()`
- `d20260214/analyze_deep_dive.py`: inline market price loading in `analyze_gbt_vs_lr()`
- `d20260217/analyze_deep_dive.py`: `load_predictions()`, `load_all_predictions()`,
  `load_market_prices()`, `get_market_prob()`, `build_model_market_df()`,
  `load_feature_importances()`

**Proposed API:**

```python
# Constants
AWARDS_SEASON_EVENTS: dict[str, str]  # date_str -> event_name

# Data loading
def load_snapshot_predictions(
    models_dir: Path,
    model_type: str,
    snapshot_dates: list[str],
) -> pd.DataFrame:
    """Load model predictions across snapshots.

    Returns DataFrame with columns:
        outcome_name, group_id, snapshot_date, model_prob, rank, is_winner
    """

def load_market_prices(
    market: OscarMarket,   # or generic market interface
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load daily market prices.

    Returns DataFrame with columns:
        outcome_name, date, market_prob
    """

def build_model_market_df(
    predictions_df: pd.DataFrame,
    market_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join model predictions with market prices at each snapshot.

    Returns DataFrame with columns:
        outcome_name, group_id, snapshot_date, model_type,
        model_prob, market_prob, divergence_pp
    """

def load_feature_importances(
    models_dir: Path,
    model_types: list[str],
    snapshot_dates: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Load feature importance per model_type × snapshot_date.

    Returns: {model_type: {date_str: {feature_name: importance}}}
    """

def load_cv_predictions(
    models_dir: Path,
    model_type: str,
    snapshot_date: str,
) -> pd.DataFrame:
    """Load LOYO cross-validation predictions for reliability diagrams.

    Returns DataFrame with columns:
        outcome_name, group_id, model_prob, is_winner
    """
```

**Key design points:**
- Functions return DataFrames with standardized column names
- Market loading is parameterized (takes a market object, not hardcoded BP)
- `snapshot_dates` are passed explicitly (not discovered from filesystem)
- No model-type-specific logic — works for any model type string

### Cluster 2: `model_evaluation.py` — CV metrics & calibration

**What it does:** Computes and visualizes model performance from CV predictions.

**Duplicated across:**
- `modeling/analyze_run.py`: `plot_calibration_curve()`, `plot_feature_importance()`,
  `plot_year_by_year_accuracy()`, `plot_rolling_performance()`,
  `plot_winner_rank_distribution()`, `plot_probability_vs_rank()`
- `d20260211/analysis.py`: `analyze_reliability()`, `analyze_brier_score()`,
  `analyze_feature_importance_evolution()`, `analyze_marginal_info()`
- `d20260217/analyze_cv.py`: `_compute_metrics_from_df()`, `plot_metrics_over_time()`,
  `plot_prob_sum_distribution()`, `plot_final_predictions()`
- `d20260217/analyze_deep_dive.py`: `plot_reliability_diagrams()`,
  `plot_feature_importance_evolution()`, `analyze_prob_concentration()`,
  `plot_metrics_over_time_detailed()`, `analyze_rank_agreement()`,
  `analyze_marginal_info()`, `analyze_top_pick_agreement()`

**Proposed API:**

```python
# Metric computation
def compute_cv_metrics(
    predictions_df: pd.DataFrame,  # outcome_name, group_id, model_prob, is_winner
) -> dict[str, float]:
    """Compute accuracy, top-3, MRR, Brier, log loss, winner prob."""

def compute_prob_concentration(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute entropy, Herfindahl, top-1/top-3 share per group."""

def compute_rank_agreement(
    predictions: dict[str, pd.DataFrame],  # model_type -> predictions
) -> pd.DataFrame:
    """Pairwise Spearman ρ between model types per snapshot."""

# Plots
def plot_reliability_diagram(
    predictions_df: pd.DataFrame,
    model_labels: list[str] | None = None,
    output_path: Path | None = None,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Calibration reliability diagram from CV predictions."""

def plot_metrics_over_time(
    metrics_by_snapshot: pd.DataFrame,  # snapshot_date, model_type, metric_name, value
    metrics: list[str] = ["accuracy", "brier_score", "log_loss", "winner_prob"],
    output_path: Path | None = None,
) -> plt.Figure:
    """Multi-panel line plot of CV metrics across snapshots."""

def plot_feature_importance_heatmap(
    importances: dict[str, dict[str, float]],  # {date: {feature: importance}}
    top_n: int = 15,
    output_path: Path | None = None,
    title: str = "Feature Importance Evolution",
) -> plt.Figure:
    """Heatmap of top features evolving across snapshots."""

def plot_feature_importance_bar(
    importance_df: pd.DataFrame,  # feature, importance
    top_n: int = 15,
    output_path: Path | None = None,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Horizontal bar chart of feature importances. Absorbs analyze_run.py's version."""

def plot_year_by_year_accuracy(
    predictions_df: pd.DataFrame,  # group_id, is_winner, model_prob, is_correct
    output_path: Path | None = None,
) -> plt.Figure:
    """Year-by-year accuracy timeline + winner probability. From analyze_run.py."""

def plot_prob_sum_distribution(
    predictions_df: pd.DataFrame,  # per-group probability sums by model type
    output_path: Path | None = None,
) -> plt.Figure:
    """Histogram of per-group probability sums per model."""

def plot_probability_vs_rank(
    predictions_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter of predicted probability vs rank. From analyze_run.py."""

def plot_prob_distribution_anatomy(
    predictions_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """3-panel: prob-sum, top-1 share, entropy over time."""
```

**Key design points:**
- All plot functions accept optional `ax` or return `Figure` for composability
- All accept optional `output_path` — if None, return figure without saving
- Metric computation is separated from visualization
- `analyze_run.py` gets deleted; its functions absorbed here

### Cluster 3: `backtest_analysis.py` — Trading backtest plots

**What it does:** Visualizes trading backtest results — wealth curves, positions,
settlements, edges, model-vs-market comparisons.

**Duplicated across:**
- `d20260211/analysis.py`: `plot_model_vs_market()`, `plot_divergence_heatmap()`,
  `analyze_trading_signals()`, `analyze_market_blend()`
- `d20260214/analyze_deep_dive.py`: `plot_config_deep_dive()` (4-panel: wealth,
  positions, model-vs-market, fees), `analyze_gbt_vs_lr()` (edge histograms)
- `d20260217/analyze_deep_dive.py`: `plot_model_vs_market()`,
  `plot_divergence_heatmaps()`, `plot_wealth_curves_annotated()`,
  `plot_settlement_heatmap()`, `plot_position_evolution()`,
  `plot_edge_distributions()`, `plot_binary_vs_multinomial()`

**Proposed API:**

```python
# Model vs market
def plot_model_vs_market(
    model_market_df: pd.DataFrame,
    # columns: outcome_name, snapshot_date, model_type, model_prob, market_prob
    outcome_names: list[str] | None = None,  # subset to plot
    model_types: list[str] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Per-outcome subplot: model prob lines vs market price line."""

def plot_divergence_heatmap(
    model_market_df: pd.DataFrame,
    model_type: str | None = None,  # single model, or multi-panel if None
    output_path: Path | None = None,
) -> plt.Figure:
    """Outcome × date heatmap of model−market divergence (pp)."""

# Trading
def plot_wealth_curves(
    backtest_results: dict,  # from run_ablation output or similar
    annotate_trades: bool = True,
    output_path: Path | None = None,
) -> plt.Figure:
    """Wealth curve(s) over time, optionally annotated with trade counts."""

def plot_settlement_heatmap(
    settlements: dict,  # model_type -> {possible_winner -> return_pct}
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of return % for each model × possible winner."""

def plot_position_evolution(
    position_history: pd.DataFrame,
    # columns: date, outcome_name, contracts, cost_basis
    output_path: Path | None = None,
    title: str = "Position Evolution",
) -> plt.Figure:
    """Stacked area chart of positions held over time."""

def plot_edge_distributions(
    edges_df: pd.DataFrame,
    # columns: outcome_name, snapshot_date, model_type, net_edge
    model_types: list[str] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Edge histograms per model type."""

# Composite
def plot_backtest_deep_dive(
    backtest_results: dict,
    model_market_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """4-panel deep dive: wealth curve, positions, model-vs-market, fees.
    Absorbs d20260214's plot_config_deep_dive()."""
```

**Key design points:**
- Input is DataFrames with standard column names (from Cluster 1 loaders)
- `backtest_results` dict matches the output of `run_ablation.py` — the existing
  structure becomes the de facto schema
- All functions are stateless — no market loading inside

### Module interaction diagram

```
one_offs/d20260217_multinomial/analyze_*.py
one_offs/d20260214_trade_signal/analyze_*.py
one_offs/d20260211_temporal/analysis.py
modeling/analyze_run.py (REMOVE)
         │
         │ calls
         ▼
analysis/
├── temporal_snapshots.py  ──► OscarMarket (trading/)
│   (load data, join)          storage dirs
│
├── model_evaluation.py    ──► pure computation + matplotlib
│   (CV metrics, calibration)
│
└── backtest_analysis.py   ──► pure matplotlib
    (wealth, positions, edges)
```

### Backporting plan

After the shared modules are implemented:

| One-off | Refactoring scope | Approach |
|---------|-------------------|----------|
| **d20260217_multinomial_modeling** | `analyze_cv.py` + `analyze_deep_dive.py` | Refactor first (we're on this worktree). Replace data loading with Cluster 1, metric computation with Cluster 2, backtest plots with Cluster 3. One-off retains multinomial-specific analyses (binary-vs-multinomial comparison, model agreement tables). |
| **d20260214_trade_signal_ablation** | `analyze_ablation.py` + `analyze_deep_dive.py` | Second priority. Replace wealth curves, settlement, edge plots with Cluster 3. `analyze_ablation.py` (parameter sensitivity) stays one-off-specific. `run_ablation.py` stays in one-off (per user decision). |
| **d20260211_temporal_model_snapshots** | `analysis.py` | Third. Thin wrapper over Cluster 1 + 2 + 3. Most of its 11 analyses map directly to shared functions. |
| **modeling/analyze_run.py** | Remove entirely | All 6 plots absorbed into `model_evaluation.py`. Update `modeling/__init__.py` or callers if any. |

### Backporting sequence

1. Build the three shared modules with tests
2. Refactor d20260217 (current worktree — we're already modifying it for calibrated softmax GBT)
3. Merge d20260217 to main
4. Rebase multi-category-expansion worktree onto main (gets the shared modules)
5. Backport d20260214 and d20260211 (can be done on main or a dedicated worktree)

### What stays in one-offs

Each one-off retains analysis that is genuinely experiment-specific:

- **d20260217:** Binary-vs-multinomial side-by-side, model agreement matrix,
  top-pick agreement table, Softmax GBT failure anatomy
- **d20260214:** Parameter sensitivity analysis, interaction heatmaps, config grid
  ranking, normalization ablation comparison
- **d20260211:** Event-impact delta analysis, market blend α sweep

### Relationship to multi-category expansion

The multi-category expansion (PLAN.md on that worktree) will benefit directly:

- **Agent A/B** (new category models) use `model_evaluation.py` for CV analysis
  and `temporal_snapshots.py` for snapshot loading — no copy-paste
- **Agent C** (trading infrastructure) uses `backtest_analysis.py` for per-category
  backtest visualization
- `temporal_snapshots.py` is parameterized by `OscarMarket` (not hardcoded BP) —
  works for any category's event ticker
- Feature importance, calibration, metrics-over-time plots all work unchanged
  regardless of category

---

## Part 3: Implementation Sequence

### Phase 1: Calibrated Softmax GBT model (code)

1. Add `CalibratedSoftmaxGBTConfig` + `CalibratedSoftmaxGBTModel` to `models.py`
2. Add model config JSON
3. Add to build_models.sh
4. Build 10 snapshots
5. Verify with analyze_cv.py (add to MODEL_TYPES)

### Phase 2: Shared analysis modules (extract + test)

1. Create `analysis/__init__.py`
2. Extract `temporal_snapshots.py` from d20260217 + d20260211 loaders
3. Extract `model_evaluation.py` from d20260217 + analyze_run.py
4. Extract `backtest_analysis.py` from d20260217 + d20260214
5. Add tests for non-trivial computation functions (metric aggregation, joins)

### Phase 3: Refactor d20260217 (current worktree)

1. Replace data loading in analyze_cv.py with temporal_snapshots calls
2. Replace metric computation with model_evaluation calls
3. Replace backtest plots in analyze_deep_dive.py with backtest_analysis calls
4. Keep experiment-specific code in the one-off scripts
5. Rerun full analysis, verify outputs match

### Phase 4: Update README with calibrated softmax GBT results

1. Add 5th model to all comparison tables
2. Update findings sections
3. Regenerate plots

### Phase 5: Backport other one-offs

1. Remove `modeling/analyze_run.py`, update any references
2. Refactor d20260214 to use shared modules
3. Refactor d20260211 to use shared modules

---

## Open Questions

1. **CalibratedSoftmaxGBT: tune T inside fit() or externally?**
   Recommendation: externally (T in config JSON). Simpler, no nested CV.
   But this means we need to sweep T values — add to hyperparameter tuning
   infrastructure? Or just test T ∈ {0.5, 0.75, 1.0, 1.25, 1.5} manually?

2. **Generic naming granularity:** The proposed APIs use `outcome_name` and
   `group_id`. Should `snapshot_date` be renamed to just `date` or `timestamp`?
   The term "snapshot" implies temporal model snapshots specifically — but that IS
   the primary use case. Keeping `snapshot_date` for now; can generalize later.

3. **Plot style consistency:** The three one-offs use slightly different matplotlib
   rcParams, color palettes, figure sizes. Should the shared module enforce a
   consistent style? Recommend yes — define a `set_analysis_style()` function
   that sets rcParams, and a `MODEL_COLORS` dict mapping model types to colors.

4. **How to handle `AWARDS_SEASON_EVENTS`?** Currently each one-off has its own
   dict mapping dates to event names. For multi-category, events differ by category
   (SAG Ensemble is a BP precursor, SAG Lead Actor is an Actor precursor). Should
   this be loaded from the `AwardsCalendar` config, or kept as a display-only
   constant?
   Recommendation: Display-only constant in `temporal_snapshots.py` for common
   events. Category-specific events passed by the caller.

5. **Test strategy for plotting functions:** Pure computation functions
   (metric aggregation, joins) get proper unit tests. Plotting functions are
   harder to test — use smoke tests (call with synthetic data, check figure is
   returned without error, but don't assert pixel values). Worth it?

6. **analyze_run.py removal timing:** Should we remove it in this PR (d20260217)
   or defer to a separate cleanup PR? Removing in this PR means the multi-category
   expansion worktree needs to rebase. If it's already diverged significantly,
   a separate PR might be cleaner.
