# Multinomial Modeling — From Independent Binary to Constrained Probabilities

**Date:** 2026-02-17
**Storage:** `storage/d20260217_multinomial_modeling/`
**Worktree:** `.worktrees/feature/d20260217_multinomial_modeling`

## Problem Statement

The current pipeline models each nominee as an **independent Bernoulli** trial:
$P(\text{win}_i | x_i)$ is estimated separately for each nominee $i$, with no
constraint that $\sum_i P(\text{win}_i) = 1$ within a ceremony year. This is
structurally wrong — exactly one nominee wins.

### Empirical severity

Analyzing the 10 temporal model snapshots from the trade signal ablation:

| Model | Mean $\sum P$ | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| **LR**  | **0.737** | 0.435 | 0.328 | 1.530 |
| **GBT** | **1.219** | 0.122 | 0.918 | 1.363 |

LR is severely **under-distributed** (mean sum 0.74, minimum 0.33 at Jan 22 when
the model is most confident). GBT **over-distributes** slightly (mean 1.22). Neither
is close to the required 1.0, and both vary wildly across snapshots.

### Consequences

**Example: LR at Feb 7 (final snapshot, sum = 0.57)**

| Nominee | Raw P | Normalized P | Market |
|---------|-------|-------------|--------|
| One Battle after Another | 25.9% | 45.5% | 70% |
| Marty Supreme | 9.8% | 17.3% | 3% |
| Sinners | 6.5% | 11.3% | 18% |
| Hamnet | 5.6% | 9.9% | 2% |
| ... | ... | ... | ... |

LR's raw 25.9% on the frontrunner implies a massive −44pp "edge" vs market 70%.
But the model is really saying "25.9% out of the 57% probability mass it distributes,
not out of 100%." Naive normalization says 45.5%, still far from 70% but less
catastrophically wrong.

**GBT at Jan 22 (sum = 1.36)**

| Nominee | Raw P | Normalized P | Market |
|---------|-------|-------------|--------|
| One Battle after Another | 76.2% | 55.9% | 54% |
| Sinners | 14.8% | 10.9% | 18% |

Normalization *deflates* GBT's confident frontrunner estimate from 76% → 56%,
losing real signal. This explains why normalization hurts GBT trading (−19.9pp).

### Trading backtest impact (from ablation)

| Model | Normalize | Return | Trades | Fees |
|-------|-----------|--------|--------|------|
| GBT | False (raw) | **+23.5%** | 28 | $30.47 |
| GBT | True (norm) | +3.6% | 18 | $14.14 |
| LR | False (raw) | +2.5% | 14 | $17.18 |
| LR | True (norm) | **+6.1%** | 24 | $27.73 |

Normalization helps LR (+3.6pp) by concentrating its diffuse predictions. It
catastrophically hurts GBT (−19.9pp) because GBT already over-distributes slightly —
normalization deflates the high-conviction frontrunner call that drives all the
profitable trades.

**Root cause:** Post-hoc normalization is a band-aid. It preserves ranking but
distorts the *magnitudes* that drive edge computation. The right fix is to train
models that output properly constrained probabilities natively.

---

## Approaches

### Approach 1: Calibrated Softmax Post-Processing

Instead of training a new model architecture, apply **temperature-scaled softmax**
to existing model outputs:

$$\hat{p}_i = \frac{\exp(s_i / T)}{\sum_{j \in \text{year}} \exp(s_j / T)}$$

where $s_i$ is the model's raw score (log-odds for LR, logit-transformed
`predict_proba` for GBT) and $T$ is a temperature parameter.

- $T = 1$: standard softmax
- $T > 1$: more uniform (less confident)
- $T < 1$: more peaked (more confident)

**Tuning $T$:** Cross-validation over historical years, minimizing Brier score or
log-loss of the calibrated probabilities. With ~25 years, we have ~25 data points to
tune a single scalar — feasible.

**Feasibility analysis:** Temperature scaling has exactly **one parameter** ($T$),
so the risk of overfitting with 25 years of data is low. Platt scaling (2 params)
would also work. The key question is whether the *shape* of the miscalibration is
consistent across years — if sometimes the model is overconfident and sometimes
underconfident, a fixed $T$ won't help. Empirically, LR is consistently under-distributed
(sum < 1), suggesting a consistent $T < 1$ would help. GBT varies more (0.92 to 1.36),
so a fixed $T$ may not capture the variation.

**Important:** For LR, operating on **log-odds** (the linear predictor $x_i^\top \beta$)
before softmax is equivalent to running conditional logit with a temperature parameter. So
calibrated softmax on LR log-odds ≈ conditional logit with temperature. For GBT, we can
extract log-odds via `log(p / (1-p))` from the binary probabilities.

| Pros | Cons |
|------|------|
| Minimal code change — post-processing only | Model still trained independently (doesn't learn competition) |
| One parameter to tune | Fixed $T$ assumes consistent miscalibration pattern |
| Works with any model type | Doesn't improve the model itself — just rescales output |
| Easy to integrate with existing pipeline | Sensitive to outlier probabilities near 0 or 1 (log-odds explode) |
| Good baseline to compare against approaches 2-3 | |

**Integration plan:** Add a `calibrate_probabilities(probs, ceremony_groups, T)`
function. Can be applied in `train_predict.py` after `model.predict_proba()` and before
generating `YearPrediction`. Tune $T$ inside `evaluate_cv.py`'s CV loop.

### Approach 2: Conditional Logistic Regression (McFadden's Choice Model)

The principled econometric approach. Models the choice among $K$ alternatives
(nominees) within each choice set (ceremony year):

$$P(\text{win}_i | \text{year}) = \frac{\exp(x_i^\top \beta)}{\sum_{j \in \text{year}} \exp(x_j^\top \beta)}$$

This is softmax over nominee-specific features. Probabilities sum to 1 by
construction. Each nominee's features compete directly against the field.

**Why not plain sklearn multinomial LR?** Plain `LogisticRegression(multi_class="multinomial")`
assumes shared feature vectors and fixed class labels across observations. Our problem
has **per-alternative features** (each nominee has its own metacritic, nominations, etc.)
and the "classes" (nominees) change every year. This is a **conditional logit** /
**discrete choice** problem, not a standard multiclass problem.

Sklearn's multinomial LR learns one weight vector *per class*. Conditional logit
learns one shared weight vector $\beta$ applied to per-alternative features. The
difference:

- Multinomial LR: $P(y=k | x) = \frac{\exp(x^\top w_k)}{\sum_j \exp(x^\top w_j)}$
  — learns $K$ weight vectors, needs fixed $K$ classes
- Conditional logit: $P(y=i | \text{choice set}) = \frac{\exp(x_i^\top \beta)}{\sum_{j} \exp(x_j^\top \beta)}$
  — learns 1 weight vector, flexible choice set size

Our features are nominee-specific (metacritic, nominations, etc.), so conditional
logit is the correct formulation. It handles variable numbers of nominees per year
naturally.

**Implementation:** `statsmodels.discrete.conditional_models.ConditionalLogit` with
`fit_regularized(method='elastic_net')` for regularization.

**statsmodels API details** (confirmed from source code):

```python
from statsmodels.discrete.conditional_models import ConditionalLogit

# Constructor: groups is required keyword
model = ConditionalLogit(endog=y, exog=X, groups=ceremony_ids)

# Regularized fit: elastic net with L1/L2 control
result = model.fit_regularized(
    method='elastic_net',
    alpha=0.1,           # overall penalty (scalar or per-coef array)
    L1_wt=0.5,           # 0=pure L2, 1=pure L1
    start_params=None,
    refit=False,
    maxiter=50,
)

# Prediction: returns LINEAR PREDICTOR (X @ params), NOT probabilities
linear_pred = result.predict(exog=X_test)  # or X_test @ result.params

# Must compute softmax manually per group:
for ceremony in np.unique(groups_test):
    mask = groups_test == ceremony
    logits = linear_pred[mask]
    logits -= logits.max()  # numerical stability
    probs[mask] = np.exp(logits) / np.exp(logits).sum()
```

**Key requirements:**
- `endog` must be 0/1 (winner/loser) — same as current `y`
- No intercept (absorbed by conditional likelihood)
- With exactly 1 winner per group, conditional likelihood = softmax
- Feature scaling still recommended (regularization is scale-sensitive)

| Pros | Cons |
|------|------|
| Theoretically correct formulation | Effective N = ~25 years (not ~250 rows) |
| Probabilities sum to 1 by construction | `predict()` returns linear predictor — manual softmax needed |
| Handles variable nominee count | Need to pass `groups` through pipeline |
| Single $\beta$ vector — interpretable | |
| Well-studied in econometrics | |
| `fit_regularized` supports elastic net — no custom MLE needed | |

**Integration plan:** New model class `ConditionalLogitModel(PredictionModel)` in
`models.py`. The `fit` method takes grouped data; `predict_proba` computes softmax
from linear predictor within groups, outputting probabilities that sum to 1 by
construction. Feature engineering unchanged — same per-nominee features.

### Approach 3: XGBoost/GBT with Softmax Objective

Use `XGBClassifier(objective="multi:softprob")` where each ceremony is a multi-class
classification problem with $K$ classes (nominees).

**Variable nominee count:** Fix $K = \min(\text{nominees across years})$ so every
year naturally fills all $K$ slots. The pre-trained binary model (already on disk)
ranks all nominees; top-$K$ enter the multi-class problem.

**Two-stage approach:**
1. Binary model (on disk) ranks all nominees per ceremony
2. Top-$K$ nominees per year enter the multi-class model
3. Multi-class softmax produces calibrated probabilities summing to 1

**Data format:** Each year becomes one row with $K \times F$ features (features for
each nominee concatenated). XGBoost's `multi:softprob` outputs a $K$-vector of
probabilities per observation.

| Pros | Cons |
|------|------|
| Probabilities sum to 1 by construction | $K \times F$ feature space; overfitting risk with ~25 observations |
| XGBoost handles feature interactions naturally | Two-stage: depends on binary model's ranking quality |
| Native XGBoost objective — well-optimized | Ordering of nominees in feature vector is arbitrary — need symmetry handling |
| Non-linear; captures frontrunner dynamics | Feature importance harder to interpret |
| Can leverage existing hyperparameter tuning | |

**Key design decision:** Sort nominees by precursor signal strength when assigning
to feature slots. Slot 1 = strongest nominee, slot 2 = second strongest, etc. Each
slot gets the same feature set. The model learns "slot 1 features predict class 0
(slot 1 wins)."

---

## Implementation Plan

### Phase 1: Diagnostic Measurement (prereq for all approaches)

**Goal:** Establish baseline metrics for probability calibration quality, and add
probability-sum tracking to the CV evaluation pipeline.

1. **Add probability-sum metrics to `evaluation.py`:**
   - `prob_sum_per_year(predictions)`: returns list of per-year probability sums
   - `prob_sum_stats(predictions)`: mean, std, min, max of probability sums
   - Include in `EvaluationMetrics` output from `evaluate_cv.py`
   - Add category-specific calibration metrics (e.g., per-probability-bin accuracy
     within each ceremony year, reliability diagrams)

2. **Run existing CV with probability-sum tracking** — quantify how LR vs GBT
   probability sums vary across ceremony years and hyperparameter configs

3. **Measure calibrated softmax as a quick baseline:**
   - Extract log-odds from existing CV predictions
   - Grid search $T \in [0.1, 0.2, ..., 2.0]$ using LOYO-CV
   - Measure Brier score, accuracy, and probability sums

### Phase 2: Calibrated Softmax (Approach 1)

**Goal:** Implement temperature-scaled softmax as a post-processing step.

1. **Add `SoftmaxCalibrator` class** — fits temperature $T$ on training data,
   transforms probabilities on test data. Must handle grouping by ceremony.

2. **Integrate into `evaluate_cv.py`** — option to apply calibration after
   model prediction within each CV fold (tune $T$ on train folds, apply to test fold).

3. **Integrate into `train_predict.py`** — apply calibration when generating
   predictions for deployment.

4. **Backtest:** Re-run trading backtest with calibrated probabilities. Compare
   against raw and naively-normalized baselines.

### Phase 3: Conditional Logistic Regression (Approach 2)

**Goal:** Implement conditional logit as a new model type.

**Priority: this is the primary modeling improvement.** Calibrated softmax (Phase 2)
is a quick baseline for comparison, but conditional logit is the principled fix.

1. **Add `ConditionalLogitConfig` and `ConditionalLogitModel` to `models.py`:**
   - Uses `statsmodels.ConditionalLogit` with `fit_regularized(method='elastic_net')`
   - Config params: `alpha` (penalty strength), `L1_wt` (L1 vs L2 mix)
   - `fit(X, y, groups)` — needs ceremony grouping
   - `predict_proba(X, groups)` — computes softmax from linear predictor per group

2. **Interface change (Option A — groups in ABC):**
   - Add `groups: np.ndarray | None = None` to `PredictionModel.fit()` and
     `predict_proba()` — backward compatible, binary models ignore it
   - At call sites that train: change `_` → `meta` and pass
     `groups=meta["ceremony"].values`. ~8 training sites, ~10 prediction sites.
   - No change to `prepare_model_data()` — ceremony already in metadata.

3. **Feature engineering** — same features as LR. Feature interactions deferred
   to later ablation study.

4. **Hyperparameter tuning** — `alpha` (penalty) and `L1_wt` (L1/L2 mix) via
   existing nested CV infrastructure. Feature scaling via `StandardScaler`
   (needed because regularization penalty is scale-sensitive).

5. **Compare against binary LR + calibrated softmax** — if conditional logit
   doesn't improve over calibrated softmax, the training-time constraint doesn't
   add value and the simpler approach wins.

### Phase 4: Softmax GBT/XGBoost (Approach 3)

**Goal:** Multi-class tree model with fixed K.

1. **Pre-filter to top-K nominees** — use binary model (already on disk) to rank
   nominees. $K = \min(\text{nominees across all years})$ so no padding needed.

2. **Data format (stacked):** Each year = 1 row with $K \times F$ features.
   Sort nominees by precursor signal, assign to slots. XGBoost `multi:softprob`
   with $K$ classes.

3. **Evaluate** — compare against binary GBT + calibrated softmax. If the
   multi-class model doesn't improve, the overhead isn't justified.

### Phase 5: Trading Integration

1. **Trading integration:** Calibrated probabilities feed directly into edge
   computation. No normalization needed. The multi-outcome Kelly optimizer
   will receive consistent probabilities (sum ≈ 1).

2. **Full backtest comparison:** Raw binary → naively normalized → calibrated
   softmax → conditional logit → softmax GBT. Across all trading parameter configs.

---

## Interface Change Detail

### Current pattern (18 call sites)

```python
# Training (8 sites) — metadata discarded:
X_train, y_train, _ = prepare_model_data(train_df, feature_set)
model.fit(X_train, y_train)

# Prediction (10 sites) — metadata used for film_id/title:
X_test, y_test, meta = prepare_model_data(test_df, feature_set)
probs = model.predict_proba(X_test)
```

### Proposed change (Option A: groups in ABC)

```python
# models.py — backward-compatible optional param
class PredictionModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series,
            groups: np.ndarray | None = None) -> "PredictionModel": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame,
                      groups: np.ndarray | None = None) -> np.ndarray: ...

# Existing models — add groups param, ignore it:
class LogisticRegressionModel(PredictionModel):
    def fit(self, X, y, groups=None):  # groups unused
        ...
    def predict_proba(self, X, groups=None):  # groups unused
        ...

# New model — requires groups:
class ConditionalLogitModel(PredictionModel):
    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("ConditionalLogitModel requires groups")
        ...
    def predict_proba(self, X, groups=None):
        if groups is None:
            raise ValueError("ConditionalLogitModel requires groups")
        ...
```

### Call site changes

```python
# Training site — change _ to meta, pass groups:
X_train, y_train, meta_train = prepare_model_data(train_df, feature_set)
model.fit(X_train, y_train, groups=meta_train["ceremony"].values)

# Prediction site — pass groups:
X_test, y_test, meta_test = prepare_model_data(test_df, feature_set)
probs = model.predict_proba(X_test, groups=meta_test["ceremony"].values)
```

No change to `prepare_model_data()` — ceremony is already in metadata.
Binary models ignore the `groups` arg. ~18 call sites need mechanical update.

---

## Integration Points with Existing Code

### `models.py` — New model types

- `ConditionalLogitConfig` + `ConditionalLogitModel` as new discriminated union variant
- `SoftmaxGBTConfig` + `SoftmaxGBTModel` wrapping multi-class XGBoost
- Both implement `PredictionModel` interface: `fit()`, `predict_proba()`, `get_feature_importance()`
- `predict_proba()` returns probabilities that sum to 1 per ceremony group

### `evaluate_cv.py` — CV with group-aware models

- Pass ceremony labels to `fit()` / `predict_proba()` via groups arg
- Add `SoftmaxCalibrator` option: tune $T$ within CV folds
- Report probability-sum metrics and category-specific calibration metrics

### `train_predict.py` — Production predictions

- Support new model types via existing `ModelConfig` discriminated union
- Apply calibration if configured
- Output probabilities that sum to 1

### `trading/edge.py` — Edge computation

- No changes needed if model outputs proper probabilities
- Remove the `normalize_probabilities` hack from backtest configs

### `trading/kelly.py` — Multi-outcome Kelly

- Currently clamps `prob_none = max(1 - sum(probs), 0.01)`. With properly
  calibrated probs, `prob_none ≈ 0` and Kelly optimization becomes cleaner.
- May want to add a small epsilon floor instead of 0.01.

### Feature selection (`evaluate_cv.py`, `build_model.py`)

- Reuse existing forward/backward feature selection
- The selection operates on CV metrics (Brier, accuracy) which automatically
  reflect calibration quality if the model outputs calibrated probabilities

---

## Dependencies

- `statsmodels` — for `ConditionalLogit` with `fit_regularized` (elastic net)
- No other new dependencies expected

## Resolved Questions

1. **Regularization for conditional logit:** `statsmodels.ConditionalLogit` has
   `fit_regularized(method='elastic_net', alpha=..., L1_wt=...)` — no custom MLE
   needed. `alpha` controls overall penalty, `L1_wt` controls L1/L2 mix (0=L2, 1=L1).
   Can pass per-coefficient alpha as array_like for differential regularization.

2. **Top-K selection for softmax GBT:** Use $K = \min(\text{nominees across years})$
   so every year naturally fills all slots. No padding needed.

3. **`prepare_model_data()` interface:** No change needed. Ceremony is already in
   metadata. Add `groups: np.ndarray | None = None` to `PredictionModel.fit()` and
   `predict_proba()` — backward compatible. Extract groups from metadata at call sites.

4. **Feature interactions:** Defer to ablation study after conditional logit is working.

5. **Ensemble approach:** Skipped. Focus on individual approaches first.

6. **Priority ordering:** Phase 1 (diagnostics) → Phase 2 (calibrated softmax) →
   Phase 3 (conditional logit) → Phase 4 (softmax GBT). Conditional logit is the
   primary modeling improvement; calibrated softmax is a quick baseline.

## Open Questions

1. **Feature scaling for conditional logit:** `StandardScaler` needed for regularization
   (penalty magnitude is scale-sensitive). Should we share LR's scaler code or have
   ConditionalLogitModel manage its own scaler?

2. **`BaggedClassifierModel` + ConditionalLogit:** Bootstrap sampling must preserve
   group structure (can't mix nominees across ceremonies). Need to bootstrap over
   *years* not rows. Is bagged conditional logit worth implementing, or is
   regularization sufficient for variance reduction?

3. **ConditionalLogit test-time usage:** At test time, we typically predict one
   ceremony at a time (single group). The softmax is straightforward. But during
   training, all training ceremonies are passed as groups. Confirm this is the
   expected usage.
