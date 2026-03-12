"""Hyperparameter tuning for Oscar Best Picture prediction.

Provides two approaches for hyperparameter selection:

┌─────────────────────┬──────────┬──────────┬────────────┬─────────────────────────────────────────┐
│ Approach            │ Bias     │ Variance │ Complexity │ Best For / Motivation                   │
├─────────────────────┼──────────┼──────────┼────────────┼─────────────────────────────────────────┤
│ Non-Nested CV       │ Optimis- │ Lower    │ Medium     │ DEVELOPMENT & DEBUGGING. Fast iteration.│
│ (Simple)            │ tic      │          │            │ Reports are biased upward since test    │
│                     │          │          │            │ data is used for selection. Use to      │
│                     │          │          │            │ narrow down candidate configs.          │
├─────────────────────┼──────────┼──────────┼────────────┼─────────────────────────────────────────┤
│ Nested CV           │ Unbiased │ Higher   │ High       │ FINAL REPORTING. Honest performance     │
│ (Proper)            │          │          │            │ estimates. Outer loop for evaluation,   │
│                     │          │          │            │ inner loop for selection. Gold standard │
│                     │          │          │            │ for small datasets.                     │
└─────────────────────┴──────────┴──────────┴────────────┴─────────────────────────────────────────┘

The module supports:
1. Grid search over hyperparameter configurations
2. Pluggable CV splitters (Expanding, LOYO, Sliding, Bootstrap)
3. Combined metric optimization (accuracy + calibration)
4. Complexity-aware tie-breaking (prefer simpler models)

Usage:
    from hyperparameter_tuning import (
        run_non_nested_cv,
        run_nested_cv,
    )
    from cv_splitting import ExpandingWindowSplitter, LeaveOneYearOutSplitter

    # Non-nested CV (for development)
    results = run_non_nested_cv(
        df=df,
        feature_set=feature_set,
        param_grid=param_grid,
        splitter=ExpandingWindowSplitter(min_train_years=5),
        feature_family=FeatureFamily.LR,
    )

    # Nested CV (for final reporting)
    results = run_nested_cv(
        df=df,
        feature_set=feature_set,
        param_grid=param_grid,
        outer_splitter=ExpandingWindowSplitter(min_train_years=5),
        inner_splitter=LeaveOneYearOutSplitter(),
        feature_family=FeatureFamily.LR,
    )
"""

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from oscar_prediction_market.modeling.cv_splitting import (
    CVFold,
    CVSplitter,
)
from oscar_prediction_market.modeling.data_loader import (
    get_ceremony_years,
    prepare_model_data,
)
from oscar_prediction_market.modeling.evaluation import (
    EvaluationMetrics,
    YearPrediction,
    compute_all_metrics,
)
from oscar_prediction_market.modeling.feature_engineering import FeatureSet
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)
from oscar_prediction_market.modeling.models import (
    BaggedClassifierConfig,
    CalibratedSoftmaxGBTConfig,
    ConditionalLogitConfig,
    GradientBoostingConfig,
    LogisticRegressionConfig,
    ModelConfig,
    SoftmaxGBTConfig,
    XGBoostConfig,
    create_model,
)
from oscar_prediction_market.modeling.utils import ceremony_to_year

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _deduplicate_predictions(
    predictions: list[YearPrediction],
) -> list[YearPrediction]:
    """Keep the first prediction per ceremony, dropping duplicates.

    Some CV splitters (e.g., Bootstrap) may produce overlapping test sets,
    yielding multiple predictions for the same ceremony. We keep the first
    occurrence (from the earliest fold) and log a warning if duplicates are
    dropped.
    """
    seen: set[int] = set()
    unique: list[YearPrediction] = []
    for pred in predictions:
        if pred.ceremony not in seen:
            seen.add(pred.ceremony)
            unique.append(pred)
    n_dropped = len(predictions) - len(unique)
    if n_dropped > 0:
        logger.debug("Dropped %d duplicate predictions (kept first per ceremony)", n_dropped)
    return unique


# ============================================================================
# Scoring and Selection
# ============================================================================


class CombinedScore(BaseModel):
    """Combined score for model selection.

    Combines top-1 accuracy (discrimination) and log-loss (calibration).
    Lower is better for log-loss, higher is better for accuracy.
    """

    model_config = {"extra": "forbid"}

    accuracy: float = Field(..., ge=0, le=1)
    log_loss: float = Field(..., ge=0)
    combined: float = Field(..., ge=0, le=1)

    @staticmethod
    def compute(metrics: EvaluationMetrics, accuracy_weight: float = 0.5) -> "CombinedScore":
        """Compute combined score from evaluation metrics.

        Args:
            metrics: Evaluation metrics (uses macro-averaged values)
            accuracy_weight: Weight for accuracy (1 - accuracy_weight for calibration)

        Returns:
            CombinedScore with accuracy, log_loss, and combined score
        """
        acc = metrics.macro.accuracy
        ll = metrics.macro.log_loss

        # Normalize log-loss to [0, 1] range (invert so higher is better)
        # Typical log-loss range is 0.3-1.5 for this problem
        # Use sigmoid-like transform: 1 / (1 + log_loss)
        ll_score = 1.0 / (1.0 + ll)

        combined = accuracy_weight * acc + (1.0 - accuracy_weight) * ll_score

        return CombinedScore(accuracy=acc, log_loss=ll, combined=combined)


def get_model_complexity(config: ModelConfig) -> float:
    """Get complexity score for a model config (lower = simpler).

    Used for tie-breaking: prefer simpler models when scores are similar.
    """
    if isinstance(config, LogisticRegressionConfig):
        # Higher C = less regularization = more complex
        # L1 is sparser (simpler) than L2
        c_score = np.log10(config.C) + 2  # Normalize to ~[0, 4]
        l1_penalty = 1.0 - config.l1_ratio  # L1=0 penalty, L2=1 penalty
        return c_score + l1_penalty
    elif isinstance(config, GradientBoostingConfig):
        # More trees, deeper trees = more complex
        return (
            config.n_estimators / 50  # Normalize to ~[1, 4]
            + config.max_depth  # [2, 4]
            - config.min_samples_leaf / 5  # Larger leaf = simpler
        )
    elif isinstance(config, XGBoostConfig):
        # Similar to GBT: more trees + deeper + less regularization = more complex
        return (
            config.n_estimators / 50
            + config.max_depth
            - config.reg_lambda / 2  # More L2 reg = simpler
        )
    elif isinstance(config, BaggedClassifierConfig):
        # Complexity of base model + small penalty for number of bags
        return get_model_complexity(config.base_model_config) + config.n_bags / 1000
    elif isinstance(config, ConditionalLogitConfig):
        # Higher alpha = more regularization = simpler
        # L1 is sparser (simpler) than L2
        alpha_score = -np.log10(max(config.alpha, 1e-10))  # Normalize ~[0, 10]
        l1_penalty = 1.0 - config.L1_wt  # L1=0 penalty, L2=1 penalty
        return alpha_score + l1_penalty
    elif isinstance(config, SoftmaxGBTConfig):
        # Similar to GBT/XGBoost
        return (
            config.n_estimators / 50
            + config.max_depth
            - config.reg_lambda / 2
            + config.top_k / 5  # Higher K = more complex
        )
    elif isinstance(config, CalibratedSoftmaxGBTConfig):
        # Same as GBT + small penalty for temperature deviating from 1
        return (
            config.n_estimators / 50
            + config.max_depth
            - config.min_samples_leaf / 5
            + abs(config.temperature - 1.0)  # Prefer T near 1
        )
    else:
        return 0.0


def select_best_config(
    results: list[tuple[ModelConfig, EvaluationMetrics]],
    tolerance: float = 0.01,
) -> tuple[ModelConfig, EvaluationMetrics, CombinedScore]:
    """Select best config with complexity-aware tie-breaking.

    Args:
        results: List of (config, metrics) tuples
        tolerance: Score difference within which to prefer simpler model

    Returns:
        Best (config, metrics, score) tuple
    """
    scored = [(config, metrics, CombinedScore.compute(metrics)) for config, metrics in results]

    # Sort by combined score (descending), then complexity (ascending)
    scored.sort(key=lambda x: (-x[2].combined, get_model_complexity(x[0])))

    best = scored[0]

    # Check if simpler model is within tolerance
    for config, metrics, score in scored[1:]:
        if best[2].combined - score.combined <= tolerance:
            if get_model_complexity(config) < get_model_complexity(best[0]):
                logger.info(
                    f"Preferring simpler model (score diff: {best[2].combined - score.combined:.4f})"
                )
                best = (config, metrics, score)
                break

    return best


# ============================================================================
# Core CV Functions
# ============================================================================


def run_single_fold(
    df: pd.DataFrame,
    fold: CVFold,
    feature_set: FeatureSet,
    model_config: ModelConfig,
) -> list[YearPrediction]:
    """Run a single CV fold and return predictions.

    Args:
        df: Full dataset
        fold: CV fold with train/test ceremonies
        feature_set: Features to use
        model_config: Model configuration

    Returns:
        List of YearPrediction for each test year in the fold
    """
    # Prepare training data
    train_df = df[df["ceremony"].isin(fold.train_ceremonies)].copy()
    X_train, y_train, meta_train = prepare_model_data(train_df, feature_set)

    # Train model
    model = create_model(model_config)
    model.fit(X_train, y_train, groups=np.asarray(meta_train["ceremony"].values))

    # Predict for each test year
    predictions = []
    for test_ceremony in fold.test_ceremonies:
        test_df = df[df["ceremony"] == test_ceremony].copy()
        if len(test_df) == 0:
            continue

        X_test, y_test, metadata_test = prepare_model_data(test_df, feature_set)
        probabilities = model.predict_proba(
            X_test, groups=np.asarray(metadata_test["ceremony"].values)
        )
        y_test_array = np.asarray(y_test.values)
        actual_winner_idx = int(np.argmax(y_test_array))

        predictions.append(
            YearPrediction(
                ceremony=test_ceremony,
                film_ids=metadata_test["film_id"].tolist(),
                titles=metadata_test["title"].tolist(),
                probabilities=probabilities.tolist(),
                actual_winner_idx=actual_winner_idx,
                y_true=y_test_array.tolist(),
            )
        )

    return predictions


def run_cv_for_config(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_config: ModelConfig,
    splitter: CVSplitter,
) -> tuple[EvaluationMetrics, list[YearPrediction]]:
    """Run cross-validation for a single model configuration.

    Args:
        df: Full dataset
        feature_set: Features to use
        model_config: Model configuration
        splitter: CV splitter defining train/test splits

    Returns:
        (metrics, predictions) tuple
    """
    ceremony_years = get_ceremony_years(df)
    folds = splitter.generate_folds(ceremony_years)

    all_predictions = []
    for fold in folds:
        fold_predictions = run_single_fold(df, fold, feature_set, model_config)
        all_predictions.extend(fold_predictions)

    # Deduplicate by ceremony (some splitters may have overlapping test sets)
    unique_predictions = _deduplicate_predictions(all_predictions)

    if len(unique_predictions) == 0:
        raise ValueError("No predictions generated - check splitter configuration")

    metrics = compute_all_metrics(unique_predictions)
    return metrics, unique_predictions


# ============================================================================
# Non-Nested CV
# ============================================================================


class NonNestedCVResult(BaseModel):
    """Results from non-nested cross-validation.

    Note: Performance estimates are OPTIMISTIC because test data
    was used for model selection.
    """

    splitter_name: str = Field(..., description="CV splitter used")
    splitter_config: dict = Field(..., description="Splitter configuration")
    feature_family: FeatureFamily = Field(..., description="Feature engineering family")
    num_configs_evaluated: int = Field(..., description="Number of configs in grid")
    best_config: dict = Field(..., description="Best model configuration")
    best_metrics: EvaluationMetrics = Field(..., description="Metrics for best config")
    best_score: float = Field(..., description="Combined score for best config")
    all_results: list[dict] = Field(..., description="Results for all configs")
    note: str = Field(
        default="Performance estimates are OPTIMISTIC (test data used for selection)",
        description="Warning about bias",
    )

    model_config = {"extra": "forbid"}


def run_non_nested_cv(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    param_grid: list[ModelConfig],
    splitter: CVSplitter,
    feature_family: FeatureFamily,
    verbose: bool = True,
) -> NonNestedCVResult:
    """Run non-nested CV for hyperparameter selection.

    For each config in param_grid:
    1. Run full CV using the given splitter
    2. Compute combined score (accuracy + calibration)
    3. Select best config (with complexity tie-breaking)

    Note: This produces OPTIMISTIC estimates because the same data
    is used for both selection and evaluation.

    Args:
        df: Full dataset
        feature_set: Features to use
        param_grid: List of model configurations to try
        splitter: CV splitter
        feature_family: Feature engineering family (FeatureFamily.LR or FeatureFamily.GBT)
        verbose: Print progress

    Returns:
        NonNestedCVResult with best config and all results
    """
    if verbose:
        logger.info(f"Running non-nested CV with {splitter.name}")
        logger.info(f"  Evaluating {len(param_grid)} configurations")

    all_results = []
    config_metrics = []

    for i, config in enumerate(param_grid):
        try:
            metrics, _ = run_cv_for_config(df, feature_set, config, splitter)
            score = CombinedScore.compute(metrics)

            result = {
                "config": config.model_dump(),
                "accuracy": metrics.macro.accuracy,
                "log_loss": metrics.macro.log_loss,
                "combined_score": score.combined,
                "complexity": get_model_complexity(config),
            }
            all_results.append(result)
            config_metrics.append((config, metrics))

            if verbose:
                logger.info(
                    f"  [{i + 1}/{len(param_grid)}] "
                    f"Acc={metrics.macro.accuracy:.3f}, "
                    f"LogLoss={metrics.macro.log_loss:.3f}, "
                    f"Score={score.combined:.4f}"
                )

        except Exception as e:
            logger.warning(f"  Config {i + 1} failed: {e}")
            continue

    if len(config_metrics) == 0:
        raise ValueError("All configurations failed")

    # Select best
    best_config, best_metrics, best_score = select_best_config(config_metrics)

    if verbose:
        logger.info(f"\nBest config (score={best_score.combined:.4f}):")
        logger.info(f"  Accuracy: {best_metrics.macro.accuracy:.3f}")
        logger.info(f"  Log-loss: {best_metrics.macro.log_loss:.3f}")

    return NonNestedCVResult(
        splitter_name=splitter.name,
        splitter_config=splitter.get_config().model_dump(),
        feature_family=feature_family,
        num_configs_evaluated=len(param_grid),
        best_config=best_config.model_dump(),
        best_metrics=best_metrics,
        best_score=best_score.combined,
        all_results=all_results,
    )


# ============================================================================
# Nested CV
# ============================================================================


class NestedCVFoldResult(BaseModel):
    """Result for a single outer fold of nested CV."""

    outer_fold_idx: int = Field(..., description="Outer fold index")
    test_ceremonies: list[int] = Field(..., description="Test ceremony years")
    selected_config: dict = Field(..., description="Config selected by inner CV")
    inner_cv_score: float = Field(..., description="Inner CV score for selected config")
    predictions: list[dict] = Field(..., description="Predictions for test years")

    model_config = {"extra": "forbid"}


class NestedCVResult(BaseModel):
    """Results from nested cross-validation.

    Performance estimates are UNBIASED because model selection
    happens in the inner loop, separate from evaluation.
    """

    outer_splitter_name: str = Field(..., description="Outer CV splitter")
    inner_splitter_name: str = Field(..., description="Inner CV splitter")
    outer_splitter_config: dict = Field(..., description="Outer splitter configuration")
    inner_splitter_config: dict = Field(..., description="Inner splitter configuration")
    feature_family: FeatureFamily = Field(..., description="Feature engineering family")
    num_configs_in_grid: int = Field(..., description="Number of configs in grid")
    num_outer_folds: int = Field(..., description="Number of outer folds")
    fold_results: list[NestedCVFoldResult] = Field(..., description="Results per outer fold")
    aggregated_metrics: EvaluationMetrics = Field(..., description="Aggregated metrics")
    config_selection_frequency: dict = Field(..., description="How often each config was selected")
    note: str = Field(
        default="Performance estimates are UNBIASED (proper train/validation/test separation)",
        description="Note about validity",
    )

    model_config = {"extra": "forbid"}


def run_nested_cv(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    param_grid: list[ModelConfig],
    outer_splitter: CVSplitter,
    inner_splitter: CVSplitter,
    feature_family: FeatureFamily,
    verbose: bool = True,
) -> NestedCVResult:
    """Run nested CV for unbiased hyperparameter evaluation.

    Outer loop (evaluation):
        For each test fold:
            Inner loop (selection):
                For each config: run CV on train data → select best
            Train with best config on all train data
            Evaluate on test fold

    Args:
        df: Full dataset
        feature_set: Features to use
        param_grid: List of model configurations to try
        outer_splitter: Splitter for outer evaluation loop
        inner_splitter: Splitter for inner selection loop
        feature_family: Feature engineering family (FeatureFamily.LR or FeatureFamily.GBT)
        verbose: Print progress

    Returns:
        NestedCVResult with unbiased performance estimates
    """
    if verbose:
        logger.info("Running nested CV")
        logger.info(f"  Outer: {outer_splitter.name}")
        logger.info(f"  Inner: {inner_splitter.name}")
        logger.info(f"  Grid size: {len(param_grid)} configurations")

    ceremony_years = get_ceremony_years(df)
    outer_folds = outer_splitter.generate_folds(ceremony_years)

    if verbose:
        logger.info(f"  Outer folds: {len(outer_folds)}")

    fold_results = []
    all_predictions = []
    config_selections: dict[str, int] = {}

    for outer_fold in outer_folds:
        if verbose:
            logger.info(
                f"\nOuter fold {outer_fold.fold_idx + 1}/{len(outer_folds)}: {outer_fold.description}"
            )

        # Get train data for this outer fold
        train_df = df[df["ceremony"].isin(outer_fold.train_ceremonies)].copy()
        train_ceremonies = outer_fold.train_ceremonies

        # Inner CV: select best config using train data only
        inner_results = []
        for config in param_grid:
            try:
                # Generate inner folds on train ceremonies only
                inner_folds = inner_splitter.generate_folds(train_ceremonies)

                if len(inner_folds) == 0:
                    continue

                # Run inner CV
                inner_predictions = []
                for inner_fold in inner_folds:
                    inner_train_df = train_df[
                        train_df["ceremony"].isin(inner_fold.train_ceremonies)
                    ].copy()
                    inner_test_df = train_df[
                        train_df["ceremony"].isin(inner_fold.test_ceremonies)
                    ].copy()

                    if len(inner_train_df) == 0 or len(inner_test_df) == 0:
                        continue

                    X_train, y_train, meta_train = prepare_model_data(inner_train_df, feature_set)
                    model = create_model(config)
                    model.fit(
                        X_train,
                        y_train,
                        groups=np.asarray(meta_train["ceremony"].values),
                    )

                    for test_ceremony in inner_fold.test_ceremonies:
                        test_df_single = inner_test_df[
                            inner_test_df["ceremony"] == test_ceremony
                        ].copy()
                        if len(test_df_single) == 0:
                            continue

                        X_test, y_test, metadata = prepare_model_data(test_df_single, feature_set)
                        probs = model.predict_proba(
                            X_test, groups=np.asarray(metadata["ceremony"].values)
                        )
                        y_test_arr = np.asarray(y_test.values)

                        inner_predictions.append(
                            YearPrediction(
                                ceremony=test_ceremony,
                                film_ids=metadata["film_id"].tolist(),
                                titles=metadata["title"].tolist(),
                                probabilities=probs.tolist(),
                                actual_winner_idx=int(np.argmax(y_test_arr)),
                                y_true=y_test_arr.tolist(),
                            )
                        )

                if len(inner_predictions) > 0:
                    # Deduplicate
                    unique = _deduplicate_predictions(inner_predictions)
                    inner_metrics = compute_all_metrics(unique)
                    inner_results.append((config, inner_metrics))

            except Exception as e:
                logger.debug(f"    Inner CV failed for config: {e}")
                continue

        if len(inner_results) == 0:
            logger.warning(f"  No valid configs for outer fold {outer_fold.fold_idx}")
            continue

        # Select best config from inner CV
        best_config, best_inner_metrics, best_score = select_best_config(inner_results)

        # Track config selection frequency
        config_key = str(best_config.model_dump())
        config_selections[config_key] = config_selections.get(config_key, 0) + 1

        if verbose:
            logger.info(f"  Selected config (inner score={best_score.combined:.4f})")

        # Train with best config on ALL train data
        X_train, y_train, meta_train = prepare_model_data(train_df, feature_set)
        final_model = create_model(best_config)
        final_model.fit(X_train, y_train, groups=np.asarray(meta_train["ceremony"].values))

        # Evaluate on outer test fold
        outer_predictions = []
        for test_ceremony in outer_fold.test_ceremonies:
            test_df = df[df["ceremony"] == test_ceremony].copy()
            if len(test_df) == 0:
                continue

            X_test, y_test, metadata = prepare_model_data(test_df, feature_set)
            probs = final_model.predict_proba(
                X_test, groups=np.asarray(metadata["ceremony"].values)
            )
            y_test_arr = np.asarray(y_test.values)

            pred = YearPrediction(
                ceremony=test_ceremony,
                film_ids=metadata["film_id"].tolist(),
                titles=metadata["title"].tolist(),
                probabilities=probs.tolist(),
                actual_winner_idx=int(np.argmax(y_test_arr)),
                y_true=y_test_arr.tolist(),
            )
            outer_predictions.append(pred)
            all_predictions.append(pred)

            status = "✓" if pred.is_correct else "✗"
            if verbose:
                logger.info(
                    f"    {test_ceremony} ({ceremony_to_year(test_ceremony)}): {status} "
                    f"Pred={pred.top_predicted_title}, Actual={pred.winner_title}"
                )

        fold_results.append(
            NestedCVFoldResult(
                outer_fold_idx=outer_fold.fold_idx,
                test_ceremonies=outer_fold.test_ceremonies,
                selected_config=best_config.model_dump(),
                inner_cv_score=best_score.combined,
                # Exclude computed fields to allow reconstruction
                predictions=[
                    p.model_dump(
                        include={
                            "ceremony",
                            "film_ids",
                            "titles",
                            "probabilities",
                            "actual_winner_idx",
                            "y_true",
                        }
                    )
                    for p in outer_predictions
                ],
            )
        )

    # Aggregate metrics across all outer folds
    if len(all_predictions) == 0:
        raise ValueError("No predictions generated - check splitter configuration")

    # Deduplicate
    unique_predictions = _deduplicate_predictions(all_predictions)

    aggregated_metrics = compute_all_metrics(unique_predictions)

    if verbose:
        logger.info(f"\n{'=' * 60}")
        logger.info("Nested CV Results (UNBIASED)")
        logger.info(f"{'=' * 60}")
        logger.info(f"Accuracy: {aggregated_metrics.macro.accuracy:.3f}")
        logger.info(f"Log-loss: {aggregated_metrics.macro.log_loss:.3f}")
        logger.info(f"Test years: {len(unique_predictions)}")

    return NestedCVResult(
        outer_splitter_name=outer_splitter.name,
        inner_splitter_name=inner_splitter.name,
        outer_splitter_config=outer_splitter.get_config().model_dump(),
        inner_splitter_config=inner_splitter.get_config().model_dump(),
        feature_family=feature_family,
        num_configs_in_grid=len(param_grid),
        num_outer_folds=len(outer_folds),
        fold_results=fold_results,
        aggregated_metrics=aggregated_metrics,
        config_selection_frequency=config_selections,
    )
