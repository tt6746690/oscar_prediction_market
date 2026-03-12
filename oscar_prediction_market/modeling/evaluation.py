"""Evaluation metrics for Oscar Best Picture prediction.

Provides:
- Discrimination metrics: Accuracy, Top-K accuracy, MRR, AUC-ROC
- Calibration metrics: Brier score, Log-loss
- Aggregation across multiple years (micro and macro averaging)
"""

from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from oscar_prediction_market.modeling.utils import ceremony_to_year


class YearPrediction(BaseModel):
    """Prediction results for a single ceremony year.

    All fields required - no defaults.
    """

    ceremony: int = Field(..., description="Oscar ceremony number", ge=1)
    film_ids: list[str] = Field(..., description="IMDb IDs of nominees")
    titles: list[str] = Field(..., description="Film titles")
    probabilities: list[float] = Field(..., description="Predicted P(win) for each nominee")
    actual_winner_idx: int = Field(..., description="Index of actual winner", ge=0)
    y_true: list[int] = Field(..., description="Binary labels (1 for winner, 0 for others)")

    model_config = {"extra": "forbid"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def year(self) -> int:
        """Ceremony year (e.g., 2026 for ceremony 98)."""
        return ceremony_to_year(self.ceremony)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def predicted_ranks(self) -> list[int]:
        """Rank of each nominee by predicted probability (1 = highest prob)."""
        probs = np.array(self.probabilities)
        order = np.argsort(-probs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks.tolist()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def winner_predicted_rank(self) -> int:
        """Rank of the actual winner by predicted probability."""
        return self.predicted_ranks[self.actual_winner_idx]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def top_predicted_idx(self) -> int:
        """Index of the nominee with highest predicted probability."""
        return int(np.argmax(self.probabilities))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_correct(self) -> bool:
        """Did the model's top prediction win?"""
        return self.top_predicted_idx == self.actual_winner_idx

    @computed_field  # type: ignore[prop-decorator]
    @property
    def winner_probability(self) -> float:
        """Predicted probability for the actual winner."""
        return self.probabilities[self.actual_winner_idx]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def winner_title(self) -> str:
        """Title of the actual winner."""
        return self.titles[self.actual_winner_idx]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def top_predicted_title(self) -> str:
        """Title of the model's top prediction."""
        return self.titles[self.top_predicted_idx]


# Type alias for validated prediction list (min 1 element)
PredictionList = Annotated[list[YearPrediction], Field(min_length=1)]


# ============================================================================
# Discrimination Metrics
# ============================================================================


def accuracy(predictions: list[YearPrediction]) -> float:
    """Fraction of years where top prediction was the winner."""
    correct = sum(1 for p in predictions if p.is_correct)
    return correct / len(predictions)


def top_k_accuracy(predictions: list[YearPrediction], k: int) -> float:
    """Fraction of years where winner was in top-k predictions."""
    in_top_k = sum(1 for p in predictions if p.winner_predicted_rank <= k)
    return in_top_k / len(predictions)


def mean_reciprocal_rank(predictions: list[YearPrediction]) -> float:
    """Mean of 1/rank for the actual winner across all years."""
    reciprocal_ranks = [1.0 / p.winner_predicted_rank for p in predictions]
    return float(np.mean(reciprocal_ranks))


def auc_roc_pooled(predictions: list[YearPrediction]) -> float:
    """
    Pool all predictions and compute AUC-ROC (micro-average).

    Note: This pools across years, which may not be ideal since
    each year has different base rates. Use with caution.
    """
    y_true_all = np.concatenate([p.y_true for p in predictions])
    y_prob_all = np.concatenate([p.probabilities for p in predictions])

    # Handle edge case where all labels are same class
    if len(np.unique(y_true_all)) < 2:
        return 0.5

    return float(roc_auc_score(y_true_all, y_prob_all))


def auc_roc_per_year(predictions: list[YearPrediction]) -> list[float]:
    """Compute AUC-ROC for each year (returns list of AUCs)."""
    aucs = []
    for p in predictions:
        # Can't compute AUC with only one class
        if len(np.unique(p.y_true)) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(float(roc_auc_score(p.y_true, p.probabilities)))
    return aucs


# ============================================================================
# Calibration Metrics
# ============================================================================


def brier_score_pooled(predictions: list[YearPrediction]) -> float:
    """
    Pool all predictions and compute Brier score (micro-average).

    Brier score = mean squared error of probability estimates.
    Lower is better, 0 = perfect, 0.25 = random for balanced classes.
    """
    y_true_all = np.concatenate([p.y_true for p in predictions])
    y_prob_all = np.concatenate([p.probabilities for p in predictions])

    return float(brier_score_loss(y_true_all, y_prob_all))


def brier_score_per_year(predictions: list[YearPrediction]) -> list[float]:
    """Compute Brier score for each year."""
    return [float(brier_score_loss(p.y_true, p.probabilities)) for p in predictions]


def log_loss_pooled(predictions: list[YearPrediction]) -> float:
    """
    Pool all predictions and compute log loss (micro-average).

    Lower is better. Perfect = 0, random = log(2) ≈ 0.693 for balanced.
    """
    y_true_all = np.concatenate([p.y_true for p in predictions])
    y_prob_all = np.concatenate([p.probabilities for p in predictions])

    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob_all, 1e-15, 1 - 1e-15)

    return float(log_loss(y_true_all, y_prob_clipped))


def log_loss_per_year(predictions: list[YearPrediction]) -> list[float]:
    """Compute log loss for each year."""
    losses = []
    for p in predictions:
        y_prob_clipped = np.clip(p.probabilities, 1e-15, 1 - 1e-15)
        # Handle edge case where all labels are same class
        if len(np.unique(p.y_true)) < 2:
            # Compute manual log loss for binary case
            y_true_arr = np.array(p.y_true)
            loss = -np.mean(
                y_true_arr * np.log(y_prob_clipped)
                + (1 - y_true_arr) * np.log(1 - np.array(y_prob_clipped))
            )
            losses.append(float(loss))
        else:
            losses.append(float(log_loss(p.y_true, y_prob_clipped)))
    return losses


def mean_winner_probability(predictions: list[YearPrediction]) -> float:
    """Average predicted probability assigned to the actual winner."""
    return float(np.mean([p.winner_probability for p in predictions]))


def prob_sum_per_year(predictions: list[YearPrediction]) -> list[float]:
    """Sum of predicted probabilities per ceremony year.

    For a properly constrained model (e.g. conditional logit), this should be
    exactly 1.0. For independent binary models, the sum can be far from 1.0:
    - LR typically under-distributes (sum < 1)
    - GBT typically over-distributes (sum > 1)
    """
    return [float(np.sum(p.probabilities)) for p in predictions]


# ============================================================================
# Aggregate Metrics with Nested Structure
# ============================================================================


class ProbSumStats(BaseModel):
    """Probability sum statistics across ceremony years.

    For properly constrained models (e.g. conditional logit, softmax GBT),
    all values should be 1.0. For independent binary models, deviations
    indicate probability mass is not properly distributed.
    """

    mean: float = Field(..., description="Mean probability sum across years")
    std: float = Field(..., description="Std of probability sums")
    min: float = Field(..., description="Minimum probability sum")
    max: float = Field(..., description="Maximum probability sum")

    model_config = {"extra": "forbid"}


class MicroMetrics(BaseModel):
    """Micro-averaged (pooled) metrics across all samples."""

    accuracy: float = Field(..., description="Accuracy pooled across all years", ge=0, le=1)
    top_3_accuracy: float = Field(..., description="Top-3 accuracy pooled", ge=0, le=1)
    top_5_accuracy: float = Field(..., description="Top-5 accuracy pooled", ge=0, le=1)
    mrr: float = Field(..., description="Mean Reciprocal Rank pooled", ge=0, le=1)
    auc_roc: float = Field(..., description="AUC-ROC pooled across samples", ge=0, le=1)
    brier_score: float = Field(..., description="Brier score pooled", ge=0)
    log_loss: float = Field(..., description="Log loss pooled", ge=0)
    mean_winner_prob: float = Field(..., description="Mean winner probability pooled", ge=0, le=1)

    model_config = {"extra": "forbid"}


class MacroMetrics(BaseModel):
    """Macro-averaged (per-year then averaged) metrics."""

    accuracy: float = Field(..., description="Accuracy averaged across years", ge=0, le=1)
    top_3_accuracy: float = Field(
        ..., description="Top-3 accuracy averaged across years", ge=0, le=1
    )
    top_5_accuracy: float = Field(
        ..., description="Top-5 accuracy averaged across years", ge=0, le=1
    )
    mrr: float = Field(..., description="MRR averaged across years", ge=0, le=1)
    auc_roc: float = Field(
        ..., description="AUC-ROC averaged across years (excluding NaN)", ge=0, le=1
    )
    brier_score: float = Field(..., description="Brier score averaged across years", ge=0)
    log_loss: float = Field(..., description="Log loss averaged across years", ge=0)
    mean_winner_prob: float = Field(..., description="Mean winner probability averaged", ge=0, le=1)

    model_config = {"extra": "forbid"}


class EvaluationMetrics(BaseModel):
    """All evaluation metrics for an experiment.

    Nested structure:
    - micro: Pooled across all samples
    - macro: Computed per year, then averaged
    """

    micro: MicroMetrics = Field(..., description="Micro-averaged (pooled) metrics")
    macro: MacroMetrics = Field(..., description="Macro-averaged (per-year) metrics")

    prob_sum: ProbSumStats = Field(
        ..., description="Probability sum statistics (should be ~1.0 for calibrated models)"
    )

    # Details
    num_years: int = Field(..., description="Number of test years", ge=1)
    correct_ceremonies: list[int] = Field(..., description="Ceremonies with correct predictions")
    incorrect_ceremonies: list[int] = Field(
        ..., description="Ceremonies with incorrect predictions"
    )

    model_config = {"extra": "forbid"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def correct_years(self) -> list[int]:
        """Years (not ceremony numbers) with correct predictions."""
        return [ceremony_to_year(c) for c in self.correct_ceremonies]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def incorrect_years(self) -> list[int]:
        """Years (not ceremony numbers) with incorrect predictions."""
        return [ceremony_to_year(c) for c in self.incorrect_ceremonies]


def compute_all_metrics(predictions: list[YearPrediction]) -> EvaluationMetrics:
    """Compute all evaluation metrics from predictions.

    Returns metrics with both micro (pooled) and macro (per-year averaged) variants.

    Raises:
        ValueError: If predictions list is empty.
    """
    if not predictions:
        raise ValueError("No predictions provided. Cannot compute metrics on empty list.")

    correct_ceremonies = [p.ceremony for p in predictions if p.is_correct]
    incorrect_ceremonies = [p.ceremony for p in predictions if not p.is_correct]

    # Compute probability sum stats
    prob_sums = prob_sum_per_year(predictions)
    prob_sum_statistics = ProbSumStats(
        mean=float(np.mean(prob_sums)),
        std=float(np.std(prob_sums)),
        min=float(np.min(prob_sums)),
        max=float(np.max(prob_sums)),
    )

    # Compute per-year metrics for macro averaging
    brier_per_year = brier_score_per_year(predictions)
    log_loss_per_year_vals = log_loss_per_year(predictions)
    accuracy_per_year = [1.0 if p.is_correct else 0.0 for p in predictions]
    top_3_per_year = [1.0 if p.winner_predicted_rank <= 3 else 0.0 for p in predictions]
    top_5_per_year = [1.0 if p.winner_predicted_rank <= 5 else 0.0 for p in predictions]
    mrr_per_year = [1.0 / p.winner_predicted_rank for p in predictions]
    winner_prob_per_year = [p.winner_probability for p in predictions]

    # AUC per year (may have NaN)
    auc_per_year = auc_roc_per_year(predictions)
    auc_valid = [a for a in auc_per_year if not np.isnan(a)]
    auc_macro = float(np.mean(auc_valid)) if auc_valid else 0.5

    micro = MicroMetrics(
        accuracy=accuracy(predictions),
        top_3_accuracy=top_k_accuracy(predictions, k=3),
        top_5_accuracy=top_k_accuracy(predictions, k=5),
        mrr=mean_reciprocal_rank(predictions),
        auc_roc=auc_roc_pooled(predictions),
        brier_score=brier_score_pooled(predictions),
        log_loss=log_loss_pooled(predictions),
        mean_winner_prob=mean_winner_probability(predictions),
    )

    macro = MacroMetrics(
        accuracy=float(np.mean(accuracy_per_year)),
        top_3_accuracy=float(np.mean(top_3_per_year)),
        top_5_accuracy=float(np.mean(top_5_per_year)),
        mrr=float(np.mean(mrr_per_year)),
        auc_roc=auc_macro,
        brier_score=float(np.mean(brier_per_year)),
        log_loss=float(np.mean(log_loss_per_year_vals)),
        mean_winner_prob=float(np.mean(winner_prob_per_year)),
    )

    return EvaluationMetrics(
        micro=micro,
        macro=macro,
        prob_sum=prob_sum_statistics,
        num_years=len(predictions),
        correct_ceremonies=correct_ceremonies,
        incorrect_ceremonies=incorrect_ceremonies,
    )


def format_metrics(metrics: EvaluationMetrics) -> str:
    """Format metrics as a readable string."""
    m = metrics.micro
    mm = metrics.macro
    lines = [
        "=" * 60,
        "Evaluation Metrics",
        "=" * 60,
        "",
        f"{'Metric':<25} {'Micro (Pooled)':<18} {'Macro (Per-Year)':<18}",
        "-" * 60,
        f"{'Accuracy':<25} {m.accuracy:>12.1%}      {mm.accuracy:>12.1%}",
        f"{'Top-3 Accuracy':<25} {m.top_3_accuracy:>12.1%}      {mm.top_3_accuracy:>12.1%}",
        f"{'Top-5 Accuracy':<25} {m.top_5_accuracy:>12.1%}      {mm.top_5_accuracy:>12.1%}",
        f"{'MRR':<25} {m.mrr:>12.3f}      {mm.mrr:>12.3f}",
        f"{'AUC-ROC':<25} {m.auc_roc:>12.3f}      {mm.auc_roc:>12.3f}",
        f"{'Brier Score':<25} {m.brier_score:>12.4f}      {mm.brier_score:>12.4f}",
        f"{'Log Loss':<25} {m.log_loss:>12.4f}      {mm.log_loss:>12.4f}",
        f"{'Mean Winner Prob':<25} {m.mean_winner_prob:>12.1%}      {mm.mean_winner_prob:>12.1%}",
        "",
        "Probability Sum (should be ~1.0 for calibrated models)",
        f"  Mean: {metrics.prob_sum.mean:.3f}  Std: {metrics.prob_sum.std:.3f}"
        f"  Min: {metrics.prob_sum.min:.3f}  Max: {metrics.prob_sum.max:.3f}",
        "-" * 60,
        f"Test Years: {metrics.num_years}",
        f"Correct ({len(metrics.correct_ceremonies)}): {[f'{c} ({ceremony_to_year(c)})' for c in metrics.correct_ceremonies]}",
        f"Incorrect ({len(metrics.incorrect_ceremonies)}): {[f'{c} ({ceremony_to_year(c)})' for c in metrics.incorrect_ceremonies]}",
        "=" * 60,
    ]
    return "\n".join(lines)


def predictions_to_dataframe(predictions: list[YearPrediction]) -> pd.DataFrame:
    """Flatten predictions into a DataFrame with one row per nominee.

    This is the canonical way to convert predictions to tabular format for
    CSV export. Used by evaluate_cv and train_predict.

    Returns:
        DataFrame with columns: ceremony, year, film_id, title, probability,
        rank, is_actual_winner, is_predicted_winner, correct_prediction.
    """
    rows = []
    for pred in predictions:
        for i, (fid, title, prob, rank) in enumerate(
            zip(pred.film_ids, pred.titles, pred.probabilities, pred.predicted_ranks, strict=True)
        ):
            rows.append(
                {
                    "ceremony": pred.ceremony,
                    "year": pred.year,
                    "film_id": fid,
                    "title": title,
                    "probability": float(prob),
                    "rank": int(rank),
                    "is_actual_winner": i == pred.actual_winner_idx,
                    "is_predicted_winner": rank == 1,
                    "correct_prediction": pred.is_correct,
                }
            )
    return pd.DataFrame(rows)
