"""Probability calibration for multinomial prediction.

Post-processing methods to transform independent binary model probabilities
into properly constrained multinomial probabilities (summing to 1 within
each ceremony group).

Methods:
- Temperature-scaled softmax: applies softmax with tunable temperature T
  to log-odds from binary models. T < 1 sharpens, T > 1 flattens.
- Naive normalization: divides by sum (preserves ranking but distorts magnitudes).

The key insight: for Logistic Regression, calibrated softmax on log-odds is
mathematically equivalent to conditional logit with a temperature parameter.
For GBT, we extract log-odds via log(p/(1-p)) and apply the same transform.

Usage:
    from calibration import SoftmaxCalibrator

    calibrator = SoftmaxCalibrator()
    calibrator.fit(train_probs, train_groups, train_y)
    calibrated = calibrator.transform(test_probs, test_groups)
"""

import logging

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SoftmaxCalibratorConfig(BaseModel):
    """Configuration for temperature-scaled softmax calibration.

    Temperature T controls sharpness:
    - T = 1.0: standard softmax
    - T < 1.0: sharper (more confident)
    - T > 1.0: flatter (less confident)

    If T is None, it will be tuned via cross-validation on training data.
    """

    temperature: float | None = Field(
        default=None,
        description="Temperature for softmax. None = auto-tune via grid search.",
    )
    temperature_grid: list[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0],
        description="Grid of T values to search when auto-tuning.",
    )

    model_config = {"extra": "forbid"}


def probs_to_logodds(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert probabilities to log-odds, clipping to avoid infinities."""
    p = np.clip(probs, eps, 1 - eps)
    return np.log(p / (1 - p))


def softmax_per_group(
    logodds: np.ndarray,
    groups: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature-scaled softmax within each group.

    Args:
        logodds: 1-D array of log-odds for each sample.
        groups: 1-D array of group identifiers (e.g. ceremony numbers).
        temperature: Temperature parameter T.

    Returns:
        1-D array of calibrated probabilities summing to 1 within each group.
    """
    result = np.empty_like(logodds, dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        logits = logodds[mask] / temperature
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        result[mask] = exp_logits / exp_logits.sum()
    return result


def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between true labels and predicted probabilities."""
    return float(np.mean((y_true - y_pred) ** 2))


class SoftmaxCalibrator:
    """Temperature-scaled softmax calibrator.

    Converts independent binary probabilities into properly constrained
    multinomial probabilities (summing to 1 per group) via:

        P(i | group) = exp(logodds_i / T) / sum_j exp(logodds_j / T)

    The temperature T is either fixed or tuned on training data using
    leave-one-group-out CV to minimize Brier score.
    """

    def __init__(self, config: SoftmaxCalibratorConfig | None = None):
        self.config = config or SoftmaxCalibratorConfig()
        self.temperature_: float | None = self.config.temperature

    def fit(
        self,
        probs: np.ndarray,
        groups: np.ndarray,
        y: np.ndarray,
    ) -> "SoftmaxCalibrator":
        """Fit the calibrator (tune temperature if not fixed).

        Uses leave-one-group-out CV over all groups in the training data to
        select the temperature that minimizes Brier score.

        Args:
            probs: 1-D array of binary model probabilities.
            groups: 1-D array of group IDs (ceremony numbers).
            y: 1-D array of binary labels (0/1).

        Returns:
            self (fitted calibrator)
        """
        if self.config.temperature is not None:
            self.temperature_ = self.config.temperature
            return self

        logodds = probs_to_logodds(probs)
        unique_groups = np.unique(groups)

        # Leave-one-group-out CV to tune T
        best_T = 1.0
        best_brier = float("inf")

        for T in self.config.temperature_grid:
            fold_briers = []
            for hold_out_group in unique_groups:
                # We only need to evaluate on the held-out group
                test_mask = groups == hold_out_group
                # Apply softmax with this T on the held-out group
                calibrated = softmax_per_group(logodds[test_mask], groups[test_mask], T)
                fold_briers.append(_brier_score(y[test_mask], calibrated))

            mean_brier = float(np.mean(fold_briers))
            if mean_brier < best_brier:
                best_brier = mean_brier
                best_T = T

        self.temperature_ = best_T
        logger.info(f"SoftmaxCalibrator: tuned T={best_T:.2f} (Brier={best_brier:.4f})")
        return self

    def transform(self, probs: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Transform binary probabilities to calibrated multinomial probabilities.

        Args:
            probs: 1-D array of binary model probabilities.
            groups: 1-D array of group IDs (ceremony numbers).

        Returns:
            1-D array of calibrated probabilities (sum to 1 per group).
        """
        if self.temperature_ is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        logodds = probs_to_logodds(probs)
        return softmax_per_group(logodds, groups, self.temperature_)

    def fit_transform(
        self,
        probs: np.ndarray,
        groups: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probs, groups, y)
        return self.transform(probs, groups)


def normalize_per_group(probs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Naive normalization: divide probabilities by their sum within each group.

    This preserves ranking but distorts magnitudes. Included for comparison.
    """
    result = np.empty_like(probs, dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        total = probs[mask].sum()
        if total > 0:
            result[mask] = probs[mask] / total
        else:
            # Uniform if all zero
            result[mask] = 1.0 / mask.sum()
    return result
