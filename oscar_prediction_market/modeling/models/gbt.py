"""Tree-based model implementations (GBT feature family).

- GradientBoostingModel — sklearn gradient boosting
- XGBoostModel — xgboost binary classifier
- SoftmaxGBTModel — multi-class XGBoost with fixed K
- CalibratedSoftmaxGBTModel — binary GBT + temperature-scaled softmax
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from oscar_prediction_market.modeling.calibration import (
    probs_to_logodds,
    softmax_per_group,
)
from oscar_prediction_market.modeling.models.base import PredictionModel
from oscar_prediction_market.modeling.models.configs import (
    CalibratedSoftmaxGBTConfig,
    GradientBoostingConfig,
    SoftmaxGBTConfig,
    XGBoostConfig,
)
from oscar_prediction_market.modeling.models.types import ModelType

logger = logging.getLogger(__name__)


# ============================================================================
# Gradient Boosting Model
# ============================================================================


class GradientBoostingModel(PredictionModel):
    """Gradient Boosting model (no scaling needed)."""

    model_type = ModelType.GRADIENT_BOOSTING

    def __init__(self, config: GradientBoostingConfig):
        self.config = config
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )
        self._is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None
    ) -> GradientBoostingModel:
        """Fit the model."""
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
        """Predict probability of winning (class 1)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        # Return probability of class 1 (winner)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importances from the model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        importances = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)


# ============================================================================
# XGBoost Model
# ============================================================================


class XGBoostModel(PredictionModel):
    """XGBoost model (no scaling needed, tree-based)."""

    model_type = ModelType.XGBOOST

    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> XGBoostModel:
        """Fit the model."""
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
        """Predict probability of winning (class 1)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importances from the model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        importances = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)


# ============================================================================
# Softmax GBT Model (Multi-class XGBoost with fixed K)
# ============================================================================


class SoftmaxGBTModel(PredictionModel):
    """Multi-class GBT with softmax objective for constrained probabilities.

    Two-stage approach:
    1. Sort nominees by precursor signal within each ceremony (strongest first)
    2. Take top-K per ceremony → each ceremony is one row with K×F features
    3. XGBoost multi:softprob with K classes → probabilities sum to 1

    The "class" is the slot position (0 = strongest nominee wins,
    1 = second-strongest wins, etc.). Features for each slot contain
    the nominee's original features.

    At prediction time, probabilities are mapped back to the original
    nominee ordering.
    """

    model_type = ModelType.SOFTMAX_GBT

    def __init__(self, config: SoftmaxGBTConfig):
        self.config = config
        self.model: XGBClassifier | None = None
        self.feature_names_: list[str] | None = None
        self._is_fitted = False

    def _reshape_to_multiclass(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
        """Reshape per-nominee rows into per-ceremony rows with K×F features.

        Nominees within each ceremony are sorted by a signal strength proxy
        (sum of binary precursor winner/nominee features, falling back to
        predicted probability or arbitrary order).

        Returns:
            X_multi: (n_ceremonies, K*F) array
            y_multi: (n_ceremonies,) array — slot index of winner
            group_to_original_indices: maps group → list of original row indices
                in sorted order (for mapping back at prediction time)
        """
        K = self.config.top_k
        F = X.shape[1]

        unique_groups = np.unique(groups)
        X_multi_rows = []
        y_multi_rows = []
        group_to_original: dict[int, list[int]] = {}

        for g in unique_groups:
            mask = groups == g
            indices = np.where(mask)[0]
            X_group = X.iloc[indices]
            y_group = y.iloc[indices]

            # Sort by signal strength: sum of all features as proxy
            # (precursor winners/nominees are binary, so sum ≈ signal count)
            signal_strength = np.asarray(X_group.sum(axis=1).values)
            sorted_order = np.argsort(-signal_strength)  # descending

            # Take top-K
            if len(sorted_order) < K:
                # Pad with zeros if fewer than K nominees
                logger.debug(
                    f"Ceremony {g}: {len(sorted_order)} nominees < top_k={K}, padding with zeros"
                )
                top_indices = sorted_order
            else:
                top_indices = sorted_order[:K]

            original_indices = indices[top_indices].tolist()
            group_to_original[int(g)] = original_indices

            # Build K×F feature row
            row = np.zeros(K * F)
            for slot_idx, orig_idx in enumerate(top_indices):
                row[slot_idx * F : (slot_idx + 1) * F] = np.asarray(X.iloc[orig_idx].values)

            X_multi_rows.append(row)

            # Find winner slot
            winner_mask = np.asarray(y_group.iloc[top_indices].values) == 1
            if winner_mask.any():
                winner_slot = int(np.argmax(winner_mask))
            else:
                # Winner not in top-K (rare) — assign to slot 0
                winner_slot = 0
            y_multi_rows.append(winner_slot)

        return (
            np.array(X_multi_rows),
            np.array(y_multi_rows),
            group_to_original,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> SoftmaxGBTModel:
        """Fit multi-class XGBoost with softmax objective.

        Args:
            X: Per-nominee feature matrix.
            y: Binary target (0/1).
            groups: Required. Ceremony numbers.
        """
        if groups is None:
            raise ValueError("SoftmaxGBTModel requires groups")

        self.feature_names_ = list(X.columns)
        X_multi, y_multi, _ = self._reshape_to_multiclass(X, y, groups)

        K = self.config.top_k

        # XGBoost multi:softprob requires contiguous class labels 0..K-1.
        # In some CV folds, not all winner slots appear in training data.
        # Add zero-weight dummy samples for missing classes.
        present_classes = set(y_multi.tolist())
        missing_classes = [c for c in range(K) if c not in present_classes]

        sample_weight: np.ndarray | None = None
        if missing_classes:
            n_features = X_multi.shape[1]
            dummy_X = np.zeros((len(missing_classes), n_features))
            dummy_y = np.array(missing_classes)
            X_multi = np.vstack([X_multi, dummy_X])
            y_multi = np.concatenate([y_multi, dummy_y])
            # Zero weight for dummy samples, unit weight for real samples
            sample_weight = np.ones(len(y_multi))
            sample_weight[-len(missing_classes) :] = 0.0

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=K,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.model.fit(X_multi, y_multi, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict probabilities, mapped back to original nominee ordering.

        The model outputs K probabilities per ceremony (one per slot).
        These are mapped back to the original per-nominee rows.
        Nominees not in top-K get probability 0.

        Returns:
            1-D array of P(win) for each row. Sums to 1 within each group
            (among the top-K nominees; non-top-K get 0).
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        if groups is None:
            raise ValueError("SoftmaxGBTModel requires groups for prediction")

        K = self.config.top_k
        F = len(self.feature_names_) if self.feature_names_ else X.shape[1]

        probs = np.zeros(len(X), dtype=float)

        for g in np.unique(groups):
            mask = groups == g
            indices = np.where(mask)[0]
            X_group = X.iloc[indices]

            # Sort by signal strength (same as training)
            signal_strength = np.asarray(X_group.sum(axis=1).values)
            sorted_order = np.argsort(-signal_strength)

            top_indices = sorted_order[:K] if len(sorted_order) >= K else sorted_order

            # Build K×F feature row
            row = np.zeros((1, K * F))
            for slot_idx, orig_idx_in_group in enumerate(top_indices):
                row[0, slot_idx * F : (slot_idx + 1) * F] = np.asarray(
                    X.iloc[indices[orig_idx_in_group]].values
                )

            # Predict K-class probabilities
            slot_probs = self.model.predict_proba(row)[0]

            # Map back to original indices
            for slot_idx, orig_idx_in_group in enumerate(top_indices):
                if slot_idx < len(slot_probs):
                    probs[indices[orig_idx_in_group]] = slot_probs[slot_idx]

        return probs

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importances averaged across slots.

        Since the model has K×F features (F features per slot), we average
        importances across the K slots for each original feature.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted first")

        K = self.config.top_k
        F = len(feature_names)
        raw_importances = self.model.feature_importances_

        # Average importance across slots for each base feature
        avg_importances = np.zeros(F)
        for slot in range(K):
            start = slot * F
            end = (slot + 1) * F
            if end <= len(raw_importances):
                avg_importances += raw_importances[start:end]
        avg_importances /= K

        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": avg_importances,
            }
        ).sort_values("importance", ascending=False)


# ============================================================================
# Calibrated Softmax GBT Model
# ============================================================================


class CalibratedSoftmaxGBTModel(PredictionModel):
    """Binary GBT with temperature-scaled softmax normalization.

    Combines the parameter efficiency of binary GBT (F features, ~250 training
    rows) with the probabilistic coherence of multinomial models (probabilities
    sum to 1 per group).

    Training: fits a standard binary GBT on stacked nominee-level data.
    Prediction: converts binary probabilities to log-odds, applies per-group
    softmax with temperature T.

    The temperature parameter controls sharpness:
      - T < 1: sharper predictions (frontrunner gets more mass)
      - T = 1: standard logit-space softmax
      - T > 1: smoother predictions (mass spreads more evenly)

    This is different from naive normalization (p_i / sum p_j), which operates
    on probability space. Temperature-scaled softmax operates on log-odds space,
    preserving the score ratios the binary model learned.
    """

    model_type = ModelType.CALIBRATED_SOFTMAX_GBT

    def __init__(self, config: CalibratedSoftmaxGBTConfig):
        self.config = config
        # Build inner binary GBT config
        self._inner_config = GradientBoostingConfig(
            model_type="gradient_boosting",
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            subsample=config.subsample,
            random_state=config.random_state,
        )
        self._inner_model = GradientBoostingModel(self._inner_config)
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> CalibratedSoftmaxGBTModel:
        """Fit the inner binary GBT on stacked nominee-level data.

        Args:
            X: Feature matrix (per-nominee features).
            y: Binary target (0/1).
            groups: Optional ceremony numbers (not used for training, but
                accepted for interface consistency).
        """
        self._inner_model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict probabilities via temperature-scaled softmax on log-odds.

        For each group (ceremony), converts binary probabilities to log-odds,
        divides by temperature T, then applies softmax to produce a proper
        probability distribution summing to 1.

        Args:
            X: Feature matrix.
            groups: Required. Ceremony numbers for per-group normalization.

        Returns:
            1-D array of P(win) for each row, summing to 1 within each group.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if groups is None:
            raise ValueError("CalibratedSoftmaxGBTModel requires groups for prediction")

        # Get binary probabilities and convert to calibrated multinomial
        binary_probs = self._inner_model.predict_proba(X)
        logodds = probs_to_logodds(binary_probs)
        return softmax_per_group(logodds, groups, self.config.temperature)

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Delegate to inner binary GBT's feature importance."""
        return self._inner_model.get_feature_importance(feature_names)
