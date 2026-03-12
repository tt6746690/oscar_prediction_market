"""Bootstrap aggregation (bagging) meta-model.

- BaggedClassifierModel — wraps any base model with bootstrap aggregation
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from oscar_prediction_market.modeling.models.base import PredictionModel
from oscar_prediction_market.modeling.models.configs import (
    BaggedClassifierConfig,
)
from oscar_prediction_market.modeling.models.types import ModelType

logger = logging.getLogger(__name__)


# ============================================================================
# Bagged Classifier Model
# ============================================================================


class BaggedClassifierModel(PredictionModel):
    """Bootstrap-aggregated (bagged) classifier.

    Trains n_bags models on bootstrap samples of the training data.
    Averages predicted probabilities across all bags for final output.

    Also stores per-bag probabilities for distribution analysis:
    - predict_proba_distribution() returns (n_samples, n_bags) array
    - Useful for assessing prediction uncertainty and confidence intervals
    """

    model_type = ModelType.BAGGED_CLASSIFIER

    def __init__(self, config: BaggedClassifierConfig):
        self.config = config
        self.models: list[PredictionModel] = []
        self._is_fitted = False
        self._rng = np.random.RandomState(config.random_state)

    def _create_base_model(self) -> PredictionModel:
        """Create a single base model instance from config.

        Delegates to create_model() factory — no need to update when adding new model types.
        """
        # Late import to avoid circular dependency: registry imports this module
        from oscar_prediction_market.modeling.models.registry import (
            create_model,
        )

        return create_model(self.config.base_model_config)

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None
    ) -> BaggedClassifierModel:
        """Fit n_bags models on bootstrap samples of the training data.

        Each bootstrap sample has the same size as the original training set,
        drawn with replacement. Some samples will be duplicated, others omitted
        (out-of-bag) — this is the source of diversity between bags.

        Note: groups is accepted for interface compatibility but ignored.
        Bagging does not support group-aware models (ConditionalLogit, SoftmaxGBT).
        """
        n_samples = len(X)
        self.models = []

        for _ in range(self.config.n_bags):
            # Bootstrap sample: same size, with replacement
            indices = self._rng.randint(0, n_samples, size=n_samples)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            model = self._create_base_model()
            model.fit(X_boot, y_boot)
            self.models.append(model)

        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
        """Predict mean probability across all bags."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        all_probs = self.predict_proba_distribution(X)
        return np.mean(all_probs, axis=1)

    def predict_proba_distribution(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities from each bag individually.

        Returns:
            Array of shape (n_samples, n_bags) with per-bag probabilities.
            Useful for uncertainty analysis: std across axis=1 shows
            prediction variance, min/max shows range.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        all_probs = np.column_stack([model.predict_proba(X) for model in self.models])
        return all_probs

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get mean feature importance across all bags.

        All base models now provide a unified 'importance' column,
        so no column-name branching is needed.
        Returns both mean and std to show stability.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        all_importances = []
        for model in self.models:
            fi = model.get_feature_importance(feature_names)
            all_importances.append(fi.set_index("feature")["importance"])

        importance_df = pd.DataFrame(all_importances)
        mean_importance = importance_df.mean()
        std_importance = importance_df.std()

        result = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": [mean_importance[f] for f in feature_names],
                "importance_std": [std_importance[f] for f in feature_names],
            }
        ).sort_values("importance", ascending=False)

        return result
