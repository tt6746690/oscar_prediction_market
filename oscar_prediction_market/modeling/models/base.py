"""Abstract base class for prediction models."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from oscar_prediction_market.modeling.models.types import ModelType


class PredictionModel(ABC):
    """Abstract base class for prediction models.

    The optional ``groups`` parameter supports group-aware models like
    ConditionalLogitModel that need ceremony grouping. Binary models
    (LR, GBT, XGBoost) ignore it.
    """

    model_type: ModelType

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> "PredictionModel":
        """Fit the model on training data.

        Args:
            X: Feature matrix.
            y: Binary target (0/1).
            groups: Optional group IDs (e.g. ceremony numbers). Required for
                group-aware models (ConditionalLogit, SoftmaxGBT).
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict probability of winning for each sample.

        Args:
            X: Feature matrix.
            groups: Optional group IDs. Required for group-aware models.

        Returns:
            1-D array of P(win) for each sample.
        """
        pass

    @abstractmethod
    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importances from the model."""
        pass
