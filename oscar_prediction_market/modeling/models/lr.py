"""Linear model implementations (LR feature family).

- LogisticRegressionModel — sklearn logistic regression with standardization
- ConditionalLogitModel — McFadden's choice model (statsmodels)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.conditional_models import (
    ConditionalLogit as _StatsmodelsConditionalLogit,
)

from oscar_prediction_market.modeling.models.base import PredictionModel
from oscar_prediction_market.modeling.models.configs import (
    ConditionalLogitConfig,
    LogisticRegressionConfig,
)
from oscar_prediction_market.modeling.models.types import ModelType

logger = logging.getLogger(__name__)


# ============================================================================
# Logistic Regression Model
# ============================================================================


class LogisticRegressionModel(PredictionModel):
    """Logistic Regression model with standardization."""

    model_type = ModelType.LOGISTIC_REGRESSION

    def __init__(self, config: LogisticRegressionConfig):
        self.config = config
        self.scaler = StandardScaler()
        # In sklearn 1.8+, l1_ratio alone controls the regularization type:
        # l1_ratio=0 → L2, l1_ratio=1 → L1, 0 < l1_ratio < 1 → elasticnet.
        # The penalty parameter is deprecated (removed in 1.10).
        # saga solver is required for l1_ratio support.
        self.model = LogisticRegression(
            C=self.config.C,
            l1_ratio=self.config.l1_ratio,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )
        self._is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None
    ) -> LogisticRegressionModel:
        """Fit the model with standardized features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
        """Predict probability of winning (class 1)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        # Return probability of class 1 (winner)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature coefficients as importance.

        Returns DataFrame with columns: feature, coefficient, abs_coefficient, importance.
        The 'importance' column is an alias for abs_coefficient, providing a unified
        column name across all model types.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        coefs = self.model.coef_[0]
        abs_coefs = np.abs(coefs)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coefficient": abs_coefs,
                "importance": abs_coefs,
            }
        ).sort_values("importance", ascending=False)


# ============================================================================
# Conditional Logit Model (McFadden's Choice Model)
# ============================================================================


class ConditionalLogitModel(PredictionModel):
    """Conditional logistic regression for discrete choice modeling.

    Models the choice among K nominees within each ceremony year:

        P(win_i | year) = exp(x_i^T β) / Σ_j exp(x_j^T β)

    This is the principled fix for independent binary models: probabilities
    sum to 1 by construction, and the model directly learns competitive
    dynamics (nominee features compete against the field).

    Uses statsmodels ConditionalLogit with elastic net regularization.
    Feature scaling via StandardScaler (required since regularization
    penalty is scale-sensitive).

    The key difference from sklearn multinomial LR:
    - Multinomial LR: learns K weight vectors, needs fixed K classes
    - Conditional logit: learns 1 shared β, flexible choice set sizes
    """

    model_type = ModelType.CONDITIONAL_LOGIT

    def __init__(self, config: ConditionalLogitConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.params_: np.ndarray | None = None
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> ConditionalLogitModel:
        """Fit conditional logit model with elastic net regularization.

        Args:
            X: Feature matrix (per-nominee features).
            y: Binary target (0/1), exactly one 1 per group.
            groups: Required. Ceremony numbers for each row.

        Raises:
            ValueError: If groups is None.
        """
        if groups is None:
            raise ValueError("ConditionalLogitModel requires groups (ceremony numbers)")

        X_scaled = self.scaler.fit_transform(X)

        model = _StatsmodelsConditionalLogit(
            endog=np.asarray(y.values, dtype=float),
            exog=X_scaled,
            groups=groups,
        )

        result = model.fit_regularized(
            method="elastic_net",
            alpha=self.config.alpha,
            L1_wt=self.config.L1_wt,
        )

        self.params_ = np.asarray(result.params)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict probabilities via per-group softmax of linear predictor.

        At test time, typically predict one ceremony at a time (single group).
        The softmax over a single group produces properly calibrated probs
        summing to 1.

        Args:
            X: Feature matrix.
            groups: Required. Ceremony numbers for each row.

        Returns:
            1-D array of P(win) for each row, summing to 1 within each group.
        """
        if not self._is_fitted or self.params_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        if groups is None:
            raise ValueError("ConditionalLogitModel requires groups for prediction")

        X_scaled = self.scaler.transform(X)
        linear_pred = X_scaled @ self.params_

        # Softmax per group
        probs = np.empty(len(X), dtype=float)
        for g in np.unique(groups):
            mask = groups == g
            logits = linear_pred[mask]
            logits = logits - logits.max()  # numerical stability
            exp_logits = np.exp(logits)
            probs[mask] = exp_logits / exp_logits.sum()

        return probs

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature coefficients as importance (single shared β vector).

        Returns DataFrame with columns: feature, coefficient, abs_coefficient, importance.
        The 'importance' column is an alias for abs_coefficient, providing a unified
        column name across all model types.
        """
        if not self._is_fitted or self.params_ is None:
            raise RuntimeError("Model must be fitted first")
        coefs = self.params_
        abs_coefs = np.abs(coefs)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coefficient": abs_coefs,
                "importance": abs_coefs,
            }
        ).sort_values("importance", ascending=False)
