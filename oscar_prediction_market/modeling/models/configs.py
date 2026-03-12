"""Model configuration classes (Pydantic).

Each model type has a corresponding config class with all hyperparameters
as required fields (no defaults for semantic parameters).

Also defines the discriminated union ``ModelConfig`` for polymorphic
config loading, and grid/list schemas for hyperparameter tuning.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, RootModel, Tag, TypeAdapter


class LogisticRegressionConfig(BaseModel):
    """Hyperparameters for Logistic Regression.

    All fields required - no defaults to ensure explicit configuration.
    """

    model_type: Literal["logistic_regression"] = Field(..., description="Model type identifier")
    C: float = Field(..., description="Regularization strength (inverse)", gt=0)
    l1_ratio: float = Field(
        ..., description="L1 ratio: 0=L2, 1=L1 (use with solver='saga')", ge=0, le=1
    )
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = Field(
        ..., description="Optimization algorithm"
    )
    max_iter: int = Field(..., description="Maximum iterations", ge=1, le=10000)
    class_weight: str | None = Field(..., description="Class weighting strategy")
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    model_config = {"extra": "forbid"}


class GradientBoostingConfig(BaseModel):
    """Hyperparameters for Gradient Boosting.

    All fields required - no defaults to ensure explicit configuration.
    """

    model_type: Literal["gradient_boosting"] = Field(..., description="Model type identifier")
    n_estimators: int = Field(..., description="Number of trees", ge=1)
    learning_rate: float = Field(..., description="Learning rate", gt=0)
    max_depth: int = Field(..., description="Maximum tree depth")
    min_samples_split: int = Field(..., description="Min samples to split")
    min_samples_leaf: int = Field(..., description="Min samples in leaf")
    subsample: float = Field(..., description="Subsample ratio", gt=0, le=1)
    random_state: int = Field(..., description="Random seed for reproducibility")

    model_config = {"extra": "forbid"}


class XGBoostConfig(BaseModel):
    """Hyperparameters for XGBoost.

    All fields required - no defaults to ensure explicit configuration.
    """

    model_type: Literal["xgboost"] = Field(..., description="Model type identifier")
    n_estimators: int = Field(..., description="Number of boosting rounds", ge=1)
    learning_rate: float = Field(..., description="Learning rate (eta)", gt=0)
    max_depth: int = Field(..., description="Maximum tree depth")
    min_child_weight: int = Field(..., description="Minimum sum of instance weight in child")
    subsample: float = Field(..., description="Subsample ratio of training instances", gt=0, le=1)
    colsample_bytree: float = Field(
        ..., description="Subsample ratio of columns per tree", gt=0, le=1
    )
    reg_alpha: float = Field(..., description="L1 regularization term", ge=0)
    reg_lambda: float = Field(..., description="L2 regularization term", ge=0)
    random_state: int = Field(..., description="Random seed for reproducibility")

    model_config = {"extra": "forbid"}


class ConditionalLogitConfig(BaseModel):
    """Hyperparameters for Conditional Logistic Regression (McFadden's choice model).

    Models the choice among K alternatives within each choice set (ceremony year):

        P(win_i | year) = exp(x_i^T β) / Σ_j exp(x_j^T β)

    Probabilities sum to 1 by construction. Uses statsmodels ConditionalLogit
    with elastic net regularization via fit_regularized().

    All fields required — no defaults to ensure explicit configuration.
    """

    model_type: Literal["conditional_logit"] = Field(..., description="Model type identifier")
    alpha: float = Field(
        ...,
        description="Overall regularization penalty strength",
        ge=0,
    )
    L1_wt: float = Field(
        ...,
        description="L1 weight in elastic net: 0=pure L2, 1=pure L1",
        ge=0,
        le=1,
    )

    model_config = {"extra": "forbid"}


class SoftmaxGBTConfig(BaseModel):
    """Configuration for multi-class GBT with softmax objective.

    Two-stage approach:
    1. Binary model ranks all nominees per ceremony
    2. Top-K nominees enter multi-class XGBoost with softmax objective
    3. Outputs calibrated probabilities summing to 1

    The feature space is K × F (one feature set per slot, sorted by
    precursor signal strength). With ~25 training years and K×F features,
    strong regularization is critical.
    """

    model_type: Literal["softmax_gbt"] = Field(..., description="Model type identifier")
    n_estimators: int = Field(..., description="Number of boosting rounds", ge=1)
    learning_rate: float = Field(..., description="Learning rate (eta)", gt=0)
    max_depth: int = Field(..., description="Maximum tree depth")
    min_child_weight: int = Field(..., description="Minimum sum of instance weight in child")
    subsample: float = Field(..., description="Subsample ratio of training instances", gt=0, le=1)
    colsample_bytree: float = Field(
        ..., description="Subsample ratio of columns per tree", gt=0, le=1
    )
    reg_alpha: float = Field(..., description="L1 regularization term", ge=0)
    reg_lambda: float = Field(..., description="L2 regularization term", ge=0)
    top_k: int = Field(
        ...,
        description="Number of top nominees to include per ceremony (fixed K)",
        ge=2,
    )
    random_state: int = Field(..., description="Random seed for reproducibility")

    model_config = {"extra": "forbid"}


class CalibratedSoftmaxGBTConfig(BaseModel):
    """Configuration for Calibrated Softmax GBT.

    Wraps a binary GBT model with temperature-scaled softmax post-processing.
    The binary GBT is trained on stacked nominee-level data (F features, ~250 rows),
    then at prediction time probabilities are converted to log-odds and passed
    through a per-group softmax with learned temperature T:

        p_hat_i = exp(logit(p_i) / T) / sum_j exp(logit(p_j) / T)

    This differs from naive normalization (p_i / sum p_j) — temperature-scaled
    softmax operates on log-odds (preserving score ratios) and has a tunable
    parameter T that controls sharpness:
        - T < 1: sharper (more confident, frontrunner gets more mass)
        - T = 1: equivalent to converting binary probs to log-odds then softmax
        - T > 1: smoother (more uniform, probability mass spreads out)

    Temperature is tuned externally via the hyperparameter tuning infrastructure,
    not inside fit(). This keeps the model stateless and avoids nested CV.
    """

    model_type: Literal["calibrated_softmax_gbt"] = Field(..., description="Model type identifier")
    n_estimators: int = Field(..., description="Number of trees", ge=1)
    learning_rate: float = Field(..., description="Learning rate", gt=0)
    max_depth: int = Field(..., description="Maximum tree depth")
    min_samples_split: int = Field(..., description="Min samples to split")
    min_samples_leaf: int = Field(..., description="Min samples in leaf")
    subsample: float = Field(..., description="Subsample ratio", gt=0, le=1)
    temperature: float = Field(
        ...,
        description="Softmax temperature: <1 sharper, >1 smoother",
        gt=0,
    )
    random_state: int = Field(..., description="Random seed for reproducibility")

    model_config = {"extra": "forbid"}


class BaggedClassifierConfig(BaseModel):
    """Configuration for bootstrap-aggregated (bagged) classifier.

    Wraps any base model config. Trains n_bags models on bootstrap samples
    of the training data and averages their predicted probabilities.

    Note: Does not support ConditionalLogitModel or SoftmaxGBTModel as base
    models because those require group structure that bootstrap sampling
    would violate (can't mix nominees across ceremonies).

    Outputs:
    - Mean predicted probabilities (via predict_proba)
    - Per-bag probability distributions (via predict_proba_distribution)
    """

    model_type: Literal["bagged_classifier"] = Field(..., description="Model type identifier")
    base_model_config: LogisticRegressionConfig | GradientBoostingConfig | XGBoostConfig = Field(
        ..., description="Configuration for each base model instance"
    )
    n_bags: int = Field(..., description="Number of bootstrap samples/models", ge=2, le=1000)
    random_state: int = Field(..., description="Random seed for bootstrap sampling")

    model_config = {"extra": "forbid"}

    @property
    def base_model_type(self) -> str:
        """Model type of the base classifier (for feature engineering routing)."""
        return self.base_model_config.model_type


def _get_model_type_discriminator(v: dict) -> str:
    """Discriminator function for ModelConfig union."""
    if isinstance(v, dict):
        return v.get("model_type", "")
    return getattr(v, "model_type", "")


# Discriminated union for model configs - Pydantic handles type discrimination automatically
ModelConfig = Annotated[
    Annotated[LogisticRegressionConfig, Tag("logistic_regression")]
    | Annotated[GradientBoostingConfig, Tag("gradient_boosting")]
    | Annotated[XGBoostConfig, Tag("xgboost")]
    | Annotated[BaggedClassifierConfig, Tag("bagged_classifier")]
    | Annotated[ConditionalLogitConfig, Tag("conditional_logit")]
    | Annotated[SoftmaxGBTConfig, Tag("softmax_gbt")]
    | Annotated[CalibratedSoftmaxGBTConfig, Tag("calibrated_softmax_gbt")],
    Discriminator(_get_model_type_discriminator),
]


# ============================================================================
# Model Config Grid Schema
# ============================================================================


class ModelConfigGrid(BaseModel):
    """Schema for a grid of model configurations (for hyperparameter tuning).

    JSON format:
    {
        "model_type": "logistic_regression",
        "grid": [
            {"C": 0.01, "l1_ratio": 0.0, ...},
            {"C": 0.1, "l1_ratio": 0.0, ...},
            ...
        ]
    }
    """

    model_type: Literal[
        "logistic_regression",
        "gradient_boosting",
        "xgboost",
        "conditional_logit",
        "softmax_gbt",
        "calibrated_softmax_gbt",
    ] = Field(..., description="Model type for all configs in grid")
    grid: list[dict] = Field(..., description="List of parameter dictionaries", min_length=1)

    model_config = {"extra": "forbid"}


class ModelConfigList(RootModel[list[ModelConfig]]):
    """Pydantic model wrapping list[ModelConfig] for TypeAdapter usage."""

    pass


# TypeAdapters for config loading
model_config_adapter: TypeAdapter[ModelConfig] = TypeAdapter(ModelConfig)
model_config_list_adapter: TypeAdapter[list[ModelConfig]] = TypeAdapter(list[ModelConfig])
