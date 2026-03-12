"""Config loading and validation utilities."""

import logging
from pathlib import Path

from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)
from oscar_prediction_market.modeling.models.configs import (
    BaggedClassifierConfig,
    ModelConfig,
    ModelConfigGrid,
    model_config_adapter,
)
from oscar_prediction_market.modeling.models.types import ModelType

logger = logging.getLogger(__name__)


def load_model_config(config_path: Path) -> ModelConfig:
    """Load a single model config using Pydantic TypeAdapter.

    Args:
        config_path: Path to JSON config file

    Returns:
        Validated ModelConfig (LogisticRegressionConfig or GradientBoostingConfig)
    """
    with open(config_path, "rb") as f:
        return model_config_adapter.validate_json(f.read())


def load_model_config_grid(config_path: Path) -> list[ModelConfig]:
    """Load model config grid from JSON file.

    Expected format (grid format only):
    {
        "model_type": "lr",
        "grid": [
            {"C": 0.01, "l1_ratio": 0.0, ...},
            {"C": 0.1, "l1_ratio": 0.0, ...},
            ...
        ]
    }

    Args:
        config_path: Path to JSON config file

    Returns:
        List of validated ModelConfig objects

    Raises:
        ValueError: If grid format is invalid or model_type unknown
    """
    with open(config_path, "rb") as f:
        grid_config = ModelConfigGrid.model_validate_json(f.read())

    configs: list[ModelConfig] = []
    for i, params in enumerate(grid_config.grid):
        # Add model_type to each config for validation
        full_config = {"model_type": grid_config.model_type, **params}
        try:
            config = model_config_adapter.validate_python(full_config)
            configs.append(config)
        except Exception as e:
            raise ValueError(f"Invalid config at grid index {i}: {e}") from e

    return configs


def validate_model_feature_consistency(
    model_configs: list[ModelConfig],
    feature_family: FeatureFamily | None,
) -> FeatureFamily:
    """Validate that all model configs have consistent model_type and match feature config.

    Args:
        model_configs: List of model configurations.
        feature_family: Feature family from feature config (FeatureFamily.LR,
            FeatureFamily.GBT, or None for universal).

    Returns:
        Inferred FeatureFamily for feature engineering.

    Raises:
        ValueError: If inconsistencies detected.
    """
    if not model_configs:
        raise ValueError("Empty model config list")

    # Check all configs have same model_type
    model_types = {c.model_type for c in model_configs}
    if len(model_types) > 1:
        raise ValueError(f"Mixed model types in config: {model_types}")

    config = model_configs[0]

    # Determine effective feature family from model config
    if isinstance(config, BaggedClassifierConfig):
        effective_family = ModelType(config.base_model_type).feature_family
    else:
        effective_family = ModelType(config.model_type).feature_family

    # Validate against feature config if specified
    if feature_family is not None and feature_family != effective_family:
        raise ValueError(
            f"Feature config family '{feature_family}' doesn't match "
            f"model config's effective family '{effective_family}'"
        )

    return effective_family
