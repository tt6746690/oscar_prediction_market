"""Feature registry — flat mapping from feature name to FeatureDefinition.

The FEATURE_REGISTRY is the canonical lookup for feature definitions by name.
Feature groups and configs reference features by string name;
resolve_features() converts those names to FeatureDefinition objects.
"""

from oscar_prediction_market.modeling.feature_engineering.features import (
    ALL_FEATURE_LISTS,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureDefinition,
)

# ============================================================================
# FEATURE REGISTRY (all features by name)
# ============================================================================

FEATURE_REGISTRY: dict[str, FeatureDefinition] = {}

for _feat_list in ALL_FEATURE_LISTS:
    for _feat in _feat_list:
        if _feat.name not in FEATURE_REGISTRY:
            FEATURE_REGISTRY[_feat.name] = _feat


def resolve_features(feature_names: list[str]) -> list[FeatureDefinition]:
    """Resolve feature names to FeatureDefinition objects.

    Args:
        feature_names: List of feature names (matching FEATURE_REGISTRY keys).

    Returns:
        List of FeatureDefinition objects in the same order.

    Raises:
        ValueError: If any feature name is not found in the registry.
    """
    result = []
    missing = []
    for name in feature_names:
        if name in FEATURE_REGISTRY:
            result.append(FEATURE_REGISTRY[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Unknown features: {missing}. Available: {sorted(FEATURE_REGISTRY.keys())}"
        )
    return result


def get_feature_names(features: list[FeatureDefinition]) -> list[str]:
    """Get list of feature column names."""
    return [f.name for f in features]
