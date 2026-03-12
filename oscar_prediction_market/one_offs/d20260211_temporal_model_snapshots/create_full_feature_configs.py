"""Generate full-feature configs for temporal model snapshots.

Creates feature config JSON files listing ALL features for each model type.
These configs are used by build_model.py's feature selection pipeline, which
then prunes to nonzero-importance features automatically.

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260211_temporal_model_snapshots.create_full_feature_configs \
        --output-dir storage/d20260211_temporal_model_snapshots/configs/features
"""

import argparse
import json
from pathlib import Path

from oscar_prediction_market.modeling.feature_engineering import (
    ABLATION_CATEGORIES,
    FeatureFamily,
    get_all_features,
)


def _all_features_for_family(feature_family: FeatureFamily) -> list[str]:
    """Union of features across all categories for a feature family."""
    all_feats: set[str] = set()
    for category in ABLATION_CATEGORIES:
        all_feats.update(get_all_features(feature_family, category))
    return sorted(all_feats)


def create_feature_config(feature_family: FeatureFamily, output_dir: Path) -> Path:
    """Create a full-feature config JSON for a feature family.

    Args:
        feature_family: FeatureFamily.LR or FeatureFamily.GBT
        output_dir: Directory to write the config file

    Returns:
        Path to the created config file
    """
    feature_names = _all_features_for_family(feature_family)
    short = feature_family.value

    config = {
        "name": f"{short}_full",
        "description": (
            f"All {len(feature_names)} features for {short}. "
            f"Feature selection via build_model.py will prune to nonzero-importance."
        ),
        "feature_family": short,
        "features": feature_names,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{short}_full.json"
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Created {output_path} ({len(feature_names)} features)")
    return output_path


def main() -> None:
    """Generate full-feature configs for LR and GBT."""
    parser = argparse.ArgumentParser(
        description="Generate full-feature config JSONs for temporal model snapshots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write feature config JSON files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("Creating full-feature configs...")

    for ff in [FeatureFamily.LR, FeatureFamily.GBT]:
        create_feature_config(ff, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
