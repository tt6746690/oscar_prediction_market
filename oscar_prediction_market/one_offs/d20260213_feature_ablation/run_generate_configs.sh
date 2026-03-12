#!/bin/bash
# Generate all configs for the feature ablation experiment
#
# Copies base configs from modeling/configs/ and generates ablation configs
# into storage/d20260213_feature_ablation/configs/. All experiment scripts
# reference configs from this output directory.
#
# This script is idempotent — it skips files already present (cp -n).
# To regenerate, delete the target configs/ directory first.
#
# Source configs (modeling/configs/):
#   features/: lr_{full,standard,minimal}.json, gbt_{full,standard,minimal}.json
#   param_grids/: lr_grid.json, gbt_grid.json
#   cv_splits/: 6 configs
#
# Generated experiment configs (storage/.../configs/):
#   features/: base configs + ablation configs (additive, leave-one-out, single-group, BASE)
#   param_grids/: lr_grid_wide.json (renamed from lr_grid), gbt_grid.json
#   cv_splits/: copied from source
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260213_feature_ablation/run_generate_configs.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Paths
# ============================================================================

SRC_CONFIGS="oscar_prediction_market/modeling/configs"
OUTPUT_DIR="storage/d20260213_feature_ablation"
CONFIGS="$OUTPUT_DIR/configs"
ABLATION_MODULE="oscar_prediction_market.modeling.generate_feature_ablation_configs"

mkdir -p "$CONFIGS/param_grids" "$CONFIGS/features" "$CONFIGS/cv_splits"

echo "============================================================"
echo "Step 1: Copy base configs from modeling/configs/"
echo "============================================================"

# Param grids (rename lr_grid.json -> lr_grid_wide.json for experiment clarity)
cp -n "$SRC_CONFIGS/param_grids/lr_grid.json" "$CONFIGS/param_grids/lr_grid_wide.json" 2>/dev/null || true
cp -n "$SRC_CONFIGS/param_grids/gbt_grid.json" "$CONFIGS/param_grids/gbt_grid.json" 2>/dev/null || true

# Base feature configs
for f in "$SRC_CONFIGS"/features/*.json; do
    cp -n "$f" "$CONFIGS/features/" 2>/dev/null || true
done

# CV splits
for f in "$SRC_CONFIGS"/cv_splits/*.json; do
    cp -n "$f" "$CONFIGS/cv_splits/" 2>/dev/null || true
done

echo "  Base configs copied."
echo ""

# ============================================================================
# Step 2: Generate group ablation configs via generate_feature_ablation_configs
# ============================================================================

echo "============================================================"
echo "Step 2: Generate ablation feature configs"
echo "============================================================"

FEATURES_DIR="$CONFIGS/features"

# Count existing ablation configs to decide if we need to generate
existing_count=$(find "$FEATURES_DIR" -maxdepth 1 -name '*_additive_*' -o -name '*_without_*' -o -name '*_only_*' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$existing_count" -ge 50 ]]; then
    echo "  SKIP: $existing_count ablation configs already exist."
else
    echo "  Generating LR ablation configs..."
    uv run python -m "$ABLATION_MODULE" \
        --model-type logistic_regression \
        --ablation-types all \
        --output-dir "$FEATURES_DIR" \
        -v

    echo ""
    echo "  Generating GBT ablation configs..."
    uv run python -m "$ABLATION_MODULE" \
        --model-type gradient_boosting \
        --ablation-types all \
        --output-dir "$FEATURES_DIR" \
        -v
fi

echo ""

# ============================================================================
# Step 3: Generate custom BASE additive_3 configs
# ============================================================================
# BASE = features from the original lr_standard/gbt_standard (29/24 feature sets)
#        restricted to the winning 3-group subset
# FULL = features from FULL definitions (47/45 features) restricted to groups
#        (generated in step 2 as additive_3_oscar_nominations)

echo "============================================================"
echo "Step 3: Generate custom additive configs (BASE variants)"
echo "============================================================"

# LR additive_3 BASE: precursor_winners + precursor_nominations + oscar_nominations
# Uses only features from the original lr_standard (22 features — no acting_nomination_count,
# major_category_count, nominees_in_year which are FULL-only expansions)
if [[ ! -f "$FEATURES_DIR/lr_additive_3_base.json" ]]; then
cat > "$FEATURES_DIR/lr_additive_3_base.json" << 'EOF'
{
  "name": "lr_additive_3_base",
  "description": "additive_3 with BASE features only (from original lr_standard). Groups: precursor_winners, precursor_nominations, oscar_nominations. 22 features.",
  "model_type": "logistic_regression",
  "features": [
    "pga_winner",
    "dga_winner",
    "sag_ensemble_winner",
    "bafta_winner",
    "golden_globe_winner",
    "critics_choice_winner",
    "has_pga_dga_combo",
    "precursor_wins_count",
    "pga_nominee",
    "dga_nominee",
    "sag_ensemble_nominee",
    "bafta_nominee",
    "golden_globe_nominee",
    "critics_choice_nominee",
    "has_pga_dga_nomination_combo",
    "precursor_nominations_count",
    "oscar_total_nominations",
    "has_director_nomination",
    "has_editing_nomination",
    "has_acting_nomination",
    "has_screenplay_nomination",
    "nominations_percentile_in_year"
  ]
}
EOF
echo "  Created lr_additive_3_base.json"
else
echo "  SKIP: lr_additive_3_base.json exists"
fi

# GBT additive_3 BASE: same groups but using original gbt_standard features (17 features)
if [[ ! -f "$FEATURES_DIR/gbt_additive_3_base.json" ]]; then
cat > "$FEATURES_DIR/gbt_additive_3_base.json" << 'EOF'
{
  "name": "gbt_additive_3_base",
  "description": "additive_3 with BASE features only (from original gbt_standard). Groups: precursor_winners, precursor_nominations, oscar_nominations. 17 features.",
  "model_type": "gradient_boosting",
  "features": [
    "pga_winner",
    "dga_winner",
    "sag_ensemble_winner",
    "bafta_winner",
    "golden_globe_winner",
    "critics_choice_winner",
    "pga_nominee",
    "dga_nominee",
    "sag_ensemble_nominee",
    "bafta_nominee",
    "golden_globe_nominee",
    "critics_choice_nominee",
    "oscar_total_nominations",
    "has_director_nomination",
    "has_editing_nomination",
    "has_acting_nomination",
    "has_screenplay_nomination"
  ]
}
EOF
echo "  Created gbt_additive_3_base.json"
else
echo "  SKIP: gbt_additive_3_base.json exists"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================"
echo "Config generation complete"
echo "============================================================"
echo "  Output: $CONFIGS/"
echo ""
echo "  Param grids:  $(ls -1 "$CONFIGS/param_grids/" | wc -l | tr -d ' ') files"
echo "  Features:     $(ls -1 "$CONFIGS/features/"*.json 2>/dev/null | wc -l | tr -d ' ') configs"
echo "  CV splits:    $(ls -1 "$CONFIGS/cv_splits/" | wc -l | tr -d ' ') files"
