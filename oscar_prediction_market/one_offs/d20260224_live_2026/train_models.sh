#!/usr/bin/env bash
# Train temporal model snapshots for 2026: all categories x model types x snapshot dates.
#
# 9 categories x 4 model types x 6 snapshots (as of Mar 2) = 216 model trainings.
# Trains on 2000-2025 (includes 2025 results) and predicts 2026 nominees.
#
# Key differences from 2025 training script:
# - TRAIN_YEARS="2000-2025" (was 2000-2024)
# - TEST_YEARS="2026" (was 2025)
# - CEREMONY_YEAR="2026" (was 2025)
# - SNAPSHOT_DATES are 2026 dates (4 snapshots vs 6 for 2025)
# - Datasets from storage/d20260224_live_2026/datasets/
# - Models saved to storage/d20260224_live_2026/models/
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260224_live_2026/train_models.sh
#
# Output: storage/d20260224_live_2026/models/{category}/{model_short}/{as_of_date}/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "Git commit: $(git rev-parse HEAD)"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Dirty: $(git diff --shortstat)"
echo "---"

EXP_DIR="storage/d20260224_live_2026"
CONFIGS="$EXP_DIR/configs"
DATASETS_DIR="$EXP_DIR/datasets"
MODELS_DIR="$EXP_DIR/models"
BUILD_MODULE="oscar_prediction_market.modeling.build_model"

# 2026 post-nomination snapshot keys (all precursors resolved as of Mar 9)
SNAPSHOT_KEYS=(
    "2026-01-22_oscar_noms"  # Oscar nominations
    "2026-02-07_dga"         # DGA winner
    "2026-02-21_annie"       # Annie winner
    "2026-02-22_bafta"       # BAFTA winner
    "2026-02-28_pga"         # PGA winner
    "2026-03-01_sag"         # SAG winner
    "2026-03-08_asc"         # ASC winner
    "2026-03-08_wga"         # WGA winner
)

CATEGORIES=(
    best_picture
    directing
    actor_leading
    actress_leading
    actor_supporting
    actress_supporting
    original_screenplay
    cinematography
    animated_feature
)

# Model types: (short_name, model_type_enum, feature_family, param_grid_short)
MODEL_CONFIGS=(
    "lr|logistic_regression|lr|lr"
    "clogit|conditional_logit|lr|conditional_logit"
    "gbt|gradient_boosting|gbt|gbt"
    "cal_sgbt|calibrated_softmax_gbt|gbt|calibrated_softmax_gbt"
)

IMPORTANCE_THRESHOLD="0.90"
CEREMONY_YEAR="2026"
TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"

echo "============================================================"
echo "Train 2026 Temporal Model Snapshots"
echo "Categories:     ${#CATEGORIES[@]}"
echo "Model types:    ${#MODEL_CONFIGS[@]}"
echo "Snapshots:      ${#SNAPSHOT_KEYS[@]}"
echo "Total trainings: $(( ${#CATEGORIES[@]} * ${#MODEL_CONFIGS[@]} * ${#SNAPSHOT_KEYS[@]} ))"
echo "Train years:    $TRAIN_YEARS"
echo "Test year:      $TEST_YEARS"
echo "Output:         $MODELS_DIR"
echo "============================================================"

# Verify configs exist
if [[ ! -d "$CONFIGS/features" ]] || [[ ! -d "$CONFIGS/param_grids" ]]; then
    echo "ERROR: Configs not found at $CONFIGS"
    echo "Run setup_configs.sh first."
    exit 1
fi

SUCCEEDED=0
SKIPPED=0
FAILED=0
TOTAL=0

for cat_slug in "${CATEGORIES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Category: $cat_slug"
    echo "=========================================="

    for model_spec in "${MODEL_CONFIGS[@]}"; do
        IFS='|' read -r short_name model_type feat_family grid_short <<< "$model_spec"

        # Feature config for this category + model family
        feature_config="$CONFIGS/features/${cat_slug}_${feat_family}_full.json"
        if [[ ! -f "$feature_config" ]]; then
            echo "  ERROR: Missing feature config: $feature_config"
            FAILED=$((FAILED + ${#SNAPSHOT_KEYS[@]}))
            TOTAL=$((TOTAL + ${#SNAPSHOT_KEYS[@]}))
            continue
        fi

        # Param grid
        param_grid="$CONFIGS/param_grids/${grid_short}_grid.json"
        if [[ ! -f "$param_grid" ]]; then
            echo "  ERROR: Missing param grid: $param_grid"
            FAILED=$((FAILED + ${#SNAPSHOT_KEYS[@]}))
            TOTAL=$((TOTAL + ${#SNAPSHOT_KEYS[@]}))
            continue
        fi

        # CV split
        cv_split="$CONFIGS/cv_splits/leave_one_year_out.json"

        echo ""
        echo "  Model: $short_name ($model_type)"

        for snap_key in "${SNAPSHOT_KEYS[@]}"; do
            TOTAL=$((TOTAL + 1))
            as_of_date="${snap_key:0:10}"

            # Dataset path
            raw_path="$DATASETS_DIR/$cat_slug/$snap_key/oscar_${cat_slug}_raw.json"
            if [[ ! -f "$raw_path" ]]; then
                echo "    SKIP $snap_key: dataset not found at $raw_path"
                FAILED=$((FAILED + 1))
                continue
            fi

            # Output directory
            model_output_dir="$MODELS_DIR/$cat_slug/$short_name/$snap_key"
            run_name="${short_name}_${snap_key}"

            # Check if already trained (look for predictions file)
            pred_file="$model_output_dir/$run_name/5_final_predict/predictions_test.csv"
            pred_file_alt="$model_output_dir/$run_name/2_final_predict/predictions_test.csv"
            if [[ -f "$pred_file" ]] || [[ -f "$pred_file_alt" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "    Training: $cat_slug / $short_name / $snap_key"

            if uv run python -m "$BUILD_MODULE" \
                --name "$run_name" \
                --param-grid "$param_grid" \
                --feature-config "$feature_config" \
                --cv-split "$cv_split" \
                --train-years "$TRAIN_YEARS" \
                --test-years "$TEST_YEARS" \
                --output-dir "$model_output_dir" \
                --raw-path "$raw_path" \
                --as-of-date "$as_of_date" \
                --ceremony-year "$CEREMONY_YEAR" \
                --n-jobs 8 \
                --feature-selection \
                --importance-threshold "$IMPORTANCE_THRESHOLD"; then
                SUCCEEDED=$((SUCCEEDED + 1))
            else
                echo "    FAILED: $cat_slug / $short_name / $snap_key"
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "Training Summary"
echo "============================================================"
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "WARNING: $FAILED trainings failed!"
    echo "Re-run this script to retry failed trainings (idempotent)."
    exit 1
fi

echo ""
echo "All 2026 models trained successfully!"
