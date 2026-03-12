#!/bin/bash
# Group feature ablation experiment
# Tests leave-one-out, additive, and single-group ablation configs
#
# Structure:
#   For each mode (no feature selection / with feature selection):
#     For each model (LR / GBT):
#       1 full baseline + 8 leave-one-out + 8 additive + 8 single-group = 25 configs
#     = 50 configs per mode
#   = 100 runs total
#
# Feature groups (8):
#   oscar_nominations, precursor_winners, precursor_nominations,
#   critic_scores, commercial, timing, film_metadata, voting_system
#
# Modes:
#   no_fs   — no --feature-selection (all features from config used as-is)
#   with_fs — --feature-selection --importance-threshold 0.80
#
# Prerequisite: run_single_date.sh (copies configs to experiment dir)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260213_feature_ablation/run_group_ablation.sh \
#       2>&1 | tee storage/d20260213_feature_ablation/group_ablation/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR="storage/d20260213_feature_ablation"
ABLATION_DIR="$OUTPUT_DIR/group_ablation"
mkdir -p "$ABLATION_DIR"

RAW_PATH="storage/d20260201_build_dataset/oscar_best_picture_raw.json"
MODULE="oscar_prediction_market.modeling.build_model"
SCRIPTS_DIR="oscar_prediction_market/one_offs/d20260213_feature_ablation"

TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
AS_OF_DATE="2026-02-07"
N_JOBS=4

# Generate configs (idempotent — skips if already present)
bash "$SCRIPTS_DIR/run_generate_configs.sh"

CONFIGS="$OUTPUT_DIR/configs"
CV_SPLIT="$CONFIGS/cv_splits/leave_one_year_out.json"
LR_GRID="$CONFIGS/param_grids/lr_grid_wide.json"
GBT_GRID="$CONFIGS/param_grids/gbt_grid.json"
LR_FULL="$CONFIGS/features/lr_full.json"
GBT_FULL="$CONFIGS/features/gbt_full.json"

# ============================================================================
# Validation
# ============================================================================

for f in "$RAW_PATH" "$CV_SPLIT" "$LR_GRID" "$GBT_GRID" "$LR_FULL" "$GBT_FULL"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f"
        echo "  (Run run_single_date.sh first to copy configs)"
        exit 1
    fi
done
echo "All required files validated."
echo ""

# ============================================================================
# Run ablation experiments
# ============================================================================

echo "============================================================"
echo "Running group ablation experiments"
echo "============================================================"

FEATURES_DIR="$CONFIGS/features"

run_ablation() {
    local mode="$1"         # no_fs or with_fs
    local name="$2"
    local param_grid="$3"
    local feature_config="$4"

    local run_dir="$ABLATION_DIR/$mode/$name"
    if [[ -d "$run_dir" ]]; then
        echo "  SKIP (exists): [$mode] $name"
        return
    fi

    local fs_args=""
    if [[ "$mode" == "with_fs" ]]; then
        fs_args="--feature-selection --importance-threshold 0.80"
    fi

    echo "  Running: [$mode] $name"
    # shellcheck disable=SC2086
    uv run python -m "$MODULE" \
        --name "$name" \
        --param-grid "$param_grid" \
        --feature-config "$feature_config" \
        --cv-split "$CV_SPLIT" \
        --train-years "$TRAIN_YEARS" \
        --test-years "$TEST_YEARS" \
        --output-dir "$ABLATION_DIR/$mode" \
        --raw-path "$RAW_PATH" \
        --as-of-date "$AS_OF_DATE" \
        --n-jobs "$N_JOBS" \
        $fs_args
    echo "  -> [$mode] $name complete"
}

for mode in no_fs with_fs; do
    echo ""
    echo "============================================================"
    echo "Mode: $mode"
    echo "============================================================"

    # --- LR: baseline + ablation configs ---
    echo ""
    echo "--- LR models ---"
    run_ablation "$mode" "lr_full" "$LR_GRID" "$LR_FULL"

    for config_file in "$FEATURES_DIR"/lr_{additive,without,only}_*.json; do
        config_name=$(basename "$config_file" .json)
        run_ablation "$mode" "$config_name" "$LR_GRID" "$config_file"
    done

    # --- GBT: baseline + ablation configs ---
    echo ""
    echo "--- GBT models ---"
    run_ablation "$mode" "gbt_full" "$GBT_GRID" "$GBT_FULL"

    for config_file in "$FEATURES_DIR"/gbt_{additive,without,only}_*.json; do
        config_name=$(basename "$config_file" .json)
        run_ablation "$mode" "$config_name" "$GBT_GRID" "$config_file"
    done
done

echo ""
echo "============================================================"
echo "Group ablation complete (100 runs across 2 modes × 2 models × 25 configs)."
echo "Results: $ABLATION_DIR/"
echo "============================================================"
