#!/bin/bash
# Phase 3: Feature Selection Method Ablation
#
# Ablates feature selection parameters across 9 categories × 2 models (clogit, cal_sgbt).
# All runs use the "full" feature config — the winning strategy from Rounds 1-2.
#
# Dimensions ablated:
#   1. Importance threshold:  0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0 (all nonzero)
#   2. Max features cap (at threshold 0.80):  3, 5
#   3. No feature selection baseline (full config, no pruning)
#
# Total: (7 thresholds + 2 max_features + 1 no_fs) × 2 models × 9 categories = 180 runs
#
# The updated clogit grid includes alpha=0.001 (shifted from [0.002, ..., 0.5]
# to [0.001, ..., 0.1]) to probe the low-regularization frontier.
#
# Output structure:
#   storage/d20260220_feature_ablation/{category}/fs_ablation/{config_name}/
#     1_full_cv/              Full-feature CV (with_fs runs)
#     2_full_train/           Full-feature train for importance extraction
#     3_selected_features.json  Selected features
#     4_selected_cv/          Selected-feature CV (honest estimate)
#     5_final_predict/        Final predictions
#
# Env var overrides:
#   CATEGORIES_OVERRIDE  — space-separated category list (default: all 9)
#   MODELS               — space-separated models: "clogit cal_sgbt" (default: both)
#   N_JOBS               — parallel jobs (default: 10)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260220_feature_ablation/run_phase3_fs_ablation.sh \
#       2>&1 | tee storage/d20260220_feature_ablation/run_phase3.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

EXP_DIR="storage/d20260220_feature_ablation"
DATASET_DIR="storage/d20260218_build_all_datasets"
CONFIGS="$EXP_DIR/configs"
MODULE="oscar_prediction_market.modeling.build_model"

TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
AS_OF_DATE="2026-02-20"
N_JOBS=${N_JOBS:-10}

# Feature selection thresholds to ablate
THRESHOLDS=(0.50 0.60 0.70 0.80 0.90 0.95 1.00)

# Max features caps (tested at threshold 0.80 only)
MAX_FEATURES_CAPS=(3 5)

# Categories
if [[ -n "${CATEGORIES_OVERRIDE:-}" ]]; then
    read -ra CATEGORIES <<< "$CATEGORIES_OVERRIDE"
else
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
fi

# Models: clogit (uses LR features) and cal_sgbt (uses GBT features)
MODELS_LIST=(${MODELS:-clogit cal_sgbt})

# Map category to dataset file
dataset_file_for() {
    local cat="$1"
    echo "${cat}/oscar_${cat}_raw.json"
}

# ============================================================================
# Validation
# ============================================================================

echo "============================================================"
echo "Phase 3: Feature Selection Method Ablation"
echo "============================================================"

CV_SPLIT="$CONFIGS/cv_splits/leave_one_year_out.json"
CLOGIT_GRID="$CONFIGS/param_grids/conditional_logit_grid.json"
CAL_SGBT_GRID="$CONFIGS/param_grids/calibrated_softmax_gbt_grid.json"

for f in "$CV_SPLIT" "$CLOGIT_GRID" "$CAL_SGBT_GRID"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

# Verify updated alpha grid
ALPHA_CHECK=$(python3 -c "import json; d=json.load(open('$CLOGIT_GRID')); print(min(c['alpha'] for c in d['grid']))")
if [[ "$ALPHA_CHECK" != "0.001" ]]; then
    echo "ERROR: Clogit grid does not contain alpha=0.001 (min=$ALPHA_CHECK)"
    echo "  Run run_generate_configs.sh to regenerate grids."
    exit 1
fi

for category in "${CATEGORIES[@]}"; do
    raw_path="$DATASET_DIR/$(dataset_file_for "$category")"
    if [[ ! -f "$raw_path" ]]; then
        echo "ERROR: Missing dataset: $raw_path"
        exit 1
    fi
    for prefix in lr gbt; do
        feat="$CONFIGS/features/$category/${prefix}_full.json"
        if [[ ! -f "$feat" ]]; then
            echo "ERROR: Missing feature config: $feat"
            exit 1
        fi
    done
done

echo "  Categories: ${#CATEGORIES[@]} (${CATEGORIES[*]})"
echo "  Models: ${MODELS_LIST[*]}"
echo "  Thresholds: ${THRESHOLDS[*]}"
echo "  Max features caps (at t=0.80): ${MAX_FEATURES_CAPS[*]}"
echo "  Clogit grid alpha min: $ALPHA_CHECK"
echo "All prerequisites validated."
echo ""

# ============================================================================
# Run helper
# ============================================================================

RUN_COUNT=0
SKIP_COUNT=0

run_fs_ablation() {
    local category="$1"
    local name="$2"
    local param_grid="$3"
    local feature_config="$4"
    local raw_path="$5"
    local fs_args="$6"

    local run_dir="$EXP_DIR/$category/fs_ablation/$name"
    if [[ -d "$run_dir" ]]; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi

    echo "  [$category/fs_ablation] $name"
    # shellcheck disable=SC2086
    uv run python -m "$MODULE" \
        --name "$name" \
        --param-grid "$param_grid" \
        --feature-config "$feature_config" \
        --cv-split "$CV_SPLIT" \
        --train-years "$TRAIN_YEARS" \
        --test-years "$TEST_YEARS" \
        --output-dir "$EXP_DIR/$category/fs_ablation" \
        --raw-path "$raw_path" \
        --as-of-date "$AS_OF_DATE" \
        --n-jobs "$N_JOBS" \
        $fs_args

    RUN_COUNT=$((RUN_COUNT + 1))
}

# ============================================================================
# Run experiments
# ============================================================================

START_TIME=$(date +%s)

for category in "${CATEGORIES[@]}"; do
    raw_path="$DATASET_DIR/$(dataset_file_for "$category")"

    echo ""
    echo "============================================================"
    echo "Category: $category"
    echo "============================================================"

    for model in "${MODELS_LIST[@]}"; do
        # Select feature config and param grid based on model family
        case "$model" in
            clogit)
                feature_config="$CONFIGS/features/$category/lr_full.json"
                param_grid="$CLOGIT_GRID"
                ;;
            cal_sgbt)
                feature_config="$CONFIGS/features/$category/gbt_full.json"
                param_grid="$CAL_SGBT_GRID"
                ;;
            *)
                echo "ERROR: Unknown model $model"
                exit 1
                ;;
        esac

        echo ""
        echo "--- $category / $model ---"

        # 1. No feature selection baseline
        run_fs_ablation "$category" "${model}_full_nofs" \
            "$param_grid" "$feature_config" "$raw_path" ""

        # 2. Threshold sweep (no max_features cap)
        for thresh in "${THRESHOLDS[@]}"; do
            # Format threshold for naming: 0.80 -> t080
            thresh_name=$(printf "t%03d" "$(echo "$thresh * 100" | bc | cut -d. -f1)")
            run_fs_ablation "$category" "${model}_full_${thresh_name}" \
                "$param_grid" "$feature_config" "$raw_path" \
                "--feature-selection --importance-threshold $thresh"
        done

        # 3. Max features cap sweep (at threshold 0.80)
        for maxf in "${MAX_FEATURES_CAPS[@]}"; do
            run_fs_ablation "$category" "${model}_full_t080_m${maxf}" \
                "$param_grid" "$feature_config" "$raw_path" \
                "--feature-selection --importance-threshold 0.80 --max-features $maxf"
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo "============================================================"
echo "Phase 3: Feature Selection Ablation complete"
echo "============================================================"
echo "  Runs completed: $RUN_COUNT"
echo "  Runs skipped:   $SKIP_COUNT"
echo "  Elapsed time:   ${ELAPSED_MIN}m ${ELAPSED}s"
echo "  Results:        $EXP_DIR/*/fs_ablation/"
echo "============================================================"
