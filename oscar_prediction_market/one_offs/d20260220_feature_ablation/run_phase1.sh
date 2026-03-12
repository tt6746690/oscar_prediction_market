#!/bin/bash
# Multi-category feature ablation experiment (Phase 1)
#
# Runs feature ablation experiments across Oscar categories × models × modes.
#
# Model types:
#   LR family  (uses LR feature configs):  logistic_regression, conditional_logit
#   GBT family (uses GBT feature configs):  gradient_boosting, calibrated_softmax_gbt
#
# Modes:
#   no_fs   — no feature selection (all features from config used as-is)
#   with_fs — feature selection with importance threshold 0.80
#
# Env var overrides:
#   CATEGORIES_OVERRIDE  — space-separated category list (default: all 9)
#   MODES                — space-separated modes (default: "no_fs with_fs")
#   MODELS               — space-separated models (default: "lr clogit gbt cal_sgbt")
#   REDUCED_ABLATION     — if "1", only run: full + additive_{1,2,3} + without_precursor_winners
#   N_JOBS               — parallel jobs (default: 10)
#
# Examples:
#   # Full ablation, all models, all categories
#   bash run_phase1.sh
#
#   # Reduced ablation, 2 models, both modes
#   MODELS="clogit cal_sgbt" REDUCED_ABLATION=1 bash run_phase1.sh
#
#   # Subset of categories, no_fs only
#   CATEGORIES_OVERRIDE="best_picture directing" MODES="no_fs" bash run_phase1.sh
#
# Prerequisites:
#   - Datasets in storage/d20260218_build_all_datasets/{category}/oscar_{category}_raw.json
#   - Run run_generate_configs.sh first (or this script runs it automatically)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260220_feature_ablation/run_phase1.sh \
#       2>&1 | tee storage/d20260220_feature_ablation/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

EXP_DIR="storage/d20260220_feature_ablation"
DATASET_DIR="storage/d20260218_build_all_datasets"
CONFIGS="$EXP_DIR/configs"
SCRIPTS_DIR="oscar_prediction_market/one_offs/d20260220_feature_ablation"
MODULE="oscar_prediction_market.modeling.build_model"

TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
AS_OF_DATE="2026-02-20"
N_JOBS=${N_JOBS:-10}

# Override via env: CATEGORIES="best_picture directing" MODES="no_fs" bash run_phase1.sh
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

MODES=(${MODES:-no_fs with_fs})

# Models to run (subset of: lr clogit gbt cal_sgbt)
MODELS_LIST=(${MODELS:-lr clogit gbt cal_sgbt})

# Reduced ablation: only full + additive_1-3 + without_precursor_winners
REDUCED_ABLATION="${REDUCED_ABLATION:-0}"

# Check if a model is in the MODELS_LIST
model_enabled() {
    local target="$1"
    for m in "${MODELS_LIST[@]}"; do
        [[ "$m" == "$target" ]] && return 0
    done
    return 1
}

# Check if a feature config should run under reduced ablation
config_passes_filter() {
    local config_name="$1"
    if [[ "$REDUCED_ABLATION" != "1" ]]; then
        return 0  # no filter, run all
    fi
    case "$config_name" in
        *_full) return 0 ;;
        *_additive_1_*) return 0 ;;
        *_additive_2_*) return 0 ;;
        *_additive_3_*) return 0 ;;
        *_without_precursor_winners) return 0 ;;
        *) return 1 ;;  # skip
    esac
}

# Map category slug to dataset filename
dataset_file_for() {
    local cat="$1"
    case "$cat" in
        best_picture)        echo "best_picture/oscar_best_picture_raw.json" ;;
        directing)           echo "directing/oscar_directing_raw.json" ;;
        actor_leading)       echo "actor_leading/oscar_actor_leading_raw.json" ;;
        actress_leading)     echo "actress_leading/oscar_actress_leading_raw.json" ;;
        actor_supporting)    echo "actor_supporting/oscar_actor_supporting_raw.json" ;;
        actress_supporting)  echo "actress_supporting/oscar_actress_supporting_raw.json" ;;
        original_screenplay) echo "original_screenplay/oscar_original_screenplay_raw.json" ;;
        cinematography)      echo "cinematography/oscar_cinematography_raw.json" ;;
        animated_feature)    echo "animated_feature/oscar_animated_feature_raw.json" ;;
        *) echo "ERROR: unknown category $cat" >&2; return 1 ;;
    esac
}

# ============================================================================
# Generate configs if needed
# ============================================================================

if [[ ! -d "$CONFIGS/features" ]] || [[ $(find "$CONFIGS/features" -name '*.json' 2>/dev/null | wc -l | tr -d ' ') -lt 50 ]]; then
    echo "Generating configs..."
    bash "$SCRIPTS_DIR/run_generate_configs.sh"
    echo ""
fi

# ============================================================================
# Validation
# ============================================================================

echo "============================================================"
echo "Validating prerequisites"
echo "============================================================"

CV_SPLIT="$CONFIGS/cv_splits/leave_one_year_out.json"
LR_GRID="$CONFIGS/param_grids/lr_grid.json"
GBT_GRID="$CONFIGS/param_grids/gbt_grid.json"
CLOGIT_GRID="$CONFIGS/param_grids/conditional_logit_grid.json"
CAL_SGBT_GRID="$CONFIGS/param_grids/calibrated_softmax_gbt_grid.json"

for f in "$CV_SPLIT" "$LR_GRID" "$GBT_GRID" "$CLOGIT_GRID" "$CAL_SGBT_GRID"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

for category in "${CATEGORIES[@]}"; do
    raw_path="$DATASET_DIR/$(dataset_file_for "$category")"
    if [[ ! -f "$raw_path" ]]; then
        echo "ERROR: Missing dataset: $raw_path"
        exit 1
    fi
    config_count=$(find "$CONFIGS/features/$category" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$config_count" -lt 25 ]]; then
        echo "ERROR: Only $config_count feature configs for $category (expected >= 25)"
        exit 1
    fi
done

echo "All prerequisites validated."
echo ""

# ============================================================================
# Run helper
# ============================================================================

RUN_COUNT=0
SKIP_COUNT=0

run_ablation() {
    local category="$1"
    local mode="$2"
    local name="$3"
    local param_grid="$4"
    local feature_config="$5"
    local raw_path="$6"

    local run_dir="$EXP_DIR/$category/$mode/$name"
    if [[ -d "$run_dir" ]]; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi

    local fs_args=""
    if [[ "$mode" == "with_fs" ]]; then
        fs_args="--feature-selection --importance-threshold 0.80"
    fi

    echo "  [$category/$mode] $name"
    # shellcheck disable=SC2086
    uv run python -m "$MODULE" \
        --name "$name" \
        --param-grid "$param_grid" \
        --feature-config "$feature_config" \
        --cv-split "$CV_SPLIT" \
        --train-years "$TRAIN_YEARS" \
        --test-years "$TEST_YEARS" \
        --output-dir "$EXP_DIR/$category/$mode" \
        --raw-path "$raw_path" \
        --as-of-date "$AS_OF_DATE" \
        --n-jobs "$N_JOBS" \
        $fs_args

    RUN_COUNT=$((RUN_COUNT + 1))
}

# ============================================================================
# Run experiments
# ============================================================================

echo "============================================================"
echo "Starting multi-category feature ablation"
echo "  Categories: ${#CATEGORIES[@]}"
echo "  Models: ${MODELS_LIST[*]}"
echo "  Reduced ablation: $REDUCED_ABLATION"
echo "  Modes: ${MODES[*]}"
echo "  Train years: $TRAIN_YEARS -> Test: $TEST_YEARS"
echo "  As-of date: $AS_OF_DATE"
echo "============================================================"
echo ""

START_TIME=$(date +%s)

for category in "${CATEGORIES[@]}"; do
    raw_path="$DATASET_DIR/$(dataset_file_for "$category")"
    features_dir="$CONFIGS/features/$category"

    echo ""
    echo "============================================================"
    echo "Category: $category"
    echo "============================================================"

    for mode in "${MODES[@]}"; do
        echo ""
        echo "--- $category / $mode ---"

        # --- LR family: lr feature configs run with lr_grid + clogit_grid ---
        for config_file in "$features_dir"/lr_*.json; do
            [[ -f "$config_file" ]] || continue
            config_name=$(basename "$config_file" .json)
            config_passes_filter "$config_name" || continue

            # Run with LR param grid
            if model_enabled lr; then
                run_ablation "$category" "$mode" "$config_name" "$LR_GRID" "$config_file" "$raw_path"
            fi

            # Run with Clogit param grid (same features, different model)
            if model_enabled clogit; then
                clogit_name="clogit_${config_name#lr_}"
                run_ablation "$category" "$mode" "$clogit_name" "$CLOGIT_GRID" "$config_file" "$raw_path"
            fi
        done

        # --- GBT family: gbt feature configs run with gbt_grid + cal_sgbt_grid ---
        for config_file in "$features_dir"/gbt_*.json; do
            [[ -f "$config_file" ]] || continue
            config_name=$(basename "$config_file" .json)
            config_passes_filter "$config_name" || continue

            # Run with GBT param grid
            if model_enabled gbt; then
                run_ablation "$category" "$mode" "$config_name" "$GBT_GRID" "$config_file" "$raw_path"
            fi

            # Run with Cal-SGBT param grid (same features, different model)
            if model_enabled cal_sgbt; then
                cal_sgbt_name="cal_sgbt_${config_name#gbt_}"
                run_ablation "$category" "$mode" "$cal_sgbt_name" "$CAL_SGBT_GRID" "$config_file" "$raw_path"
            fi
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo "============================================================"
echo "Multi-category feature ablation complete"
echo "============================================================"
echo "  Runs completed: $RUN_COUNT"
echo "  Runs skipped:   $SKIP_COUNT"
echo "  Elapsed time:   ${ELAPSED_MIN}m ${ELAPSED}s"
echo "  Results:        $EXP_DIR/"
echo "============================================================"
