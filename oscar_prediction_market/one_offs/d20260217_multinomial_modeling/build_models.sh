#!/usr/bin/env bash
# Build temporal model snapshots for multinomial modeling comparison.
#
# Builds 5 model types × 11 snapshot dates:
#   - lr (logistic regression) — binary, baseline
#   - gbt (gradient boosting) — binary, baseline
#   - conditional_logit — multinomial, probabilities sum to 1 per ceremony
#   - softmax_gbt — multinomial, multi-class softmax objective
#   - calibrated_softmax_gbt — binary GBT → temperature-scaled softmax
#
# Reuses dataset infrastructure from d20260211 temporal model snapshots.
# Each model uses feature selection with importance > 0 filtering.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/\
#       d20260217_multinomial_modeling/build_models.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

DATASET_DIR="storage/d20260201_build_dataset"
EXP_DIR="storage/d20260217_multinomial_modeling"

MODULE_PREFIX="oscar_prediction_market"

# Config paths
CONFIGS="${MODULE_PREFIX//.//}/modeling/configs"
LR_GRID="${CONFIGS}/param_grids/lr_grid.json"
GBT_GRID="${CONFIGS}/param_grids/gbt_grid.json"
CLOGIT_GRID="${CONFIGS}/param_grids/conditional_logit_grid.json"
SGBT_GRID="${CONFIGS}/param_grids/softmax_gbt_grid.json"
CSGBT_GRID="${CONFIGS}/param_grids/calibrated_softmax_gbt_grid.json"
LR_FEATURES="${CONFIGS}/features/lr_standard.json"
GBT_FEATURES="${CONFIGS}/features/gbt_standard.json"
CLOGIT_FEATURES="${CONFIGS}/features/clogit_standard.json"
SGBT_FEATURES="${CONFIGS}/features/softmax_gbt_standard.json"
CSGBT_FEATURES="${CONFIGS}/features/calibrated_softmax_gbt_standard.json"
CV_SPLIT="${CONFIGS}/cv_splits/leave_one_year_out.json"

TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
N_JOBS=4

# Snapshot dates (same as d20260211)
SNAPSHOT_DATES=(
    "2025-11-30"  # Pre-season baseline
    "2025-12-05"  # Critics Choice noms
    "2025-12-08"  # Golden Globe noms
    "2026-01-04"  # Critics Choice winner
    "2026-01-07"  # SAG noms
    "2026-01-08"  # DGA noms
    "2026-01-09"  # PGA noms
    "2026-01-11"  # Golden Globe winner
    "2026-01-22"  # Oscar noms
    "2026-01-27"  # BAFTA noms
    "2026-02-07"  # DGA winner
)

# All model types to build
MODELS=("lr" "gbt" "conditional_logit" "softmax_gbt" "calibrated_softmax_gbt")

# ============================================================================
# Validation
# ============================================================================

echo "============================================================"
echo "Build Temporal Model Snapshots (Multinomial Comparison)"
echo "============================================================"
echo "  Dataset dir: ${DATASET_DIR}"
echo "  Output dir:  ${EXP_DIR}/models"
echo "  Models: ${MODELS[*]}"
echo "  Snapshots: ${#SNAPSHOT_DATES[@]}"
echo ""

for f in oscar_nominations.json film_metadata.json precursor_awards.json; do
    if [[ ! -f "${DATASET_DIR}/${f}" ]]; then
        echo "ERROR: Missing ${DATASET_DIR}/${f}"
        echo "Run the build_dataset one-off first."
        exit 1
    fi
done
echo "  Intermediate dataset files: OK"

for f in "${LR_GRID}" "${GBT_GRID}" "${CLOGIT_GRID}" "${SGBT_GRID}" "${CSGBT_GRID}" \
         "${LR_FEATURES}" "${GBT_FEATURES}" "${CLOGIT_FEATURES}" "${SGBT_FEATURES}" "${CSGBT_FEATURES}" \
         "${CV_SPLIT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Missing config file: ${f}"
        exit 1
    fi
done
echo "  Config files: OK"
echo ""

# ============================================================================
# Step 1: Generate per-date datasets (reuse from d20260211 if available)
# ============================================================================

echo "============================================================"
echo "Step 1: Generate per-date datasets"
echo "============================================================"

DATASETS_DIR="${EXP_DIR}/datasets"
D211_DATASETS="storage/d20260211_temporal_model_snapshots/datasets"

mkdir -p "${DATASETS_DIR}"

for snap_date in "${SNAPSHOT_DATES[@]}"; do
    dataset_out="${DATASETS_DIR}/${snap_date}"
    raw_json="${dataset_out}/oscar_best_picture_raw.json"

    if [[ -f "${raw_json}" ]]; then
        echo "  SKIP ${snap_date}: dataset already exists"
        continue
    fi

    # Try to reuse from d20260211
    if [[ -f "${D211_DATASETS}/${snap_date}/oscar_best_picture_raw.json" ]]; then
        echo "  COPY ${snap_date}: reusing from d20260211"
        mkdir -p "${dataset_out}"
        cp "${D211_DATASETS}/${snap_date}/oscar_best_picture_raw.json" "${raw_json}"
        continue
    fi

    echo "  Generating dataset for ${snap_date}..."
    mkdir -p "${dataset_out}"
    uv run python -m "${MODULE_PREFIX}.data.build_dataset" \
        --mode merge \
        --year-start 2000 --year-end 2026 \
        --as-of-date "${snap_date}" \
        --input-dir "${DATASET_DIR}" \
        --output-dir "${dataset_out}"
done
echo ""

# ============================================================================
# Step 2: Build models for each date × model type
# ============================================================================

echo "============================================================"
echo "Step 2: Build models (${#MODELS[@]} models × ${#SNAPSHOT_DATES[@]} dates)"
echo "============================================================"

for model in "${MODELS[@]}"; do
    # Select config based on model type
    case "${model}" in
        lr)
            PARAM_GRID="${LR_GRID}"
            FEATURE_CONFIG="${LR_FEATURES}"
            ;;
        gbt)
            PARAM_GRID="${GBT_GRID}"
            FEATURE_CONFIG="${GBT_FEATURES}"
            ;;
        conditional_logit)
            PARAM_GRID="${CLOGIT_GRID}"
            FEATURE_CONFIG="${CLOGIT_FEATURES}"
            ;;
        softmax_gbt)
            PARAM_GRID="${SGBT_GRID}"
            FEATURE_CONFIG="${SGBT_FEATURES}"
            ;;
        calibrated_softmax_gbt)
            PARAM_GRID="${CSGBT_GRID}"
            FEATURE_CONFIG="${CSGBT_FEATURES}"
            ;;
        *)
            echo "ERROR: Unknown model type: ${model}"
            exit 1
            ;;
    esac

    for snap_date in "${SNAPSHOT_DATES[@]}"; do
        run_name="${model}_${snap_date}"
        output_dir="${EXP_DIR}/models/${model}/${snap_date}"
        raw_path="${DATASETS_DIR}/${snap_date}/oscar_best_picture_raw.json"

        final_preds="${output_dir}/${run_name}/5_final_predict/predictions_test.csv"
        if [[ -f "${final_preds}" ]]; then
            echo "  SKIP ${run_name}: already complete"
            continue
        fi

        echo ""
        echo "  ======== ${run_name} ========"
        uv run python -m "${MODULE_PREFIX}.modeling.build_model" \
            --name "${run_name}" \
            --param-grid "${PARAM_GRID}" \
            --feature-config "${FEATURE_CONFIG}" \
            --cv-split "${CV_SPLIT}" \
            --train-years "${TRAIN_YEARS}" \
            --test-years "${TEST_YEARS}" \
            --output-dir "${output_dir}" \
            --raw-path "${raw_path}" \
            --as-of-date "${snap_date}" \
            --n-jobs "${N_JOBS}" \
            --feature-selection
    done
done
echo ""

echo "============================================================"
echo "Model building complete"
echo "Results in: ${EXP_DIR}/models"
echo "============================================================"
