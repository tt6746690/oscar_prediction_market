#!/usr/bin/env bash
# Build temporal model snapshots with best-performing configs.
#
# Uses lr_standard (25 features, wide grid, thresh 0.80) and gbt_standard
# (17 features, gbt grid) identified from the Feb 14 feature ablation.
#
# Reuses the build_model.py and build_dataset infrastructure from the
# temporal snapshots one-off (d20260211), but stores output under
# storage/d20260214_trade_signal_ablation/models/.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/\
#       d20260214_trade_signal_ablation/build_models.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

DATASET_DIR="storage/d20260201_build_dataset"
EXP_DIR="storage/d20260214_trade_signal_ablation"

MODULE_PREFIX="oscar_prediction_market"

# Config paths — use the best configs from feature ablation
CONFIGS="${MODULE_PREFIX//.//}/modeling/configs"
LR_GRID="${CONFIGS}/param_grids/lr_grid.json"
GBT_GRID="${CONFIGS}/param_grids/gbt_grid.json"
LR_FEATURES="${CONFIGS}/features/lr_standard.json"
GBT_FEATURES="${CONFIGS}/features/gbt_standard.json"
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

MODELS=("lr" "gbt")

# ============================================================================
# Validation
# ============================================================================

echo "============================================================"
echo "Build Temporal Model Snapshots (Best Configs)"
echo "============================================================"
echo "  Dataset dir: ${DATASET_DIR}"
echo "  Output dir:  ${EXP_DIR}/models"
echo "  LR features: ${LR_FEATURES}"
echo "  GBT features: ${GBT_FEATURES}"
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

for f in "${LR_GRID}" "${GBT_GRID}" "${LR_FEATURES}" "${GBT_FEATURES}" "${CV_SPLIT}"; do
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

# Reuse datasets from d20260211 if they exist, otherwise generate
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
        echo "  LINK ${snap_date}: reusing from d20260211"
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
    if [[ "${model}" == "lr" ]]; then
        PARAM_GRID="${LR_GRID}"
        FEATURE_CONFIG="${LR_FEATURES}"
    else
        PARAM_GRID="${GBT_GRID}"
        FEATURE_CONFIG="${GBT_FEATURES}"
    fi

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
