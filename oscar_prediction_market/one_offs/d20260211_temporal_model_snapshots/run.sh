#!/usr/bin/env bash
# Temporal model snapshots — train LR and GBT at each awards season date.
#
# Requires:
#   - Intermediate dataset files in DATASET_DIR (oscar_nominations.json,
#     film_metadata.json, precursor_awards.json)
#   - If not present, run the build_dataset one-off first:
#       bash oscar_prediction_market/one_offs/d20260201_build_dataset/build_dataset.sh
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260211_temporal_model_snapshots/run.sh \
#       2>&1 | tee storage/d20260211_temporal_model_snapshots/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

DATASET_DIR="storage/d20260201_build_dataset"
EXP_DIR="storage/d20260211_temporal_model_snapshots"

MODULE_PREFIX="oscar_prediction_market"
ONE_OFF_PREFIX="${MODULE_PREFIX}.one_offs.d20260211_temporal_model_snapshots"

# Config paths (existing repo configs for param grids and CV splits)
CONFIGS="${MODULE_PREFIX//.//}/modeling/configs"
LR_GRID="${CONFIGS}/param_grids/lr_grid.json"
GBT_GRID="${CONFIGS}/param_grids/gbt_grid.json"
CV_SPLIT="${CONFIGS}/cv_splits/leave_one_year_out.json"

# Feature configs will be generated into experiment directory
FEATURE_CONFIG_DIR="${EXP_DIR}/configs/features"

# Training configuration
TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
N_JOBS=4

# Snapshot dates and their event labels
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
echo "Temporal Model Snapshots Experiment"
echo "============================================================"
echo "  Dataset dir: ${DATASET_DIR}"
echo "  Experiment dir: ${EXP_DIR}"
echo "  Snapshots: ${#SNAPSHOT_DATES[@]}"
echo "  Models: ${MODELS[*]}"
echo ""

# Check that intermediate dataset files exist
for f in oscar_nominations.json film_metadata.json precursor_awards.json; do
    if [[ ! -f "${DATASET_DIR}/${f}" ]]; then
        echo "ERROR: Missing ${DATASET_DIR}/${f}"
        echo "Run the build_dataset one-off first."
        exit 1
    fi
done
echo "  Intermediate dataset files: OK"

# Check that config files exist
for f in "${LR_GRID}" "${GBT_GRID}" "${CV_SPLIT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Missing config file: ${f}"
        exit 1
    fi
done
echo "  Config files: OK"
echo ""

# ============================================================================
# Step 0: Create experiment directory structure
# ============================================================================

mkdir -p "${EXP_DIR}/datasets" "${EXP_DIR}/models" "${EXP_DIR}/configs/features"

# ============================================================================
# Step 1: Generate full-feature configs
# ============================================================================

echo "============================================================"
echo "Step 1: Generate full-feature configs"
echo "============================================================"

if [[ -f "${FEATURE_CONFIG_DIR}/lr_full.json" && -f "${FEATURE_CONFIG_DIR}/gbt_full.json" ]]; then
    echo "  SKIP: Feature configs already exist"
else
    uv run python -m "${ONE_OFF_PREFIX}.create_full_feature_configs" \
        --output-dir "${FEATURE_CONFIG_DIR}"
fi
echo ""

# ============================================================================
# Step 2: Generate per-date datasets (merge step only)
# ============================================================================

echo "============================================================"
echo "Step 2: Generate per-date datasets"
echo "============================================================"

for snap_date in "${SNAPSHOT_DATES[@]}"; do
    dataset_out="${EXP_DIR}/datasets/${snap_date}"
    raw_json="${dataset_out}/oscar_best_picture_raw.json"

    if [[ -f "${raw_json}" ]]; then
        echo "  SKIP ${snap_date}: dataset already exists"
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
# Step 3: Build models for each date × model type
# ============================================================================

echo "============================================================"
echo "Step 3: Build models (${#MODELS[@]} models × ${#SNAPSHOT_DATES[@]} dates)"
echo "============================================================"

for model in "${MODELS[@]}"; do
    # Select the right param grid and feature config
    if [[ "${model}" == "lr" ]]; then
        PARAM_GRID="${LR_GRID}"
        FEATURE_CONFIG="${FEATURE_CONFIG_DIR}/lr_full.json"
    else
        PARAM_GRID="${GBT_GRID}"
        FEATURE_CONFIG="${FEATURE_CONFIG_DIR}/gbt_full.json"
    fi

    for snap_date in "${SNAPSHOT_DATES[@]}"; do
        run_name="${model}_${snap_date}"
        output_dir="${EXP_DIR}/models/${model}/${snap_date}"
        raw_path="${EXP_DIR}/datasets/${snap_date}/oscar_best_picture_raw.json"

        # Check if final predictions already exist
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

# ============================================================================
# Step 4: Collect results
# ============================================================================

echo "============================================================"
echo "Step 4: Collect results"
echo "============================================================"

uv run python -m "${ONE_OFF_PREFIX}.collect_results" \
    --models-dir "${EXP_DIR}/models" \
    --output "${EXP_DIR}/model_predictions_timeseries.csv"

echo ""

# ============================================================================
# Step 5: Run analysis
# ============================================================================

echo "============================================================"
echo "Step 5: Analysis"
echo "============================================================"

uv run python -m "${ONE_OFF_PREFIX}.analysis" \
    --predictions "${EXP_DIR}/model_predictions_timeseries.csv" \
    --models-dir "${EXP_DIR}/models" \
    --output-dir "${EXP_DIR}"

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "Results in: ${EXP_DIR}"
echo "============================================================"
