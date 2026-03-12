#!/usr/bin/env bash
# Master script for 2026 live Oscar predictions pipeline.
#
# Runs the full pipeline:
#   Step 0: Refresh precursor awards + film metadata
#   Step 1: Copy configs from 2025 backtest
#   Step 2: Build as-of-date datasets (4 snapshots × 9 categories = 36 merges)
#   Step 3: Train models (4 snapshots × 9 categories × 4 models = 144 trainings)
#   Step 5: Buy-and-hold analysis (recommended config: avg_ensemble + maxedge_100 allocation)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260224_live_2026/run.sh \
#       2>&1 | tee storage/d20260224_live_2026/run.log
#
# Output: storage/d20260224_live_2026/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="oscar_prediction_market/one_offs/d20260224_live_2026"
EXP_DIR="storage/d20260224_live_2026"
MODULE_PREFIX="oscar_prediction_market.one_offs.d20260224_live_2026"

echo "============================================================"
echo "2026 Live Oscar Predictions — Full Pipeline"
echo "Date: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Output: $EXP_DIR"
echo "============================================================"

mkdir -p "$EXP_DIR"

# ------------------------------------------------------------------
# Step 0: Refresh precursor + metadata caches
# ------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 0: Refresh precursor awards + film metadata"
echo "============================================================"

if [[ -f "$EXP_DIR/shared/precursor_awards.json" ]] && [[ -f "$EXP_DIR/shared/film_metadata.json" ]]; then
    echo "  SKIP: shared data already exists at $EXP_DIR/shared/"
    echo "  Delete to force refresh."
else
    uv run python -m "$MODULE_PREFIX.refresh_data"
fi

# ------------------------------------------------------------------
# Step 1: Setup configs
# ------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 1: Copy configs from 2025 backtest"
echo "============================================================"

bash "$SCRIPT_DIR/setup_configs.sh"

# ------------------------------------------------------------------
# Step 2: Build as-of-date datasets
# ------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 2: Build 2026 as-of-date datasets"
echo "============================================================"

bash "$SCRIPT_DIR/build_datasets.sh"

# ------------------------------------------------------------------
# Step 3: Train models
# ------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 3: Train 2026 models (this may take ~2-3 hours)"
echo "============================================================"

bash "$SCRIPT_DIR/train_models.sh"

# ------------------------------------------------------------------
# Step 5: Buy-and-hold analysis
# ------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 5: Buy-and-hold analysis (recommended config: avg_ensemble + maxedge_100 allocation)"
echo "============================================================"

uv run python -m "$MODULE_PREFIX.run_buy_hold" \
    --inferred-lag-hours 6 \
    --results-dir "$EXP_DIR/results"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results: $EXP_DIR/results/"
echo "============================================================"
