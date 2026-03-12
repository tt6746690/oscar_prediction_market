#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# d20260305 Config Selection Sweep — Full Pipeline
# ============================================================================
# Runs targeted backtests (27 configs × 6 models × 2 years), scoring, and
# extended analysis. Reuses d20260225 Python modules with --config-grid and
# --exp-dir overrides to write output to a separate storage directory.
#
# The targeted grid fixes fee_type=taker, kelly_mode=multi_outcome,
# allowed_directions=all (justified by d20260225 findings) and sweeps:
#   - edge_threshold: 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25
#   - kelly_fraction: 0.05, 0.15, 0.25
#
# Total: 27 configs × 6 models × ~17 categories × ~10 entries ≈ rapid
# ============================================================================

MODULE="oscar_prediction_market.one_offs.d20260225_buy_hold_backtest"
SHARED_DIR="storage/d20260220_backtest_strategies"
EXP_DIR="storage/d20260305_config_selection_sweep"

echo "============================================================"
echo "Config Selection Sweep Pipeline (d20260305)"
echo "============================================================"
echo "Date:   $(date)"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""
echo "Targeted sweep: 27 configs × 6 models × 2 years"
echo "  Fixed: taker fees, multi_outcome Kelly, all directions"
echo "  Swept: edge_threshold (9 values), kelly_fraction (3 values)"
echo "Results: ${EXP_DIR}/"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------
for year in 2024 2025; do
    [[ -d "${SHARED_DIR}/${year}/models" ]] || {
        echo "ERROR: Shared datasets/models not found at ${SHARED_DIR}/${year}/models."
        echo "Run the d20260220_backtest_strategies pipeline first."
        exit 1
    }
done
[[ -d "${SHARED_DIR}/configs" ]] || {
    echo "ERROR: Shared configs not found at ${SHARED_DIR}/configs."
    exit 1
}

mkdir -p "${EXP_DIR}"

# --------------------------------------------------------------------------
# Step 1: Run Targeted Backtests
# --------------------------------------------------------------------------
echo "============================================================"
echo "Step 1: Run Targeted Backtests"
echo "============================================================"

for year in 2024 2025; do
    echo ""
    echo "--- ${year} ceremony year ---"
    uv run python -m "${MODULE}.run_backtests" \
        --ceremony-year "${year}" \
        --config-grid targeted \
        --results-dir "${EXP_DIR}/${year}/results"
done

echo ""
echo "Backtest outputs:"
for year in 2024 2025; do
    echo "  ${EXP_DIR}/${year}/results/"
    ls -1 "${EXP_DIR}/${year}/results/"*.csv 2>/dev/null | sed 's/^/    /' || echo "    (no CSV files found)"
done

# --------------------------------------------------------------------------
# Step 2: Run EV + CVaR Scoring
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 2: Run EV + CVaR Scoring"
echo "============================================================"

for year in 2024 2025; do
    echo ""
    echo "--- ${year} ceremony year ---"
    uv run python -m "${MODULE}.scenario_scoring" \
        --ceremony-year "${year}" \
        --exp-dir "${EXP_DIR}"
done

echo ""
echo "--- Cross-year ---"
uv run python -m "${MODULE}.scenario_scoring" \
    --cross-year \
    --exp-dir "${EXP_DIR}"

# --------------------------------------------------------------------------
# Step 3: Extended Analysis Tables
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 3: Extended Analysis (Tables)"
echo "============================================================"

echo ""
echo "--- Cross-year tables ---"
uv run python -m "${MODULE}.generate_tables" \
    --mode cross-year \
    --extended \
    --exp-dir "${EXP_DIR}"

# --------------------------------------------------------------------------
# Step 4: Plots + Model Comparison + Reliability Analysis
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 4: Plots + Model Comparison + Reliability Analysis"
echo "============================================================"

SWEEP_MODULE="oscar_prediction_market.one_offs.d20260305_config_selection_sweep"

echo ""
echo "--- Main plots ---"
uv run python -m "${SWEEP_MODULE}.plot_results"

echo ""
echo "--- Model comparison analysis ---"
uv run python -m "${SWEEP_MODULE}.compare_models"

echo ""
echo "--- Reliability analysis ---"
uv run python -m "${SWEEP_MODULE}.reliability_analysis"

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Results: ${EXP_DIR}/"
echo "Finished at $(date)"
