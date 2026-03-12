#!/usr/bin/env bash
# Multinomial modeling experiment — compare binary vs. multinomial prediction models.
#
# Compares 5 model types for Oscar Best Picture prediction:
#   - Binary LR (logistic regression, independent per nominee)
#   - Binary GBT (gradient boosting, independent per nominee)
#   - Conditional Logit (multinomial, probabilities sum to 1 per ceremony)
#   - Softmax GBT (multi-class XGBoost with softmax objective)
#   - Calibrated Softmax GBT (binary GBT → temperature-scaled softmax)
#
# Pipeline:
#   1. Build temporal model snapshots (4 model types × 11 dates)
#   2. Analyze CV results (compare accuracy, calibration, prob-sum)
#   3. Run trading backtest (simulate trading with each model type)
#   4. Analyze backtest results (compare P&L, trades, risk)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/\
#       d20260217_multinomial_modeling/run.sh \
#       2>&1 | tee storage/d20260217_multinomial_modeling/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

EXP_DIR="storage/d20260217_multinomial_modeling"
ONE_OFF_DIR="oscar_prediction_market/one_offs/d20260217_multinomial_modeling"
BACKTEST_MODULE="oscar_prediction_market.one_offs.d20260214_trade_signal_backtest"

mkdir -p "${EXP_DIR}"

echo "============================================================"
echo "Multinomial Modeling Experiment"
echo "============================================================"
echo "  Experiment dir: ${EXP_DIR}"
echo "  Started: $(date)"
echo ""

# ============================================================================
# Step 1: Build temporal model snapshots
# ============================================================================

echo "============================================================"
echo "Step 1: Build temporal model snapshots"
echo "============================================================"
bash "${ONE_OFF_DIR}/build_models.sh"
echo ""

# ============================================================================
# Step 2: Analyze CV results
# ============================================================================

echo "============================================================"
echo "Step 2: Analyze CV results"
echo "============================================================"
uv run python -m \
    oscar_prediction_market.one_offs.d20260217_multinomial_modeling.analyze_cv \
    --exp-dir "${EXP_DIR}" \
    --output-dir "${EXP_DIR}/cv_analysis"
echo ""

# ============================================================================
# Step 3: Run trading backtest
# ============================================================================

echo "============================================================"
echo "Step 3: Run trading backtest"
echo "============================================================"

BACKTEST_DIR="${EXP_DIR}/backtest"
BACKTEST_CONFIG="${BACKTEST_DIR}/config.json"
mkdir -p "${BACKTEST_DIR}"

# Create backtest config using best settings from d20260214_trade_signal_ablation:
# - kelly=0.10, min_edge=0.05, sell_edge=-0.03, maker fees
# - multi_outcome kelly, dynamic bankroll, market_blend_alpha=0.15
if [[ ! -f "${BACKTEST_CONFIG}" ]]; then
    cat > "${BACKTEST_CONFIG}" << 'ENDOFCONFIG'
{
    "bankroll_dollars": 1000,
    "kelly_fraction": 0.10,
    "min_edge": 0.05,
    "sell_edge_threshold": -0.03,
    "max_position_per_outcome_dollars": 250,
    "max_total_exposure_dollars": 500,
    "spread_penalty_mode": "trade_data",
    "fixed_spread_penalty_cents": 2.0,
    "model_types": ["lr", "gbt", "conditional_logit", "softmax_gbt", "calibrated_softmax_gbt", "average"],
    "snapshots_dir": "storage/d20260217_multinomial_modeling",
    "price_start_date": "2025-12-01",
    "price_end_date": "2026-02-14",
    "bankroll_mode": "both",
    "fee_type": "maker",
    "min_price": 0,
    "market_blend_alpha": 0.15,
    "normalize_probabilities": false,
    "kelly_mode": "multi_outcome"
}
ENDOFCONFIG
    echo "  Created backtest config: ${BACKTEST_CONFIG}"
fi

echo "  Running backtest..."
uv run python -m "${BACKTEST_MODULE}.generate_signals" \
    --config "${BACKTEST_CONFIG}" \
    --output-dir "${BACKTEST_DIR}"

# ============================================================================
# Step 4: Analyze backtest results
# ============================================================================

echo ""
echo "============================================================"
echo "Step 4: Analyze backtest results"
echo "============================================================"
uv run python -m "${BACKTEST_MODULE}.analyze_signals" \
    --results-dir "${BACKTEST_DIR}"

# ============================================================================
# Step 5: Deep-dive analysis
# ============================================================================

echo ""
echo "============================================================"
echo "Step 5: Deep-dive analysis"
echo "============================================================"
uv run python -m \
    oscar_prediction_market.one_offs.d20260217_multinomial_modeling.analyze_deep_dive \
    --exp-dir "${EXP_DIR}" \
    --output-dir "${EXP_DIR}/deep_dive"
echo ""

# ============================================================================
# Step 6: Sync assets
# ============================================================================

echo ""
echo "============================================================"
echo "Step 6: Sync assets"
echo "============================================================"
if [[ -f "${ONE_OFF_DIR}/../sync_assets.sh" ]]; then
    bash "${ONE_OFF_DIR}/../sync_assets.sh" || true
fi

echo ""
echo "============================================================"
echo "Experiment complete: $(date)"
echo "Results in: ${EXP_DIR}"
echo "  CV analysis:  ${EXP_DIR}/cv_analysis/"
echo "  Backtest:     ${EXP_DIR}/backtest/"
echo "============================================================"
