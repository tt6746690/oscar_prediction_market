#!/usr/bin/env bash
# Trade signal backtest — simulate trading over temporal model snapshots.
#
# Requires:
#   - Temporal model snapshots in storage/d20260211_temporal_model_snapshots/
#   - Kalshi API access (for price data, cached after first fetch)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260214_trade_signal_backtest/run.sh \
#       2>&1 | tee storage/d20260214_trade_signal_backtest/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

EXP_DIR="storage/d20260214_trade_signal_backtest"
CONFIG="${EXP_DIR}/config.json"

MODULE_PREFIX="oscar_prediction_market.one_offs.d20260214_trade_signal_backtest"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${EXP_DIR}"

# Copy config into experiment directory if not already there
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found at ${CONFIG}"
    echo "Create it first, e.g.:"
    echo '  cat > '"${CONFIG}"' << '\''EOF'\'''
    echo '{}'
    echo 'EOF'
    exit 1
fi

echo "============================================================"
echo "Trade Signal Backtest"
echo "============================================================"
echo "Experiment dir: ${EXP_DIR}"
echo "Config: ${CONFIG}"
echo "Started: $(date)"
echo ""

# ============================================================================
# Step 1: Generate signals (backtest)
# ============================================================================

echo "Step 1: Running backtest..."
uv run python -m "${MODULE_PREFIX}.generate_signals" \
    --config "${CONFIG}" \
    --output-dir "${EXP_DIR}"

# ============================================================================
# Step 2: Analyze results
# ============================================================================

echo ""
echo "Step 2: Analyzing results..."
uv run python -m "${MODULE_PREFIX}.analyze_signals" \
    --results-dir "${EXP_DIR}"

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "Results in: ${EXP_DIR}"
echo "============================================================"
