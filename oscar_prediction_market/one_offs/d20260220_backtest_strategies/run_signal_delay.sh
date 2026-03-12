#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Run backtests with 3 signal-delay modes to quantify timing leakage impact.
#
# Mode 1: delay=0 (legacy) — same-day execution, baseline for comparison.
# Mode 2: delay=1 (next-day) — conservative: model can't trade until next EOD.
# Mode 3: inferred+6h — realistic: use actual event UTC times + 6h processing lag.

EXP_DIR="storage/d20260220_backtest_strategies/2025"
MODULE="oscar_prediction_market.one_offs.d20260220_backtest_strategies.run_backtests"

echo "=== Mode 1: delay=0 (legacy same-day) ==="
uv run python -m "$MODULE" \
    --signal-delay-days 0 \
    --results-dir "${EXP_DIR}/results_delay_0"

echo ""
echo "=== Mode 2: delay=1 (next-day EOD) ==="
uv run python -m "$MODULE" \
    --signal-delay-days 1 \
    --results-dir "${EXP_DIR}/results_delay_1"

echo ""
echo "=== Mode 3: inferred + 6h lag ==="
uv run python -m "$MODULE" \
    --inferred-lag-hours 6 \
    --results-dir "${EXP_DIR}/results_inferred_6h"

echo ""
echo "=== All 3 modes complete ==="
echo "Results saved to:"
echo "  ${EXP_DIR}/results_delay_0/"
echo "  ${EXP_DIR}/results_delay_1/"
echo "  ${EXP_DIR}/results_inferred_6h/"
