#!/usr/bin/env bash
# Run the live-pricing pipeline: fetch current orderbook prices, size positions,
# generate per-config reports, and run supplementary analyses.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260224_live_2026/run_live.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS_DIR="storage/d20260224_live_2026/results"
REPORTS_DIR="oscar_prediction_market/one_offs/d20260224_live_2026/reports"

echo "=== Step 1: Run buy-hold with LIVE orderbook prices ==="
uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.run_buy_hold \
    --live

echo ""
echo "=== Step 2: Generate per-config reports (live snapshot) ==="
uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.generate_report \
    --live --live-portfolio

echo ""
echo "=== Step 3: Model agreement analysis ==="
uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.analyze_model_agreement

echo ""
echo "=== Step 4: Orderbook analysis ==="
uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.analyze_orderbook

echo ""
echo "=== Done! ==="
echo "Reports (markdown):  ${REPORTS_DIR}/"
ls -lt "${REPORTS_DIR}"/*.md 2>/dev/null | head -10
echo ""
echo "Results (CSV + analysis): ${RESULTS_DIR}/"
ls -lt "${RESULTS_DIR}"/*.{csv,md} 2>/dev/null | head -10
