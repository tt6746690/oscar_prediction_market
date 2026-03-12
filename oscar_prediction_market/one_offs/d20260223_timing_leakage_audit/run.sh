#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260223_timing_leakage_audit"
mkdir -p "${EXP_DIR}/2025"

echo "=== Run inferred event-time audit ==="
uv run python -m oscar_prediction_market.one_offs.d20260223_timing_leakage_audit.audit_uniform_lag_hours \
  --output-subdir lag_audit_inferred \
  2>&1 | tee "${EXP_DIR}/2025/run_inferred.log"

echo "=== Run fixed 21:00 ET audit ==="
uv run python -m oscar_prediction_market.one_offs.d20260223_timing_leakage_audit.audit_uniform_lag_hours \
  --no-infer-event-time \
  --default-event-time-et 21:00 \
  --output-subdir lag_audit_fixed_2100 \
  2>&1 | tee "${EXP_DIR}/2025/run_fixed_2100.log"

echo "Done. Outputs in ${EXP_DIR}/2025/"
