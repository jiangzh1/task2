#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step03_label_audit"

while pgrep -f '[r]un_ollama_pilot.py.*pilot_208_qwen3vl' >/dev/null; do
  sleep 30
done

python3 "$STEP/scripts/compare_three_way.py" \
  --gemma "$STEP/artifacts/pilot_208_gemma3_results.jsonl" \
  --qwen "$STEP/artifacts/pilot_208_qwen3vl_results.jsonl" \
  --json-output "$STEP/artifacts/pilot_208_three_way_summary.json" \
  --report-output "$STEP/reports/THREE_WAY_REVIEW.md" \
  > "$STEP/artifacts/three_way_compare.log" 2>&1
