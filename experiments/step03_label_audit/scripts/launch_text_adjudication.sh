#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step03_label_audit"
RUNTIME="$STEP/runtime"

while pgrep -f '[o]llama pull qwen3:8b' >/dev/null; do
  sleep 30
done

python3 "$STEP/scripts/run_text_adjudicator.py" \
  --pilot "$STEP/artifacts/pilot_210.jsonl" \
  --gemma "$STEP/artifacts/pilot_208_gemma3_results.jsonl" \
  --qwen-vl "$STEP/artifacts/pilot_208_qwen3vl_results.jsonl" \
  --output "$STEP/artifacts/text_disagreement_qwen3_results.jsonl" \
  > "$STEP/artifacts/text_disagreement_qwen3_run.log" 2>&1
