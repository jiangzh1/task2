#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step03_label_audit"
RUNTIME="$STEP/runtime"

while pgrep -f '[o]llama pull gemma3:12b' >/dev/null; do
  sleep 30
done

env OLLAMA_HOST=127.0.0.1:11434 "$RUNTIME/ollama_dist/bin/ollama" show gemma3:12b > "$STEP/artifacts/gemma3_12b_model.txt"
python3 "$STEP/scripts/run_ollama_pilot.py" \
  --input "$STEP/artifacts/pilot_210.jsonl" \
  --output "$STEP/artifacts/smoke_14_results.jsonl" \
  --data-root "$ROOT/data" \
  --model gemma3:12b \
  --per-class-limit 2 \
  > "$STEP/artifacts/smoke_14_run.log" 2>&1
