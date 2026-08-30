#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step03_label_audit"
RUNTIME="$STEP/runtime"

while pgrep -f '[o]llama pull qwen3-vl:8b' >/dev/null; do
  sleep 30
done

env OLLAMA_HOST=127.0.0.1:11434 "$RUNTIME/ollama_dist/bin/ollama" show qwen3-vl:8b > "$STEP/artifacts/qwen3_vl_8b_model.txt"

python3 "$STEP/scripts/run_ollama_pilot.py" \
  --input "$STEP/artifacts/pilot_210.jsonl" \
  --output "$STEP/artifacts/qwen_smoke_14_v2_results.jsonl" \
  --data-root "$ROOT/data" \
  --model qwen3-vl:8b \
  --per-class-limit 2 \
  > "$STEP/artifacts/qwen_smoke_14_v2_run.log" 2>&1

python3 "$STEP/scripts/run_ollama_pilot.py" \
  --input "$STEP/artifacts/pilot_210.jsonl" \
  --output "$STEP/artifacts/qwen_smoke_14_v2_repeat_results.jsonl" \
  --data-root "$ROOT/data" \
  --model qwen3-vl:8b \
  --per-class-limit 2 \
  > "$STEP/artifacts/qwen_smoke_14_v2_repeat_run.log" 2>&1
