#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2/experiments/step04_full_dataset
SCRIPT="$ROOT/scripts/run_text_polarity.py"
ARTIFACTS="$ROOT/artifacts"

run_model() {
  local model="$1"
  local url="$2"
  local name="$3"
  for split in train validation test; do
    python3 "$SCRIPT" \
      --input "$ARTIFACTS/official_manifest_${split}.jsonl" \
      --output "$ARTIFACTS/text_polarity_${name}_${split}.jsonl" \
      --model "$model" \
      --ollama-url "$url"
  done
}

run_model gemma3:12b http://127.0.0.1:11434 gemma3 > "$ARTIFACTS/text_polarity_gemma3.log" 2>&1 &
GEMMA_PID=$!
run_model qwen3:8b http://127.0.0.1:11435 qwen3 > "$ARTIFACTS/text_polarity_qwen3.log" 2>&1 &
QWEN_PID=$!
echo "gemma_pid=$GEMMA_PID qwen_pid=$QWEN_PID"
wait "$GEMMA_PID" "$QWEN_PID"
