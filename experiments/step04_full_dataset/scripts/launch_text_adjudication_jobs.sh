#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2/experiments/step04_full_dataset
ARTIFACTS="$ROOT/artifacts"
SCRIPT="$ROOT/scripts/run_text_disagreement_adjudication.py"

for split in train validation test; do
  python3 "$SCRIPT" \
    --manifest "$ARTIFACTS/official_manifest_${split}.jsonl" \
    --gemma "$ARTIFACTS/text_polarity_gemma3_${split}.jsonl" \
    --qwen "$ARTIFACTS/text_polarity_qwen3_${split}.jsonl" \
    --output "$ARTIFACTS/text_adjudication_qwen3vl_${split}.jsonl"
done
