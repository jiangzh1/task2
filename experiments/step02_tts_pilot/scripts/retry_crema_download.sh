#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2/experiments/step02_tts_pilot/reference_data/CREMA-D
mkdir -p "$ROOT"
for index in 0 1 2 3; do
  target="$ROOT/train-${index}.parquet"
  until [[ -s "$target" ]]; do
    curl -fL --retry 20 --retry-all-errors --retry-delay 10 --connect-timeout 60 \
      -o "$target" "https://huggingface.co/api/datasets/cfahlgren1/crema-d/parquet/default/train/${index}.parquet" || true
    if [[ ! -s "$target" ]]; then sleep 30; fi
  done
done
sha256sum "$ROOT"/*.parquet > "$ROOT/sha256.txt"
