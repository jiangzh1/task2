#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2/experiments/step02_tts_pilot/reference_data
mkdir -p "$ROOT/RAVDESS" "$ROOT/CREMA-D"

download_ravdess() {
  wget -c --tries=50 --timeout=120 --retry-connrefused --waitretry=5 \
    -O "$ROOT/RAVDESS/Audio_Speech_Actors_01-24.zip" \
    'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
  md5sum "$ROOT/RAVDESS/Audio_Speech_Actors_01-24.zip" > "$ROOT/RAVDESS/md5.txt"
}

download_crema_shard() {
  local index="$1"
  wget -c --tries=20 --timeout=120 --retry-connrefused --waitretry=3 \
    -O "$ROOT/CREMA-D/train-${index}.parquet" \
    "https://huggingface.co/api/datasets/cfahlgren1/crema-d/parquet/default/train/${index}.parquet"
}

download_ravdess &
RAVDESS_PID=$!
for index in 0 1 2 3; do
  download_crema_shard "$index" &
done
wait

sha256sum "$ROOT/CREMA-D"/*.parquet > "$ROOT/CREMA-D/sha256.txt"
