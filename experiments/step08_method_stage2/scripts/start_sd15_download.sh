#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step08_method_stage2"
PY="$ROOT/experiments/step02_tts_pilot/.venv/bin/python"
MODEL_DIR="$STEP/assets/stable-diffusion-v1-5"
mkdir -p "$STEP/assets" "$STEP/artifacts"
exec 9>"$STEP/artifacts/sd15_download.lock"
flock -n 9 || { echo "SD1.5 下载任务已在运行"; exit 3; }
MAX_RETRIES="${MAX_RETRIES:-12}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
attempt=1
while true; do
  echo "[$(date -Is)] SD1.5 核心权重下载，第 $attempt/$MAX_RETRIES 次尝试"
  set +e
  "$PY" "$STEP/scripts/download_sd15_core.py" --output-dir "$MODEL_DIR"
  code=$?
  set -e
  if [[ "$code" == "0" ]]; then
    echo "[$(date -Is)] SD1.5 核心权重完整性检查通过"
    exit 0
  fi
  if (( attempt >= MAX_RETRIES )); then
    echo "[$(date -Is)] 已达到最大重试次数，保留缓存供下次断点续传"
    exit "$code"
  fi
  echo "[$(date -Is)] 下载失败 code=$code，${RETRY_SECONDS}s 后断点续传"
  sleep "$RETRY_SECONDS"
  attempt=$((attempt + 1))
done
