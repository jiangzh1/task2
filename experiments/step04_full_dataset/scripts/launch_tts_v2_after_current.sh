#!/usr/bin/env bash
# 若当前孤立批次仍在运行，等待它结束后无缝接力长期 supervisor。
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step04_full_dataset"
ART="$STEP/artifacts/full_tts_pp_hashsafe_v2"

mkdir -p "$ART"
while pgrep -f '[r]un_emovoice_pp_batch.sh' >/dev/null; do
  sleep 30
done
exec env BATCH_SIZE=16 MAX_ATTEMPTS=100 RESTART_DELAY_SECONDS=60 \
  "$STEP/scripts/full_tts_hashsafe_v2_supervisor.sh"
