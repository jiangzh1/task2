#!/usr/bin/env bash
# 等待一次 SDXL 缓存进程结束，再恢复 GPU0 的断点式 TTS worker。
set -euo pipefail
CACHE_PID_FILE="${1:?cache pid 文件}"
TTS_START="/data/jzh/2026/task2/experiments/step04_full_dataset/scripts/start_full_tts_multigpu.sh"
LOG="/data/jzh/2026/task2/experiments/step04_full_dataset/artifacts/full_tts_pp_hashsafe_v3_multigpu/restore_after_sdxl_cache.log"
cache_pid="$(cat "$CACHE_PID_FILE")"
while kill -0 "$cache_pid" 2>/dev/null; do
  sleep 30
done
"$TTS_START" >>"$LOG" 2>&1
