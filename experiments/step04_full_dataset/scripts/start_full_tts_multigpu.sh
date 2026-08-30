#!/usr/bin/env bash
# 启动或恢复两个可独立停止的 TTS worker。停止一个 worker：kill -TERM $(cat <artifacts>/pids/<worker>.pid)
set -euo pipefail
STEP=/data/jzh/2026/task2/experiments/step04_full_dataset
ART="$STEP/artifacts/full_tts_pp_hashsafe_v3_multigpu"
mkdir -p "$ART/pids"
for spec in worker_gpu0:0 worker_gpu1:1; do
  worker=${spec%%:*}; gpu=${spec##*:}; pid_file="$ART/pids/$worker.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then echo "$worker 已运行"; continue; fi
  setsid "$STEP/scripts/full_tts_multigpu_worker.sh" "$worker" "$gpu" >/dev/null 2>&1 &
  echo $! > "$pid_file"
  echo "$worker 已启动，GPU $gpu，PID $(cat "$pid_file")"
done
