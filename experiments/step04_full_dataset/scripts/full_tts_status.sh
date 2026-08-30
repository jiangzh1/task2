#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jzh/2026/task2/experiments/step04_full_dataset
ART="$ROOT/artifacts/full_tts_pp"
OUT="$ROOT/audio/full_tts_pp"
echo "=== 后台进程 ==="
ps -eo pid,etime,cmd | grep -E 'full_tts_supervisor|inference_tts.py' | grep -v grep || true
echo "=== 已即时落盘的原始分段 ==="
find "$OUT/decode/pred_audio/neutral_prompt_speech" -maxdepth 1 -type f -name '*.wav' 2>/dev/null | wc -l
echo "=== 最近一次批次结算状态 ==="
cat "$ART/status.json" 2>/dev/null || echo "首批尚未结算；原始 WAV 仍在逐条落盘。"
echo "=== 最近日志 ==="
grep -aE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}' "$ART/supervisor.log" 2>/dev/null | tail -20 || true
