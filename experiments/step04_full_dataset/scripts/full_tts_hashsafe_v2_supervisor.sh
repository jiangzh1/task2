#!/usr/bin/env bash
# 短分段 v2：复用旧成功音频，只处理旧队列未成功部分。
set -euo pipefail

ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step04_full_dataset"
PY="$ROOT/experiments/step02_tts_pilot/.venv/bin/python"
ART="$STEP/artifacts/full_tts_pp_hashsafe_v2"
OUT="$STEP/audio/full_tts_pp_hashsafe_v2"
DB="$ART/state.sqlite3"
MANIFEST="$ART/full_tts_segments.jsonl"
INDEX="$ART/full_tts_sample_index.jsonl"
BATCH="$ART/current_batch.jsonl"
RAW="$OUT/decode/pred_audio/neutral_prompt_speech"
SEGMENTS="$OUT/segments_16k"
FINAL="$OUT/final_16k"
STATUS="$ART/status.json"
LOG="$ART/supervisor.log"
LIMIT="${BATCH_SIZE:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-60}"

mkdir -p "$ART" "$OUT" "$SEGMENTS" "$FINAL"
exec 9>"$ART/supervisor.lock"
flock -n 9 || { echo "已有 v2 TTS supervisor 正在运行"; exit 3; }
exec >>"$LOG" 2>&1
echo "[$(date -Is)] v2 supervisor 启动，batch_size=$LIMIT max_attempts=$MAX_ATTEMPTS"
"$PY" "$STEP/scripts/manage_full_tts.py" init --db "$DB" --manifest "$MANIFEST"
while true; do
  "$PY" "$STEP/scripts/manage_full_tts.py" reconcile --db "$DB" --raw-dir "$RAW" --segment-dir "$SEGMENTS"
  "$PY" "$STEP/scripts/manage_full_tts.py" assemble --db "$DB" --index "$INDEX" --segment-dir "$SEGMENTS" --final-dir "$FINAL"
  "$PY" "$STEP/scripts/manage_full_tts.py" summary --db "$DB" --index "$INDEX" --final-dir "$FINAL" --max-attempts "$MAX_ATTEMPTS" --status-json "$STATUS"
  count=$("$PY" "$STEP/scripts/manage_full_tts.py" make-batch --db "$DB" --batch "$BATCH" --limit "$LIMIT" --max-attempts "$MAX_ATTEMPTS" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["batch_size"])')
  if [[ "$count" == "0" ]]; then
    echo "[$(date -Is)] 无可重试分段，supervisor 结束"
    break
  fi
  echo "[$(date -Is)] 开始短分段批次，分段数=$count"
  set +e
  "$STEP/scripts/run_emovoice_pp_batch.sh" "$BATCH" "$OUT" 0
  code=$?
  set -e
  if [[ "$code" != "0" ]]; then
    echo "[$(date -Is)] 模型异常退出 code=$code；保留状态，${RESTART_DELAY_SECONDS} 秒后自动恢复"
    sleep "$RESTART_DELAY_SECONDS"
    continue
  fi
  "$PY" "$STEP/scripts/manage_full_tts.py" reconcile --db "$DB" --raw-dir "$RAW" --segment-dir "$SEGMENTS"
  "$PY" "$STEP/scripts/manage_full_tts.py" mark-missing --db "$DB" --batch "$BATCH" --segment-dir "$SEGMENTS" --message "EmoVoice-PP 正常结束但未生成该短分段"
done
