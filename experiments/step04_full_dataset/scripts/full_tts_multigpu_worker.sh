#!/usr/bin/env bash
# 每个 worker 独占一张 GPU；SQLite 原子租约保证两个 worker 不重复领取分段。
set -euo pipefail

WORKER="${1:?worker 名称}"
GPU_ID="${2:?GPU 编号}"
ROOT=/data/jzh/2026/task2
STEP="$ROOT/experiments/step04_full_dataset"
PY="$ROOT/experiments/step02_tts_pilot/.venv/bin/python"
ART="$STEP/artifacts/full_tts_pp_hashsafe_v3_multigpu"
OUT="$STEP/audio/full_tts_pp_hashsafe_v3_multigpu"
DB="$ART/state.sqlite3"
MANIFEST="$ART/full_tts_segments.jsonl"
INDEX="$ART/full_tts_sample_index.jsonl"
BATCH="$ART/batches/${WORKER}.jsonl"
RAW="$OUT/decode/pred_audio/neutral_prompt_speech"
SEGMENTS="$OUT/segments_16k"
FINAL="$OUT/final_16k"
LOCK="$ART/maintenance.lock"
LIMIT="${BATCH_SIZE:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"

mkdir -p "$ART/batches" "$OUT" "$SEGMENTS" "$FINAL"
# 队列操作不导入 PyTorch 的 CUDA 运行时；GPU 只由实际 EmoVoice 推理进程占用。
manage() { env CUDA_VISIBLE_DEVICES=-1 "$PY" "$STEP/scripts/manage_full_tts.py" "$@"; }
release() { manage release-worker --db "$DB" --worker "$WORKER" >/dev/null 2>&1 || true; }
trap release EXIT INT TERM
release
exec >>"$ART/${WORKER}.log" 2>&1
echo "[$(date -Is)] worker=$WORKER gpu=$GPU_ID 启动 batch_size=$LIMIT"
manage init --db "$DB" --manifest "$MANIFEST"
while true; do
  exec 9>"$LOCK"; flock 9
  manage reconcile --db "$DB" --raw-dir "$RAW" --segment-dir "$SEGMENTS"
  manage assemble --db "$DB" --index "$INDEX" --segment-dir "$SEGMENTS" --final-dir "$FINAL"
  manage summary --db "$DB" --index "$INDEX" --final-dir "$FINAL" --max-attempts "$MAX_ATTEMPTS" --status-json "$ART/status.json"
  count=$(manage claim-batch --db "$DB" --batch "$BATCH" --limit "$LIMIT" --max-attempts "$MAX_ATTEMPTS" --worker "$WORKER" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["batch_size"])')
  flock -u 9
  if [[ "$count" == "0" ]]; then echo "[$(date -Is)] 无待领分段，worker 结束"; break; fi
  echo "[$(date -Is)] worker=$WORKER 领取分段数=$count"
  set +e
  "$STEP/scripts/run_emovoice_pp_batch.sh" "$BATCH" "$OUT" "$GPU_ID"
  code=$?
  set -e
  exec 9>"$LOCK"; flock 9
  manage reconcile --db "$DB" --raw-dir "$RAW" --segment-dir "$SEGMENTS"
  manage mark-missing --db "$DB" --batch "$BATCH" --segment-dir "$SEGMENTS" --worker "$WORKER" --message "worker $WORKER EmoVoice 未产出分段（code=$code）"
  flock -u 9
done
