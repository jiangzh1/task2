#!/usr/bin/env bash
# 完整音频全部就绪后的一次性可恢复编排；本文件不会自动执行。
set -euo pipefail
ROOT=/data/jzh/2026/task2_sdxl_work
FORMAL=/data/jzh/2026/task2/experiments/step04_full_dataset/policy_variants/neutral_mismatch_hash_stratified_811
AUDIO=/data/jzh/2026/task2/experiments/step04_full_dataset/audio/full_tts_pp_hashsafe_v3_multigpu/final_16k
ASSETS=/data/jzh/2026/task2_sdxl_assets
PY=/data/jzh/2026/task2/experiments/step02_tts_pilot/.venv/bin/python
FEATURE_CACHE=/data/jzh/2026/task2/experiments/step14_real_audio_smoke/model_cache
LATENTS="$ROOT/experiments/step11_feature_preprocessing/artifacts/sdxl_512_latents"
INDEX="$ROOT/experiments/step11_feature_preprocessing/artifacts/training_index_sdxl"
FEATURES="$ROOT/experiments/step14_real_audio_smoke/artifacts/full_features"

: "${STAGE1_EPOCHS:?必须明确正式 Stage 1 epochs}"
: "${STAGE1_BATCH_SIZE:?必须明确正式 Stage 1 batch size（至少为 2）}"
: "${STAGE1_LR:?必须明确正式 Stage 1 learning rate}"
: "${LAMBDA_CODE:?必须明确 lambda_code}"
: "${LAMBDA_ALIGN:?必须明确 lambda_align}"
: "${CODEBOOK_SIZE:?必须明确 VQ codebook size}"

"$PY" "$ROOT/experiments/step11_feature_preprocessing/scripts/build_training_index.py" \
  --dataset-dir "$FORMAL" --latent-dir "$LATENTS" --audio-root "$AUDIO" --output-dir "$INDEX"

for split in train validation test; do
  "$PY" "$ROOT/experiments/step14_real_audio_smoke/scripts/extract_real_features.py" \
    --subset "$INDEX/training_index_${split}.jsonl" --formal-manifest-dir "$FORMAL" \
    --cache-dir "$FEATURE_CACHE" --output-dir "$FEATURES/$split" \
    --report "$FEATURES/${split}_report.json" --device cuda
done

CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/experiments/step12_integrated_training/scripts/train_stage1_sdxl.py" \
  --training-index "$INDEX/training_index_train.jsonl" --feature-dir "$FEATURES/train" \
  --checkpoint "$ASSETS/sdxl_base/sd_xl_base_1.0.safetensors" --vae "$ASSETS/sdxl_vae" \
  --output-dir "$ROOT/experiments/step12_integrated_training/artifacts/formal_sdxl_stage1" \
  --epochs "$STAGE1_EPOCHS" --batch-size "$STAGE1_BATCH_SIZE" --learning-rate "$STAGE1_LR" \
  --lambda-code "$LAMBDA_CODE" --lambda-align "$LAMBDA_ALIGN" --codebook-size "$CODEBOOK_SIZE"

echo 'Stage 1 finished. Stage 2, final generation, and evaluation must use the selected formal settings and checkpoints.'
