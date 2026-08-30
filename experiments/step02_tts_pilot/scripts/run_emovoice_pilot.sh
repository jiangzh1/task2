#!/usr/bin/env bash
set -euo pipefail

STEP=/data/jzh/2026/task2/experiments/step02_tts_pilot
CODE="$STEP/code/EmoVoice"
PYTHON="$STEP/.venv/bin/python"
INPUT="${1:-$STEP/artifacts/emovoice_smoke_1.ready.jsonl}"
RUN_NAME="${2:-smoke_1}"
GPU_ID="${3:-0}"
OUTPUT="$STEP/outputs/$RUN_NAME"

if [[ "$INPUT" != /* ]]; then
  INPUT="$STEP/$INPUT"
fi

export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

mkdir -p "$OUTPUT"
cd "$CODE"

"$PYTHON" examples/tts/inference_tts.py \
  hydra.run.dir="$OUTPUT/hydra" \
  ++model_config.llm_name=qwen2.5-0.5b \
  ++model_config.llm_path="$STEP/assets/Qwen2.5-0.5B" \
  ++model_config.llm_dim=896 \
  ++model_config.codec_decoder_path="$STEP/assets/EmoVoice/ckpts/CosyVoice/CosyVoice-300M-SFT" \
  ++model_config.codec_decode=true \
  ++model_config.vocab_config.code_layer=3 \
  ++model_config.vocab_config.total_audio_vocabsize=4160 \
  ++model_config.vocab_config.total_vocabsize=156160 \
  ++model_config.codec_decoder_type=CosyVoice \
  ++model_config.group_decode=true \
  ++model_config.group_decode_adapter_type=linear \
  ++model_config.use_text_stream=false \
  ++dataset_config.dataset=speech_dataset_tts \
  ++dataset_config.val_data_path="$INPUT" \
  ++dataset_config.train_data_path="$INPUT" \
  ++dataset_config.inference_mode=true \
  ++dataset_config.vocab_config.code_layer=3 \
  ++dataset_config.vocab_config.total_audio_vocabsize=4160 \
  ++dataset_config.vocab_config.total_vocabsize=156160 \
  ++dataset_config.num_latency_tokens=0 \
  ++dataset_config.do_layershift=false \
  ++dataset_config.use_emo=true \
  ++train_config.model_name=tts \
  ++train_config.freeze_encoder=true \
  ++train_config.freeze_llm=true \
  ++train_config.freeze_group_decode_adapter=true \
  ++train_config.batching_strategy=custom \
  ++train_config.num_epochs=1 \
  ++train_config.val_batch_size=1 \
  ++train_config.num_workers_dataloader=0 \
  ++decode_config.text_repetition_penalty=1.2 \
  ++decode_config.audio_repetition_penalty=1.2 \
  ++decode_config.max_new_tokens=3000 \
  ++decode_config.do_sample=false \
  ++decode_config.top_p=1.0 \
  ++decode_config.top_k=0 \
  ++decode_config.temperature=1.0 \
  ++decode_config.decode_text_only=false \
  ++decode_config.num_latency_tokens=0 \
  ++decode_config.do_layershift=false \
  ++decode_log="$OUTPUT/decode" \
  ++ckpt_path="$STEP/assets/EmoVoice/EmoVoice.pt" \
  ++output_text_only=false \
  ++speech_sample_rate=22050 \
  ++log_config.log_file="$OUTPUT/decode/infer.log"

"$PYTHON" "$STEP/scripts/resample_and_qc.py" \
  --input-dir "$OUTPUT/decode/pred_audio/neutral_prompt_speech" \
  --output-dir "$OUTPUT/audio_16k" \
  --report "$OUTPUT/audio_qc.json"
