"""按论文 3.1.1 提取小样本真实音频与文本特征，支持安全断点续跑。"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import opensmile
import soundfile as sf
import torch
import torch.nn.functional as F
import whisper
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_by_word(hidden: torch.Tensor, word_ids: list[int | None], count: int) -> torch.Tensor:
    vectors = []
    for word_index in range(count):
        positions = [i for i, token_word_id in enumerate(word_ids) if token_word_id == word_index]
        vectors.append(hidden[positions].mean(dim=0) if positions else torch.zeros_like(hidden[0]))
    return torch.stack(vectors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--formal-manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wavlm-layer", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 条，用于可控的单样本冒烟")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    formal: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        formal.update({row["sample_id"]: row for row in read_jsonl(args.formal_manifest_dir / f"spchconvsti_{split}.jsonl")})
    subset = read_jsonl(args.subset)
    if args.limit is not None:
        subset = subset[: args.limit]

    # 预先通过 bootstrap_feature_models.py 缓存模型。此处必须禁止 Hub
    # 校验/补下载：服务器网络波动不应影响已完整缓存的小样本验证。
    offline = {"cache_dir": args.cache_dir, "local_files_only": True}
    # RoBERTa 对预分词（Whisper 词级时间戳）输入要求 add_prefix_space=True，
    # 才能取得与原始词索引一一对应的 word_ids。
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True, **offline)
    roberta = AutoModel.from_pretrained("roberta-base", **offline).to(device).eval()
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus", **offline)
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus", **offline).to(device).eval()
    for parameter in wavlm.parameters():
        parameter.requires_grad_(False)
    whisper_model = whisper.load_model("base", download_root=str(args.cache_dir / "whisper"), device=str(device))
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    completed, failures, diagnostics = [], [], []
    for item in subset:
        sample_id = item["sample_id"]
        started_at = time.monotonic()
        target = args.output_dir / f"{sample_id}.pt"
        if target.is_file():
            completed.append(sample_id)
            continue
        try:
            waveform, sample_rate = sf.read(item["audio_path"], dtype="float32", always_2d=False)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if sample_rate != 16000:
                waveform = F.interpolate(torch.from_numpy(waveform)[None, None], size=round(len(waveform) * 16000 / sample_rate), mode="linear", align_corners=False)[0, 0].numpy()

            # 传入已读取的 16 kHz 波形，避免 Whisper 通过外部 ffmpeg 再次读取文件。
            transcription = whisper_model.transcribe(waveform, language="en", word_timestamps=True, fp16=device.type == "cuda", verbose=False)
            words = [word_info["word"].strip() for segment in transcription.get("segments", []) for word_info in segment.get("words", []) if word_info["word"].strip()]
            spans = [[float(word_info["start"]), float(word_info["end"])] for segment in transcription.get("segments", []) for word_info in segment.get("words", []) if word_info["word"].strip()]
            if not words or len(words) != len(spans):
                raise RuntimeError("Whisper 未返回可用的词级时间戳")

            encoded_words = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            word_ids = encoded_words.word_ids(batch_index=0)
            encoded_words = {key: value.to(device) for key, value in encoded_words.items()}
            with torch.no_grad():
                word_hidden = roberta(**encoded_words).last_hidden_state[0]
            kept_word_count = max((word_id for word_id in word_ids if word_id is not None), default=-1) + 1
            text_features = mean_by_word(word_hidden, word_ids, kept_word_count).cpu()
            spans = torch.tensor(spans[:kept_word_count], dtype=torch.float32)

            context_text = " ".join(turn["text"] for turn in formal[sample_id]["context"])
            encoded_context = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=512)
            encoded_context = {key: value.to(device) for key, value in encoded_context.items()}
            with torch.no_grad():
                context_features = roberta(**encoded_context).last_hidden_state[0].cpu()

            wavlm_inputs = wavlm_processor(waveform, sampling_rate=16000, return_tensors="pt")
            wavlm_inputs = {key: value.to(device) for key, value in wavlm_inputs.items()}
            with torch.no_grad():
                wavlm_output = wavlm(**wavlm_inputs, output_hidden_states=True)
            acoustic = wavlm_output.hidden_states[args.wavlm_layer].squeeze(0).cpu()

            lld = smile.process_signal(waveform, 16000).to_numpy(dtype=np.float32)
            if lld.shape[0] == 0:
                raise RuntimeError("OpenSMILE 未返回 LLD 帧")
            prosody = F.interpolate(torch.from_numpy(lld).T[None], size=acoustic.shape[0], mode="linear", align_corners=False)[0].T.contiguous()
            torch.save(
                {
                    "sample_id": sample_id,
                    "emotion": item["sticker_emotion_official"],
                    "audio_path": item["audio_path"],
                    "asr_text": transcription["text"].strip(),
                    "asr_words": words[:kept_word_count],
                    "text": text_features,
                    "context": context_features,
                    "acoustic": acoustic,
                    "prosody_lld": prosody,
                    "word_timestamps": spans,
                    "metadata": {
                        "sample_rate": 16000,
                        "audio_duration_seconds": len(waveform) / 16000.0,
                        # WavLM-base-plus 的特征卷积总 stride 为 320 个 16 kHz 样本，即 50 Hz。
                        # 该值供缓存批处理时把 Whisper 的秒级词边界稳定映射到声学帧。
                        "wavlm_frame_rate": 50.0,
                        "wavlm_layer": args.wavlm_layer,
                        "wavlm_frozen": True,
                        "roberta": "roberta-base",
                        "opensmile": "eGeMAPSv02/LLD",
                        "whisper": "base",
                    },
                },
                target,
            )
            completed.append(sample_id)
            diagnostics.append({"sample_id": sample_id, "asr_words": len(words), "roberta_words": kept_word_count, "wavlm_frames": int(acoustic.shape[0]), "lld_dim": int(prosody.shape[1]), "elapsed_seconds": round(time.monotonic() - started_at, 2)})
        except Exception as error:  # 保留失败信息，下一次可安全重跑该样本。
            failures.append({"sample_id": sample_id, "error": repr(error), "traceback": traceback.format_exc(), "elapsed_seconds": round(time.monotonic() - started_at, 2)})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"total": len(subset), "completed": len(completed), "failures": failures, "diagnostics": diagnostics, "purpose": "真实特征小样本冒烟；不构成正式训练或指标。"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"存在 {len(failures)} 条失败样本，详见 {args.report}")


if __name__ == "__main__":
    main()
