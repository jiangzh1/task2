#!/usr/bin/env python3
"""为版本 B 构建可断点续跑的 EmoVoice-PP 分段清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EMOTION_MAP = {
    "Happiness": ("happy", "clear happiness with a bright, lively, warm and upbeat speaking style"),
    "Sadness": ("sad", "clear sadness with a subdued, heavy, sorrowful and gently slowed speaking style"),
    "Anger": ("angry", "strong anger with a tense, forceful, sharp and emphatic speaking style"),
    "Surprise": ("surprised", "clear surprise with startled energy, rising pitch and quick emphatic reactions"),
    "Disgust": ("disgusted", "clear disgust and aversion with a displeased, rejecting and scrunched-nose tone"),
    "Fear": ("fearful", "clear fear with tense, uneasy, trembling and apprehensive delivery"),
    "Neutral": ("neutral", "a calm, even, natural and emotionally neutral speaking style"),
}

REFERENCES = {
    "train": [
        "/data/jzh/2026/task2/experiments/step02_tts_pilot/official_reference/audio/neutral/gpt4o_23948_neutral_ash.wav",
        "/data/jzh/2026/task2/experiments/step02_tts_pilot/official_reference/audio/neutral/gpt4o_23664_neutral_verse.wav",
    ],
    "validation": [
        "/data/jzh/2026/task2/experiments/step02_tts_pilot/official_reference/audio/neutral/gpt4o_24397_neutral_coral.wav",
    ],
    "test": [
        "/data/jzh/2026/task2/experiments/step02_tts_pilot/official_reference/audio/neutral/gpt4o_24109_neutral_ballad.wav",
    ],
}

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def stable_choice(values: list[str], key: str) -> str:
    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(values)
    return values[index]


def split_text(text: str, max_words: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = sentence.split()
        while len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if not words:
            continue
        if current and current_words + len(words) > max_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(" ".join(words))
        current_words += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-words", type=int, default=55)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": args.dataset_dir / "spchconvsti_train.jsonl",
        "validation": args.dataset_dir / "spchconvsti_validation.jsonl",
        "test": args.dataset_dir / "spchconvsti_test.jsonl",
    }
    manifest_path = args.output_dir / "full_tts_segments.jsonl"
    index_path = args.output_dir / "full_tts_sample_index.jsonl"
    sample_count = segment_count = 0
    split_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    segment_hist: Counter[int] = Counter()

    with manifest_path.open("w", encoding="utf-8") as manifest, index_path.open("w", encoding="utf-8") as index:
        for split, source_path in split_files.items():
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            for line in source_path.open(encoding="utf-8"):
                row = json.loads(line)
                sample_id = row["sample_id"]
                emotion_official = row["derived_labels"]["sticker_emotion_official"]
                model_emotion, prompt = EMOTION_MAP[emotion_official]
                reference = stable_choice(REFERENCES[split], sample_id)
                chunks = split_text(row["current"]["text"], args.max_words)
                keys = []
                for segment_index, chunk in enumerate(chunks):
                    key = f"{sample_id}__s{segment_index:03d}"
                    keys.append(key)
                    item = {
                        "key": key,
                        "source_text": chunk,
                        "target_text": chunk,
                        "emotion": model_emotion,
                        "emotion_text_prompt": prompt,
                        "answer_cosyvoice_speech_token": [],
                        "neutral_speaker_wav": reference,
                        "metadata": {
                            "sample_id": sample_id,
                            "split": split,
                            "segment_index": segment_index,
                            "segment_count": len(chunks),
                            "sticker_emotion_official": emotion_official,
                        },
                    }
                    manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                    segment_count += 1
                index.write(json.dumps({
                    "sample_id": sample_id,
                    "split": split,
                    "sticker_emotion_official": emotion_official,
                    "model_emotion": model_emotion,
                    "reference_wav": reference,
                    "segment_keys": keys,
                    "word_count": len(row["current"]["text"].split()),
                }, ensure_ascii=False) + "\n")
                sample_count += 1
                split_counts[split] += 1
                emotion_counts[emotion_official] += 1
                segment_hist[len(chunks)] += 1

    summary = {
        "schema_version": "1.0.0",
        "source_dataset": str(args.dataset_dir),
        "max_words_per_segment": args.max_words,
        "samples": sample_count,
        "segments": segment_count,
        "split_counts": dict(split_counts),
        "emotion_counts": dict(emotion_counts),
        "segments_per_sample_histogram": {str(k): v for k, v in sorted(segment_hist.items())},
        "reference_policy": "按 split 隔离参考声线；train 内按 sample_id 的 SHA-256 确定性分配。",
        "emotion_policy": "语音目标情感严格取官方七类 sticker_emotion_official。",
    }
    (args.output_dir / "full_tts_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
