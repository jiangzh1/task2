"""从已完成的正式语音中固定抽取七类情感各两条，用于非正式的端到端试运行。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EMOTIONS = ("Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-emotion", type=int, default=2)
    args = parser.parse_args()
    audio_root = args.audio_root.resolve()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in read_jsonl(args.sample_index):
        emotion = sample.get("sticker_emotion_official")
        # audio_root 必须指向 final_16k；以绝对路径写入清单，避免从
        # step14 的工作目录执行时把相对路径解析到错误位置。
        audio_path = audio_root / sample["split"] / f'{sample["sample_id"]}.wav'
        if emotion in EMOTIONS and audio_path.is_file():
            grouped[emotion].append({**sample, "audio_path": str(audio_path)})

    selected: list[dict] = []
    available: dict[str, int] = {}
    for emotion in EMOTIONS:
        # 先选文本最短的已完成样本：真实链路冒烟不需要长篇语音，
        # 这样可避免 Whisper 在 CPU 上被偶然的超长样本长期占用。
        candidates = sorted(grouped[emotion], key=lambda item: (item.get("word_count", 10**9), item["sample_id"]))
        available[emotion] = len(candidates)
        if len(candidates) < args.per_emotion:
            raise RuntimeError(f"{emotion} 仅有 {len(candidates)} 条完整音频，少于要求的 {args.per_emotion} 条")
        selected.extend(candidates[: args.per_emotion])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "purpose": "仅用于真实音频端到端冒烟测试，不能替代正式训练或正式实验结果",
                "per_emotion": args.per_emotion,
                "available_completed_audio": available,
                "selected_samples": len(selected),
                "emotions": list(EMOTIONS),
                "sample_ids": [item["sample_id"] for item in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
