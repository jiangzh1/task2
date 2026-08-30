#!/usr/bin/env python3
"""构建七类情感 × 一致性 × 三种提示强度的 42 条诊断集。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


PROMPTS = {
    "Happiness": [
        "happy",
        "strong happiness, warm and upbeat",
        "joyful and excited, with bright lively energy",
    ],
    "Sadness": [
        "sad",
        "strong sadness, subdued and sorrowful",
        "deeply sad and heartbroken, with low energy",
    ],
    "Anger": [
        "angry",
        "strong anger, tense and forceful",
        "furious, intense anger, sharp emphasis and raised energy",
    ],
    "Surprise": [
        "pleasantly surprised",
        "strong positive surprise, bright and animated",
        "astonished and delighted, with excited rising intonation",
    ],
    "Disgust": [
        "disgusted",
        "strong disgust, aversion and repulsion",
        "intensely disgusted and repulsed, with sharp rejection",
    ],
    "Fear": [
        "fearful",
        "strong fear, tense and uneasy",
        "terrified and alarmed, with trembling urgent delivery",
    ],
    "Neutral": [
        "neutral",
        "calm and emotionally neutral",
        "completely neutral, steady and matter-of-fact",
    ],
}

LEVELS = ("simple", "strong", "intense")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    source_rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in source_rows:
        metadata = row["metadata"]
        strata[(metadata["official_sticker_emotion"], metadata["conflict_label"])].append(row)

    rows = []
    index_rows = []
    for (emotion, conflict_label), candidates in sorted(strata.items()):
        # 每个层选择文本最短者，降低长序列不停止的风险；三种提示共享文本、说话人。
        base = min(candidates, key=lambda row: len(row["source_text"]))
        for level, prompt in zip(LEVELS, PROMPTS[emotion]):
            row = json.loads(json.dumps(base))
            row["key"] = f"{base['key']}__{emotion.lower()}_{conflict_label.lower()}_{level}"
            row["emotion"] = emotion.lower()
            row["emotion_text_prompt"] = prompt
            row["metadata"]["diagnostic_level"] = level
            row["metadata"]["diagnostic_prompt"] = prompt
            rows.append(row)
            index_rows.append({
                "file": f"{row['key']}.wav",
                "emotion": emotion,
                "conflict_label": conflict_label,
                "prompt_level": level,
                "prompt": prompt,
                "text": row["source_text"],
                "reference_source": row["metadata"].get("reference_source"),
                "reference_speaker_id": row["metadata"].get("reference_speaker_id"),
            })

    if len(rows) != 42:
        raise ValueError(f"期望 42 条，实际 {len(rows)} 条；已有层数：{len(strata)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.index.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "strata": len(strata), "levels": list(LEVELS)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
