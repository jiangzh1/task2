#!/usr/bin/env python3
"""Build a balanced 28-item EmoVoice pilot from the final version-B split."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


SPLITS = ("train", "validation", "test")
PROMPTS = {
    "Happiness": "clear happiness, warm enthusiasm, and an upbeat lively tone",
    "Sadness": "genuine sadness, subdued energy, and a soft sorrowful tone",
    "Anger": "controlled anger, firm emphasis, and tense forceful delivery",
    "Surprise": "positive surprise, bright excitement, and an animated rising intonation",
    "Disgust": "clear disgust, aversion, and a restrained repulsed tone",
    "Fear": "fear and unease, with cautious tension and slightly trembling delivery",
    "Neutral": "a calm neutral tone with natural pacing and no strong emotion",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--neutral-wav", default="PENDING_REFERENCE_LIBRARY")
    args = parser.parse_args()
    candidates = collections.defaultdict(list)
    for split in SPLITS:
        with (args.dataset_dir / f"spchconvsti_{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = (row["current"]["text"] or "").strip()
                if 20 <= len(text) <= 320:
                    key = (row["sticker"]["origin_anno"], row["derived_labels"]["conflict_label"])
                    candidates[key].append(row)
    rng = random.Random(args.seed)
    selected = []
    for key in sorted(candidates):
        rows = candidates[key]
        rng.shuffle(rows)
        rows.sort(key=lambda row: abs(len(row["current"]["text"]) - 120))
        for row in rows[: args.per_stratum]:
            emotion = row["sticker"]["origin_anno"]
            text = row["current"]["text"].strip()
            selected.append({
                "key": row["sample_id"],
                "source_text": text,
                "target_text": text,
                "emotion": emotion.lower(),
                "emotion_text_prompt": PROMPTS[emotion],
                "neutral_speaker_wav": args.neutral_wav,
                "metadata": {
                    "dataset_split": row["split"],
                    "conflict_label": row["derived_labels"]["conflict_label"],
                    "text_polarity": row["derived_labels"]["text_polarity"],
                    "sticker_polarity": row["derived_labels"]["sticker_polarity"],
                    "official_sticker_emotion": emotion,
                },
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "samples": len(selected),
        "per_stratum": args.per_stratum,
        "counts": {f"{emotion}|{label}": sum(row["metadata"]["official_sticker_emotion"] == emotion and row["metadata"]["conflict_label"] == label for row in selected) for emotion, label in sorted(candidates)},
        "emotion_prompt_mapping": PROMPTS,
        "speech_emotion_source": "official sticker origin_anno",
        "reference_wav_status": "pending" if args.neutral_wav == "PENDING_REFERENCE_LIBRARY" else "assigned",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
