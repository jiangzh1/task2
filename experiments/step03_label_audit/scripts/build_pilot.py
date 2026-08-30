#!/usr/bin/env python3
"""Build a deterministic target-sticker pilot directly from official Parquet files."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


SPLITS = {
    "train": ("train-00000-of-00001.parquet", "official_train.parquet"),
    "validation": ("validation-00000-of-00001.parquet", "official_validation.parquet"),
    "test": ("test-00000-of-00001.parquet", "official_test.parquet"),
}
LABELS = ["Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"]


def stable_id(split: str, row_index: int, turn_index: int, image_path: str) -> str:
    raw = f"{split}|{row_index}|{turn_index}|{image_path}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def collect(data_dir: Path) -> list[dict]:
    records = []
    for split, filenames in SPLITS.items():
        source_path = next((data_dir / name for name in filenames if (data_dir / name).exists()), None)
        if source_path is None:
            raise FileNotFoundError(f"No official Parquet file found for {split}: {filenames}")
        rows = pq.read_table(source_path).to_pylist()
        for row_index, row in enumerate(rows):
            turns = row.get("conversations") or []
            image_indexes = [i for i, turn in enumerate(turns) if turn.get("image")]
            if not image_indexes:
                continue
            target_index = image_indexes[-1]
            # A sticker in the initial turn has no preceding dialogue context and is
            # unusable for the current-response construction.
            if target_index < 1:
                continue
            turn = turns[target_index]
            image = turn["image"]
            image_path = str(image.get("image"))
            label = image.get("origin_anno")
            if label not in LABELS:
                continue
            records.append(
                {
                    "sample_id": stable_id(split, row_index, target_index, image_path),
                    "split": split,
                    "source_row": row_index,
                    "target_turn_index": target_index,
                    "target_role": turn.get("role"),
                    "current_text": turn.get("content"),
                    "context": [
                        {"role": t.get("role"), "content": t.get("content")}
                        for t in turns[:target_index]
                    ],
                    "sticker": {
                        "image_path": image_path,
                        "origin_anno": label,
                        "description": image.get("description"),
                        "emotion_description": image.get("emotion"),
                        "recommendation": image.get("recommendation"),
                    },
                    "audit": {
                        "vision_label": None,
                        "label_status": "pending",
                        "corrected_label": None,
                        "text_polarity": None,
                        "conflict_label": None,
                    },
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    all_records = collect(args.data_dir)
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for record in all_records:
        grouped[record["sticker"]["origin_anno"]].append(record)

    rng = random.Random(args.seed)
    pilot = []
    for label in LABELS:
        candidates = grouped[label]
        rng.shuffle(candidates)
        pilot.extend(candidates[: min(args.per_class, len(candidates))])
    pilot.sort(key=lambda x: (LABELS.index(x["sticker"]["origin_anno"]), x["sample_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in pilot:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source": "official STICKERCONV Parquet only",
        "seed": args.seed,
        "per_class_requested": args.per_class,
        "all_target_counts": dict(collections.Counter(r["sticker"]["origin_anno"] for r in all_records)),
        "pilot_counts": dict(collections.Counter(r["sticker"]["origin_anno"] for r in pilot)),
        "pilot_total": len(pilot),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
