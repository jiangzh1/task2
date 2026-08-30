#!/usr/bin/env python3
"""Build the full task manifest directly from official STICKERCONV Parquet files."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "parquet_deps"))
import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")
VALID_LABELS = {"Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_id(split: str, row_index: int, turn_index: int, image_path: str) -> str:
    value = f"{split}|{row_index}|{turn_index}|{image_path}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def build_split(split: str, source: Path, output: Path) -> dict:
    rows = pq.read_table(source).to_pylist()
    labels = collections.Counter()
    role_counts = collections.Counter()
    excluded_initial = 0
    excluded_bad_label = 0
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            turns = row.get("conversations") or []
            image_turns = [index for index, turn in enumerate(turns) if turn.get("image")]
            if not image_turns:
                continue
            target_index = image_turns[-1]
            if target_index < 1:
                excluded_initial += 1
                continue
            target = turns[target_index]
            sticker = target["image"]
            official_label = sticker.get("origin_anno")
            if official_label not in VALID_LABELS:
                excluded_bad_label += 1
                continue
            record = {
                "schema_version": "2.0.0",
                "sample_id": sample_id(split, row_index, target_index, str(sticker.get("image"))),
                "split": split,
                "source": {"dataset": "STICKERCONV official Parquet", "row_index": row_index, "target_turn_index": target_index},
                "current": {"role": target.get("role"), "text": target.get("content")},
                "context": [{"role": turn.get("role"), "text": turn.get("content")} for turn in turns[:target_index]],
                "sticker": {
                    "image_path": sticker.get("image"),
                    "origin_anno": official_label,
                    "description": sticker.get("description"),
                    "emotion_description": sticker.get("emotion"),
                    "recommendation": sticker.get("recommendation"),
                    "seq_num": sticker.get("seq_num"),
                },
                "derived_labels": {
                    "sticker_label_source": "official_origin_anno",
                    "sticker_emotion_official": official_label,
                    "sticker_polarity": None,
                    "text_polarity": None,
                    "text_polarity_source": None,
                    "conflict_label": None,
                    "conflict_rule": None,
                    "sticker_review_status": "not_reviewed",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            labels[official_label] += 1
            role_counts[str(target.get("role"))] += 1
            written += 1
    return {
        "source_file": source.name,
        "source_sha256": sha256(source),
        "source_rows": len(rows),
        "usable_samples": written,
        "excluded_initial_only_sticker": excluded_initial,
        "excluded_unknown_official_label": excluded_bad_label,
        "official_sticker_label_distribution": dict(sorted(labels.items())),
        "target_role_distribution": dict(sorted(role_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "2.0.0", "source_policy": "official Parquet only; legacy Cleaned/Refined files not used", "splits": {}}
    for split in SPLITS:
        source = args.data_dir / f"official_{split}.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        summary["splits"][split] = build_split(split, source, args.output_dir / f"official_manifest_{split}.jsonl")
    summary["total_usable_samples"] = sum(item["usable_samples"] for item in summary["splits"].values())
    (args.output_dir / "official_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
