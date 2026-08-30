#!/usr/bin/env python3
"""构建正式版本 B 目标 sticker 清单，并检查缺失、损坏和字节级跨划分泄漏。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def resolve_image(data_root: Path, recorded_path: str) -> Path:
    normalized = recorded_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    candidate = data_root / normalized
    return candidate.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_files = {
        "train": "spchconvsti_train.jsonl",
        "validation": "spchconvsti_validation.jsonl",
        "test": "spchconvsti_test.jsonl",
    }
    images: dict[str, dict] = {}
    sample_rows = []
    for split, filename in split_files.items():
        with (args.dataset_dir / filename).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                path = resolve_image(args.data_root, row["sticker"]["image_path"])
                key = str(path)
                info = images.setdefault(key, {"path": path, "splits": set(), "emotions": set(), "sample_ids": []})
                info["splits"].add(split)
                info["emotions"].add(row["derived_labels"]["sticker_emotion_official"])
                info["sample_ids"].append(row["sample_id"])
                sample_rows.append({
                    "sample_id": row["sample_id"],
                    "split": split,
                    "sticker_emotion": row["derived_labels"]["sticker_emotion_official"],
                    "image_path": key,
                })

    image_rows = []
    missing, corrupt = [], []
    hashes: dict[str, list[dict]] = defaultdict(list)
    for key in sorted(images):
        info = images[key]
        path: Path = info["path"]
        if not path.exists():
            missing.append(key)
            continue
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode, image_format = image.mode, image.format
                frames = getattr(image, "n_frames", 1)
            digest = sha256_file(path)
        except Exception as error:
            corrupt.append({"path": key, "error": repr(error)})
            continue
        row = {
            "image_path": key,
            "sha256": digest,
            "width": width,
            "height": height,
            "mode": mode,
            "format": image_format,
            "frames": frames,
            "splits": sorted(info["splits"]),
            "emotions": sorted(info["emotions"]),
            "sample_count": len(info["sample_ids"]),
        }
        image_rows.append(row)
        hashes[digest].append(row)

    hash_cross_split = []
    duplicate_hash_groups = 0
    for digest, rows in hashes.items():
        if len(rows) <= 1:
            continue
        duplicate_hash_groups += 1
        all_splits = sorted({split for row in rows for split in row["splits"]})
        if len(all_splits) > 1:
            hash_cross_split.append({
                "sha256": digest,
                "splits": all_splits,
                "paths": [row["image_path"] for row in rows],
            })

    with (args.output_dir / "sticker_images.jsonl").open("w", encoding="utf-8") as handle:
        for row in image_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "sample_to_sticker.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "passed" if not missing and not corrupt and not hash_cross_split else "attention_required",
        "samples": len(sample_rows),
        "unique_recorded_paths": len(images),
        "validated_images": len(image_rows),
        "missing_images": len(missing),
        "corrupt_images": len(corrupt),
        "path_cross_split_groups": sum(len(info["splits"]) > 1 for info in images.values()),
        "duplicate_hash_groups": duplicate_hash_groups,
        "byte_identical_cross_split_groups": len(hash_cross_split),
        "missing_details": missing,
        "corrupt_details": corrupt,
        "byte_identical_cross_split_details": hash_cross_split,
    }
    (args.output_dir / "sticker_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing and not corrupt else 2


if __name__ == "__main__":
    raise SystemExit(main())
