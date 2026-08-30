#!/usr/bin/env python3
"""Assess whether a group-aware B-policy stratified split can avoid empty cells."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_image, strata = collections.defaultdict(list), collections.Counter()
    for split in SPLITS:
        with (args.dataset_dir / f"spchconvsti_{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["sticker"]["origin_anno"], row["derived_labels"]["conflict_label"])
                image = row["sticker"]["image_path"]
                by_image[image].append(key)
                strata[key] += 1
    image_strata = collections.defaultdict(set)
    for image, keys in by_image.items():
        image_strata[tuple(sorted(set(keys)))].add(image)
    per_stratum_images = collections.Counter()
    for image, keys in by_image.items():
        for key in set(keys):
            per_stratum_images[key] += 1
    result = {
        "samples": sum(strata.values()),
        "unique_target_images": len(by_image),
        "reused_image_samples": sum(len(v) for v in by_image.values() if len(v) > 1),
        "max_samples_per_image": max(map(len, by_image.values())),
        "strata": {f"{emotion}|{label}": {"samples": strata[(emotion, label)], "unique_images": per_stratum_images[(emotion, label)]} for emotion, label in sorted(strata)},
        "multi_stratum_image_groups": sum(1 for keys in image_strata if len(keys) > 1),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
