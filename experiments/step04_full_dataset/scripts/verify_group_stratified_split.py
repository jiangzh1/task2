#!/usr/bin/env python3
"""Verify label consistency, image isolation, and non-empty strata in the new B split."""

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
    image_splits, ids, strata, report = {}, set(), collections.Counter(), {"passed": True, "splits": {}, "errors": []}
    for split in SPLITS:
        count = 0
        with (args.dataset_dir / f"spchconvsti_{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                count += 1
                if row["sample_id"] in ids:
                    report["errors"].append(f"duplicate sample_id at {split}:{line_no}")
                ids.add(row["sample_id"])
                image = row["sticker"]["image_path"]
                if image in image_splits and image_splits[image] != split:
                    report["errors"].append(f"image leakage for {image}")
                image_splits[image] = split
                derived = row["derived_labels"]
                if derived.get("conflict_policy") != "neutral_mismatch" or derived.get("conflict_label") != ("Conflict" if derived["text_polarity"] != derived["sticker_polarity"] else "Consistent"):
                    report["errors"].append(f"label-rule failure at {split}:{line_no}")
                strata[(row["sticker"]["origin_anno"], derived["conflict_label"], split)] += 1
        report["splits"][split] = {"records": count}
    nonempty = {f"{emotion}|{label}": {split: strata[(emotion, label, split)] for split in SPLITS} for emotion, label, _ in strata}
    empty = [key for key, values in nonempty.items() if any(value == 0 for value in values.values())]
    if empty:
        report["errors"].append(f"empty stratum cells: {empty}")
    report["total_records"] = len(ids)
    report["unique_target_images"] = len(image_splits)
    report["target_image_overlap"] = 0 if not any("image leakage" in error for error in report["errors"]) else 1
    report["empty_stratum_cells"] = empty
    report["passed"] = not report["errors"]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
