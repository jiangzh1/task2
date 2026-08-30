#!/usr/bin/env python3
"""Summarize the thesis statistics table from a policy-variant dataset."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
EMOTIONS = ("Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = collections.Counter()
    for split in SPLITS:
        with (args.dataset_dir / f"spchconvsti_{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts[(row["sticker"]["origin_anno"], row["derived_labels"]["conflict_label"], split)] += 1
    rows, totals = {}, collections.Counter()
    for emotion in EMOTIONS:
        row = {}
        for label in ("Consistent", "Conflict"):
            for split in SPLITS:
                row[f"{label}_{split}"] = counts[(emotion, label, split)]
                totals[(label, split)] += row[f"{label}_{split}"]
        row["Total"] = sum(row.values())
        rows[emotion] = row
    result = {
        "policy": args.policy,
        "rows": rows,
        "total": {
            **{f"Consistent_{split}": totals[("Consistent", split)] for split in SPLITS},
            **{f"Conflict_{split}": totals[("Conflict", split)] for split in SPLITS},
            "Total": sum(totals.values()),
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
