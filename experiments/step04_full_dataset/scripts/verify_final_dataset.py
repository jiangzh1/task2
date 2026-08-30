#!/usr/bin/env python3
"""Verify final dataset counts, schema, and frozen label invariants."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
MAPPING = {
    "Happiness": "Positive", "Surprise": "Positive", "Sadness": "Negative", "Anger": "Negative",
    "Disgust": "Negative", "Fear": "Negative", "Neutral": "Neutral",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, overall_ids, overall_labels = {"passed": True, "splits": {}}, set(), collections.Counter()
    for split in SPLITS:
        path = args.dataset_dir / f"spchconvsti_{split}.jsonl"
        ids, labels, invalid = set(), collections.Counter(), []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                derived = row.get("derived_labels", {})
                sample_id = row.get("sample_id")
                if sample_id in ids or sample_id in overall_ids:
                    invalid.append(f"duplicate sample_id at line {line_number}")
                ids.add(sample_id)
                official = row.get("sticker", {}).get("origin_anno")
                if derived.get("sticker_label_source") != "official_origin_anno" or derived.get("sticker_emotion_official") != official:
                    invalid.append(f"official label provenance failure at line {line_number}")
                if derived.get("sticker_polarity") != MAPPING.get(official):
                    invalid.append(f"polarity mapping failure at line {line_number}")
                text, sticker, label = derived.get("text_polarity"), derived.get("sticker_polarity"), derived.get("conflict_label")
                expected = "Conflict" if {text, sticker} == {"Positive", "Negative"} else "Consistent"
                if label != expected:
                    invalid.append(f"conflict rule failure at line {line_number}")
                if label not in {"Conflict", "Consistent"}:
                    invalid.append(f"non-binary label at line {line_number}")
                labels[label] += 1
        overall_ids.update(ids)
        overall_labels.update(labels)
        report["splits"][split] = {"records": len(ids), "labels": dict(sorted(labels.items())), "invalid_records": len(invalid), "first_errors": invalid[:5]}
        if invalid:
            report["passed"] = False
    report["total_records"] = len(overall_ids)
    report["overall_labels"] = dict(sorted(overall_labels.items()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
