#!/usr/bin/env python3
"""Summarize a completed Ollama pilot without altering any source label."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


LABELS = ["Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"]
POLARITIES = ["Positive", "Negative", "Neutral"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    ok = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") != "ok"]
    confusion = {label: collections.Counter() for label in LABELS}
    text_by_official = {label: collections.Counter() for label in LABELS}
    agreements = collections.Counter()
    totals = collections.Counter()
    vision_confidences = []
    text_confidences = []
    ambiguous = collections.Counter()
    elapsed = []

    for row in ok:
        official = row["source_origin_anno"]
        vision = row["sticker_inference"]["parsed"]
        text = row["text_inference"]["parsed"]
        predicted = vision["primary_label"]
        polarity = text["polarity"]
        confusion[official][predicted] += 1
        text_by_official[official][polarity] += 1
        totals[official] += 1
        if official == predicted:
            agreements[official] += 1
        vision_confidences.append(float(vision["confidence"]))
        text_confidences.append(float(text["confidence"]))
        ambiguous["vision_true" if vision["ambiguous"] else "vision_false"] += 1
        ambiguous["text_true" if text["ambiguous"] else "text_false"] += 1
        elapsed.append(float(row["elapsed_seconds"]))

    summary = {
        "records": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "overall_official_vision_agreement": {
            "count": sum(agreements.values()),
            "total": len(ok),
            "rate": sum(agreements.values()) / len(ok) if ok else None,
        },
        "agreement_by_official": {
            label: {
                "count": agreements[label],
                "total": totals[label],
                "rate": agreements[label] / totals[label] if totals[label] else None,
            }
            for label in LABELS
        },
        "confusion_official_to_vision": {
            label: {pred: confusion[label][pred] for pred in LABELS} for label in LABELS
        },
        "text_polarity_overall": dict(
            collections.Counter(
                row["text_inference"]["parsed"]["polarity"] for row in ok
            )
        ),
        "text_polarity_by_official": {
            label: {pol: text_by_official[label][pol] for pol in POLARITIES} for label in LABELS
        },
        "self_reported_ambiguity": dict(ambiguous),
        "confidence": {
            "vision_mean": statistics.mean(vision_confidences) if vision_confidences else None,
            "vision_median": statistics.median(vision_confidences) if vision_confidences else None,
            "text_mean": statistics.mean(text_confidences) if text_confidences else None,
            "text_median": statistics.median(text_confidences) if text_confidences else None,
        },
        "elapsed_seconds": {
            "sum": sum(elapsed),
            "mean": statistics.mean(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
        },
        "policy_note": "Screening statistics only; no source label was modified.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
