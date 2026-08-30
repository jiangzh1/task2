#!/usr/bin/env python3
"""Assess whether seven-class sticker disagreements affect binary conflict labels."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


BASE_POLARITY = {
    "Happiness": "Positive",
    "Sadness": "Negative",
    "Anger": "Negative",
    "Disgust": "Negative",
    "Fear": "Negative",
    "Neutral": "Neutral",
}


def read_ok(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows[row["sample_id"]] = row
    return rows


def conflict(text: str, sticker: str) -> bool:
    return {text, sticker} == {"Positive", "Negative"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gemma, qwen = read_ok(args.gemma), read_ok(args.qwen)
    raw_rows = []
    for sample_id in sorted(set(gemma) & set(qwen)):
        g, q = gemma[sample_id], qwen[sample_id]
        official = g["source_origin_anno"]
        gv = g["sticker_inference"]["parsed"]["primary_label"]
        qv = q["sticker_inference"]["parsed"]["primary_label"]
        text_g = g["text_inference"]["parsed"]["polarity"]
        text_q = q["text_inference"]["parsed"]["polarity"]
        sticker_label = gv if gv == qv else official
        raw_rows.append(
            {
                "official_label": official,
                "consensus_label_or_official": sticker_label,
                "models_agree_image": gv == qv,
                "text_agree": text_g == text_q,
                "text_polarity": text_g if text_g == text_q else None,
            }
        )

    variants = {}
    for surprise_polarity in ("Neutral", "Positive", "Negative"):
        polarity = {**BASE_POLARITY, "Surprise": surprise_polarity}
        rows = [
            {**r, "official_polarity": polarity[r["official_label"]],
             "reviewed_polarity": polarity[r["consensus_label_or_official"]]}
            for r in raw_rows
        ]
        high = [r for r in rows if r["models_agree_image"] and r["official_label"] != r["consensus_label_or_official"]]
        eligible = [r for r in rows if r["text_agree"]]
        variants[surprise_polarity] = {
            "candidate_category_change_same_polarity": sum(r["official_polarity"] == r["reviewed_polarity"] for r in high),
            "candidate_category_change_different_polarity": sum(r["official_polarity"] != r["reviewed_polarity"] for r in high),
            "official_conflicts_among_text_agreements": sum(conflict(r["text_polarity"], r["official_polarity"]) for r in eligible),
            "reviewed_sticker_conflicts_among_text_agreements": sum(conflict(r["text_polarity"], r["reviewed_polarity"]) for r in eligible),
            "conflict_label_changed_by_sticker_review": sum(conflict(r["text_polarity"], r["official_polarity"]) != conflict(r["text_polarity"], r["reviewed_polarity"]) for r in eligible),
        }

    high = [r for r in raw_rows if r["models_agree_image"] and r["official_label"] != r["consensus_label_or_official"]]
    image_consensus = [r for r in raw_rows if r["models_agree_image"]]
    eligible = [r for r in raw_rows if r["text_agree"]]

    summary = {
        "assumption": "Sensitivity analysis tests three alternative mappings for Surprise; none is a frozen final policy.",
        "total": len(rows),
        "image_models_agree": len(image_consensus),
        "two_model_nonofficial_candidates": len(high),
        "text_models_agree": len(eligible),
        "surprise_mapping_variants": variants,
        "policy_note": "Sensitivity analysis only; no label is changed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
