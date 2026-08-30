#!/usr/bin/env python3
"""Build auditable binary pilot labels while preserving the official sticker label."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


STICKER_POLARITY = {
    "Happiness": "Positive",
    "Surprise": "Positive",
    "Sadness": "Negative",
    "Anger": "Negative",
    "Disgust": "Negative",
    "Fear": "Negative",
    "Neutral": "Neutral",
}


def read_rows(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--qwen-vl", type=Path, required=True)
    parser.add_argument("--text-adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    pilot = read_rows(args.pilot)
    gemma = read_rows(args.gemma)
    qwen = read_rows(args.qwen_vl)
    adjudication = read_rows(args.text_adjudication)
    output = []
    label_count = collections.Counter()
    text_source = collections.Counter()
    sticker_review = collections.Counter()
    adjudication_match = collections.Counter()

    for sample_id in sorted(pilot):
        record, g, q = pilot[sample_id], gemma.get(sample_id), qwen.get(sample_id)
        if not g or not q or g.get("status") != "ok" or q.get("status") != "ok":
            raise RuntimeError(f"Missing successful first-pass inference: {sample_id}")
        official = record["sticker"]["origin_anno"]
        gp = g["text_inference"]["parsed"]["polarity"]
        qp = q["text_inference"]["parsed"]["polarity"]
        gv = g["sticker_inference"]["parsed"]["primary_label"]
        qv = q["sticker_inference"]["parsed"]["primary_label"]
        if gp == qp:
            text = gp
            source = "Gemma_QwenVL_一致"
        else:
            judge = adjudication.get(sample_id)
            if not judge or judge.get("status") != "ok":
                raise RuntimeError(f"Missing successful text adjudication: {sample_id}")
            text = judge["adjudication"]["parsed"]["polarity"]
            source = "Qwen3_裁决"
            adjudication_match["matches_gemma" if text == gp else "not_gemma"] += 1
            adjudication_match["matches_qwen_vl" if text == qp else "not_qwen_vl"] += 1

        if official == gv == qv:
            image_status = "三方一致"
        elif gv == qv and gv != official:
            image_status = "双模型一致反对官方"
        elif official == gv:
            image_status = "官方与Gemma一致"
        elif official == qv:
            image_status = "官方与Qwen一致"
        else:
            image_status = "三方各异"

        sticker_polarity = STICKER_POLARITY[official]
        is_conflict = {text, sticker_polarity} == {"Positive", "Negative"}
        label = "Conflict" if is_conflict else "Consistent"
        derived = {
            **record,
            "derived_labels": {
                "sticker_label_source": "official_origin_anno",
                "sticker_emotion_official": official,
                "sticker_polarity": sticker_polarity,
                "surprise_mapping": "Positive",
                "text_polarity": text,
                "text_polarity_source": source,
                "conflict_label": label,
                "conflict_rule": "Conflict iff text polarity and official sticker polarity are Positive/Negative in either order; otherwise Consistent.",
                "sticker_review_status": image_status,
                "gemma_sticker_candidate": gv,
                "qwen_sticker_candidate": qv,
                "gemma_text_polarity": gp,
                "qwen_vl_text_polarity": qp,
            },
        }
        output.append(derived)
        label_count[label] += 1
        text_source[source] += 1
        sticker_review[image_status] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "samples": len(output),
        "official_sticker_label_preserved": True,
        "sticker_polarity_mapping": STICKER_POLARITY,
        "conflict_rule": "Conflict iff text polarity and official sticker polarity are Positive/Negative in either order; otherwise Consistent.",
        "conflict_distribution": dict(label_count),
        "text_polarity_source": dict(text_source),
        "sticker_review_status": dict(sticker_review),
        "third_model_adjudication_match": dict(adjudication_match),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
