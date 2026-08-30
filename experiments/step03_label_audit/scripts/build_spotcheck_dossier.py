#!/usr/bin/env python3
"""Create a stratified visual spot-check dossier from two-model consensus candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ["Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"]


def read(path: Path) -> dict[str, dict]:
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
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=1)
    args = parser.parse_args()

    pilot, gemma, qwen = read(args.pilot), read(args.gemma), read(args.qwen)
    candidates = []
    for sample_id, record in pilot.items():
        g, q = gemma.get(sample_id), qwen.get(sample_id)
        if not g or not q or g.get("status") != "ok" or q.get("status") != "ok":
            continue
        original = record["sticker"]["origin_anno"]
        gv = g["sticker_inference"]["parsed"]["primary_label"]
        qv = q["sticker_inference"]["parsed"]["primary_label"]
        if gv == qv and gv != original:
            candidates.append(
                {
                    "sample_id": sample_id,
                    "official_label": original,
                    "model_consensus": gv,
                    "image_path": record["sticker"]["image_path"],
                    "official_description": record["sticker"]["description"],
                    "official_emotion_description": record["sticker"]["emotion_description"],
                    "gemma_evidence": g["sticker_inference"]["parsed"]["visual_evidence"],
                    "qwen_evidence": q["sticker_inference"]["parsed"]["visual_evidence"],
                }
            )
    selected = []
    for label in LABELS:
        selected.extend([c for c in candidates if c["official_label"] == label][: args.per_class])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "selected": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
