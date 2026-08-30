#!/usr/bin/env python3
"""把 Qwen 自动评测的结构化逐样本标注严格规范为 EGA/VQ-a/CUA/DCA 指标 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EMOTIONS = {"Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def score(row: dict) -> dict:
    required = {"sample_id", "target_emotion", "predicted_emotion", "visual_integrity", "sticker_style", "visual_clarity", "cua", "dca"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{row.get('sample_id', '<unknown>')} 缺少字段: {sorted(missing)}")
    if row["target_emotion"] not in EMOTIONS or row["predicted_emotion"] not in EMOTIONS:
        raise ValueError(f"{row['sample_id']} 的七类情感标签非法")
    rating_names = ("visual_integrity", "sticker_style", "visual_clarity", "cua", "dca")
    ratings = [row[name] for name in rating_names]
    if not all(isinstance(value, (int, float)) and 1 <= value <= 5 for value in ratings):
        raise ValueError(f"{row['sample_id']} 的质量/对齐评分必须在 1 到 5")
    return {
        "sample_id": row["sample_id"],
        "metrics": {
            "EGA": 100.0 if row["target_emotion"] == row["predicted_emotion"] else 0.0,
            "VQ-a": sum(float(row[name]) for name in ("visual_integrity", "sticker_style", "visual_clarity")) / 3.0,
            "CUA": float(row["cua"]),
            "DCA": float(row["dca"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen = set()
    results = []
    for row in read_jsonl(args.annotations):
        if row["sample_id"] in seen:
            raise ValueError(f"重复标注 sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        results.append(score(row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    print(json.dumps({"samples": len(results), "metrics": ["EGA", "VQ-a", "CUA", "DCA"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
