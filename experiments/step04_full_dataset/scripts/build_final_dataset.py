#!/usr/bin/env python3
"""Build final binary labels from frozen official sticker labels and blind text judgements."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
STICKER_POLARITY = {
    "Happiness": "Positive",
    "Surprise": "Positive",
    "Sadness": "Negative",
    "Anger": "Negative",
    "Disgust": "Negative",
    "Fear": "Negative",
    "Neutral": "Neutral",
}


def read_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "2.0.0",
        "official_sticker_label_preserved": True,
        "sticker_polarity_mapping": STICKER_POLARITY,
        "conflict_rule": "Conflict iff text polarity and official sticker polarity are Positive/Negative in either order; otherwise Consistent.",
        "splits": {},
    }
    overall = collections.Counter()
    for split in SPLITS:
        manifest = read_jsonl(args.artifacts / f"official_manifest_{split}.jsonl")
        gemma = read_jsonl(args.artifacts / f"text_polarity_gemma3_{split}.jsonl")
        qwen = read_jsonl(args.artifacts / f"text_polarity_qwen3_{split}.jsonl")
        judge = read_jsonl(args.artifacts / f"text_adjudication_qwen3vl_{split}.jsonl")
        counts = collections.Counter()
        source_counts = collections.Counter()
        matrix = collections.Counter()
        roles = collections.Counter()
        output = args.output_dir / f"spchconvsti_{split}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for sample_id in sorted(manifest):
                record, g, q = manifest[sample_id], gemma.get(sample_id), qwen.get(sample_id)
                if not g or not q or g.get("status") != "ok" or q.get("status") != "ok":
                    raise RuntimeError(f"Missing successful first-pass inference: {split}/{sample_id}")
                gp = g["text_inference"]["parsed"]["polarity"]
                qp = q["text_inference"]["parsed"]["polarity"]
                if gp == qp:
                    text_polarity, source = gp, "Gemma3_Qwen3_agreement"
                else:
                    third = judge.get(sample_id)
                    if not third or third.get("status") != "ok":
                        raise RuntimeError(f"Missing successful third adjudication: {split}/{sample_id}")
                    text_polarity = third["adjudication"]["parsed"]["polarity"]
                    source = "Qwen3VL_blind_adjudication"
                official = record["sticker"]["origin_anno"]
                sticker_polarity = STICKER_POLARITY[official]
                label = "Conflict" if {text_polarity, sticker_polarity} == {"Positive", "Negative"} else "Consistent"
                record["derived_labels"] = {
                    "sticker_label_source": "official_origin_anno",
                    "sticker_emotion_official": official,
                    "sticker_polarity": sticker_polarity,
                    "surprise_mapping": "Positive",
                    "text_polarity": text_polarity,
                    "text_polarity_source": source,
                    "conflict_label": label,
                    "conflict_rule": summary["conflict_rule"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[label] += 1
                source_counts[source] += 1
                matrix[(text_polarity, sticker_polarity)] += 1
                roles[record["current"]["role"]] += 1
        summary["splits"][split] = {
            "samples": sum(counts.values()),
            "binary_label_distribution": dict(sorted(counts.items())),
            "text_polarity_source": dict(sorted(source_counts.items())),
            "text_by_sticker_polarity": {f"{text}|{sticker}": value for (text, sticker), value in sorted(matrix.items())},
            "target_role_distribution": dict(sorted(roles.items())),
        }
        overall.update(counts)
    summary["overall_binary_label_distribution"] = dict(sorted(overall.items()))
    summary["total_samples"] = sum(overall.values())
    (args.output_dir / "final_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
