#!/usr/bin/env python3
"""Create interchangeable binary datasets under two explicit conflict policies."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
POLICIES = {
    "strict_pn": {
        "name_zh": "正负冲突",
        "rule_zh": "仅当文本极性与表情包极性分别为 Positive 和 Negative（顺序不限）时标为 Conflict；其余组合均为 Consistent。",
        "rule_code": "Conflict iff {text_polarity, sticker_polarity} == {Positive, Negative}.",
    },
    "neutral_mismatch": {
        "name_zh": "极性不一致（含中性）",
        "rule_zh": "当文本极性与表情包极性不同，且其中至少一方为 Neutral 时，或二者为 Positive/Negative 配对时标为 Conflict；即只有相同极性配对（Positive/Positive、Negative/Negative、Neutral/Neutral）标为 Consistent。",
        "rule_code": "Conflict iff text_polarity != sticker_polarity.",
    },
}


def conflict(text: str, sticker: str, policy: str) -> bool:
    if policy == "strict_pn":
        return {text, sticker} == {"Positive", "Negative"}
    if policy == "neutral_mismatch":
        return text != sticker
    raise ValueError(policy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = {"source": "final_dataset generated from official origin_anno and frozen text polarity", "policies": {}}
    for policy, definition in POLICIES.items():
        target_dir = args.output_root / policy
        target_dir.mkdir(parents=True, exist_ok=True)
        policy_summary, total = {"definition": definition, "splits": {}}, collections.Counter()
        for split in SPLITS:
            source = args.source_dir / f"spchconvsti_{split}.jsonl"
            destination = target_dir / f"spchconvsti_{split}.jsonl"
            labels = collections.Counter()
            with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
                for line in reader:
                    record = json.loads(line)
                    derived = record["derived_labels"]
                    label = "Conflict" if conflict(derived["text_polarity"], derived["sticker_polarity"], policy) else "Consistent"
                    derived["conflict_label"] = label
                    derived["conflict_policy"] = policy
                    derived["conflict_rule"] = definition["rule_code"]
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    labels[label] += 1
            policy_summary["splits"][split] = {"samples": sum(labels.values()), "binary_label_distribution": dict(sorted(labels.items()))}
            total.update(labels)
        policy_summary["overall_binary_label_distribution"] = dict(sorted(total.items()))
        policy_summary["total_samples"] = sum(total.values())
        (target_dir / "dataset_summary.json").write_text(json.dumps(policy_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["policies"][policy] = policy_summary
    (args.output_root / "policy_variants_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
