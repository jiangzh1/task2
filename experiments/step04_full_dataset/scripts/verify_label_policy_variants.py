#!/usr/bin/env python3
"""Verify binary label invariants for both switchable policy variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")


def expected(text: str, sticker: str, policy: str) -> str:
    if policy == "strict_pn":
        return "Conflict" if {text, sticker} == {"Positive", "Negative"} else "Consistent"
    if policy == "neutral_mismatch":
        return "Conflict" if text != sticker else "Consistent"
    raise ValueError(policy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": True, "policies": {}}
    for policy in ("strict_pn", "neutral_mismatch"):
        policy_report, total, ids = {"splits": {}}, 0, set()
        for split in SPLITS:
            path = args.root / policy / f"spchconvsti_{split}.jsonl"
            count, errors = 0, []
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    row = json.loads(line)
                    derived = row["derived_labels"]
                    count += 1
                    if row["sample_id"] in ids:
                        errors.append(f"duplicate sample_id: {line_no}")
                    ids.add(row["sample_id"])
                    if derived.get("conflict_policy") != policy or derived.get("conflict_label") != expected(derived["text_polarity"], derived["sticker_polarity"], policy):
                        errors.append(f"rule mismatch: {line_no}")
            total += count
            policy_report["splits"][split] = {"records": count, "errors": len(errors), "first_errors": errors[:3]}
            if errors:
                report["passed"] = False
        policy_report["total_records"] = total
        policy_report["unique_sample_ids"] = len(ids)
        report["policies"][policy] = policy_report
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
