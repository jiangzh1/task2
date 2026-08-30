#!/usr/bin/env python3
"""Summarize two blind full-corpus text-polarity runs without modifying them."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")


def rows(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                result[item["sample_id"]] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"models": {}, "splits": {}, "total": {"common_success": 0, "agreement": 0, "disagreement": 0}}
    for split in SPLITS:
        gemma = rows(args.artifacts / f"text_polarity_gemma3_{split}.jsonl")
        qwen = rows(args.artifacts / f"text_polarity_qwen3_{split}.jsonl")
        for name, source in (("gemma3", gemma), ("qwen3", qwen)):
            model = report["models"].setdefault(name, {"records": 0, "ok": 0, "error": 0, "polarity": collections.Counter()})
            model["records"] += len(source)
            model["ok"] += sum(item.get("status") == "ok" for item in source.values())
            model["error"] += sum(item.get("status") != "ok" for item in source.values())
            model["polarity"].update(
                item["text_inference"]["parsed"]["polarity"] for item in source.values() if item.get("status") == "ok"
            )
        common = []
        for sample_id in sorted(set(gemma) & set(qwen)):
            left, right = gemma[sample_id], qwen[sample_id]
            if left.get("status") == "ok" and right.get("status") == "ok":
                common.append((sample_id, left["text_inference"]["parsed"]["polarity"], right["text_inference"]["parsed"]["polarity"]))
        agreement = sum(left == right for _, left, right in common)
        split_report = {"common_success": len(common), "agreement": agreement, "disagreement": len(common) - agreement}
        report["splits"][split] = split_report
        for key in split_report:
            report["total"][key] += split_report[key]
    for model in report["models"].values():
        model["polarity"] = dict(sorted(model["polarity"].items()))
    report["total"]["agreement_rate"] = round(report["total"]["agreement"] / report["total"]["common_success"], 6)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
