#!/usr/bin/env python3
"""基于同一文本和同一说话人构建愤怒提示强度对照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPTS = (
    ("anger_simple", "angry"),
    ("anger_strong", "strong anger, tense and forceful"),
    ("anger_furious", "furious, intense anger, sharp emphasis and raised energy"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.source.read_text(encoding="utf-8").splitlines()[0])
    rows = []
    for suffix, prompt in PROMPTS:
        row = json.loads(json.dumps(base))
        row["key"] = f"{base['key']}__{suffix}"
        row["emotion"] = "anger"
        row["emotion_text_prompt"] = prompt
        row.setdefault("metadata", {})["probe_type"] = "anger_prompt_strength"
        row["metadata"]["probe_prompt"] = prompt
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(rows), "prompts": [p for _, p in PROMPTS]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
