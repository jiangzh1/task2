#!/usr/bin/env python3
"""Select a deterministic, emotion-covered 100-sample TTS technical pilot."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


QUOTAS = {
    "Happiness": 15,
    "Neutral": 15,
    "Sadness": 15,
    "Anger": 15,
    "Surprise": 14,
    "Fear": 13,
    "Disgust": 13,
}


def stable_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            labels = item["labels"]
            if labels.get("refined_status") != "available":
                continue
            emotion = labels.get("target_emotion_raw")
            if emotion in QUOTAS:
                groups[emotion].append(item)

    selected = []
    shortfalls = {}
    for emotion, quota in QUOTAS.items():
        candidates = sorted(groups[emotion], key=lambda x: stable_key(x["sample_id"], args.seed))
        conflict = [x for x in candidates if x["labels"].get("conflict_llm") == 1]
        non_conflict = [x for x in candidates if x["labels"].get("conflict_llm") == 0]
        conflict_target = min(quota // 2, len(conflict))
        chosen = conflict[:conflict_target]
        chosen_ids = {x["sample_id"] for x in chosen}
        for item in non_conflict + conflict[conflict_target:]:
            if len(chosen) >= quota:
                break
            if item["sample_id"] not in chosen_ids:
                chosen.append(item)
                chosen_ids.add(item["sample_id"])
        if len(chosen) < quota:
            shortfalls[emotion] = quota - len(chosen)
        selected.extend(chosen)

    selected = sorted(selected, key=lambda x: (x["labels"]["target_emotion_raw"], x["sample_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(selected):
            pilot = {
                "pilot_id": f"PILOT_{index:03d}",
                "sample_id": item["sample_id"],
                "text": item["current"]["text"],
                "target_emotion_candidate": item["labels"]["target_emotion_raw"],
                "conflict_llm_candidate": item["labels"]["conflict_llm"],
                "manual_text_ok": None,
                "manual_emotion_ok": None,
                "manual_notes": None,
                "tts": {
                    "status": "pending_manual_review",
                    "audio_path": None,
                    "speaker_reference": None,
                    "model": None,
                },
            }
            handle.write(json.dumps(pilot, ensure_ascii=False) + "\n")

    emotion_counts = collections.Counter(x["labels"]["target_emotion_raw"] for x in selected)
    conflict_counts = collections.Counter(str(x["labels"]["conflict_llm"]) for x in selected)
    summary = {
        "seed": args.seed,
        "requested": sum(QUOTAS.values()),
        "selected": len(selected),
        "quotas": QUOTAS,
        "emotion_counts": dict(emotion_counts),
        "conflict_llm_candidate": dict(conflict_counts),
        "shortfalls": shortfalls,
        "policy": "technical coverage sample; not representative and not final ground truth",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
