#!/usr/bin/env python3
"""Build a stable, auditable manifest without modifying source data."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path


SPLITS = {
    "train": ("StickerConv_Cleaned_Train.jsonl", "StickerConv_Refined_Train.jsonl"),
    "validation": ("StickerConv_Cleaned_Vail.jsonl", "StickerConv_Refined_Vail.jsonl"),
    "test": ("StickerConv_Cleaned_Test.jsonl", "StickerConv_Refined_Test.jsonl"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            item["_source_line"] = line_number
            rows.append(item)
    return rows


def target_turn(item: dict) -> tuple[int, dict]:
    dialogue = item.get("dialogue", [])
    for index in range(len(dialogue) - 1, -1, -1):
        if dialogue[index].get("sticker_path"):
            return index, dialogue[index]
    raise ValueError(f"session {item.get('session_id')} has no sticker target")


def fingerprint(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def refined_index(rows: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        index[str(row.get("session_id"))].append(row)
    return index


def build_split(split: str, data_root: Path, output_root: Path) -> tuple[list[dict], dict]:
    cleaned_name, refined_name = SPLITS[split]
    cleaned_path = data_root / cleaned_name
    refined_path = data_root / refined_name
    cleaned = read_jsonl(cleaned_path)
    refined = read_jsonl(refined_path)
    refined_by_id = refined_index(refined)

    records = []
    summary = {
        "source": {
            "cleaned": {"path": str(cleaned_path), "sha256": sha256_file(cleaned_path)},
            "refined": {"path": str(refined_path), "sha256": sha256_file(refined_path)},
        },
        "samples": 0,
        "refined_status": collections.Counter(),
        "target_emotion": collections.Counter(),
        "cleaned_conflict": collections.Counter(),
        "refined_conflict": collections.Counter(),
        "missing_target_images": 0,
        "history_turns": [],
    }

    for clean in sorted(cleaned, key=lambda row: str(row.get("session_id"))):
        session_id = str(clean.get("session_id"))
        target_index, current = target_turn(clean)
        candidates = refined_by_id.get(session_id, [])
        refined_status = "missing"
        refined_target = None
        if len(candidates) == 1:
            _, refined_target = target_turn(candidates[0])
            if "ERROR" in str(refined_target.get("text_sentiment")):
                refined_status = "llm_error"
            else:
                refined_status = "available"
        elif len(candidates) > 1:
            refined_status = "duplicate"

        relative_image = str(current.get("sticker_path"))
        image_value = relative_image[2:] if relative_image.startswith("./") else relative_image
        image_path = data_root / image_value
        if not image_path.exists():
            summary["missing_target_images"] += 1

        history = [
            {
                "turn": turn.get("turn", index),
                "role": turn.get("role"),
                "text": turn.get("text"),
                "sticker_path": turn.get("sticker_path"),
            }
            for index, turn in enumerate(clean.get("dialogue", [])[:target_index])
        ]
        emotion = current.get("emotion_label")
        clean_conflict = current.get("is_conflict")
        refined_conflict = refined_target.get("is_conflict") if refined_target else None

        record = {
            "schema_version": "1.0.0",
            "sample_id": f"{split}:{session_id}",
            "split": split,
            "session_id": session_id,
            "source_line_cleaned": clean.get("_source_line"),
            "source_line_refined": refined_target and candidates[0].get("_source_line"),
            "current": {
                "turn_index": target_index,
                "role": current.get("role"),
                "text": current.get("text"),
                "image_path": relative_image,
                "description": current.get("description"),
            },
            "history": history,
            "labels": {
                "target_emotion_raw": emotion,
                "text_sentiment_vader": current.get("text_sentiment"),
                "conflict_vader": clean_conflict,
                "text_emotion_llm": refined_target.get("text_sentiment") if refined_target else None,
                "conflict_llm": refined_conflict,
                "refined_status": refined_status,
                "verified_conflict": None,
                "verified_emotion": None,
            },
            "audio": {
                "path": None,
                "speaker_id": None,
                "synthesis_model": None,
                "sample_rate": None,
                "status": "not_created",
            },
            "fingerprints": {
                "dialogue_text": fingerprint([turn.get("text") for turn in clean.get("dialogue", [])]),
                "current_text": fingerprint(current.get("text")),
                "target_image_path": fingerprint(relative_image),
                "current_text_and_image": fingerprint([current.get("text"), relative_image]),
            },
        }
        records.append(record)
        summary["refined_status"][refined_status] += 1
        summary["target_emotion"][str(emotion)] += 1
        summary["cleaned_conflict"][str(clean_conflict)] += 1
        summary["refined_conflict"][str(refined_conflict)] += 1
        summary["history_turns"].append(len(history))

    output_path = output_root / f"manifest_{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["samples"] = len(records)
    for key in ("refined_status", "target_emotion", "cleaned_conflict", "refined_conflict"):
        summary[key] = dict(summary[key].most_common())
    lengths = summary.pop("history_turns")
    summary["history_length"] = {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": sum(lengths) / len(lengths) if lengths else 0,
    }
    return records, summary


def overlap_summary(all_records: dict[str, list[dict]]) -> dict:
    fields = ("dialogue_text", "current_text", "target_image_path", "current_text_and_image")
    result = {}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        pair_key = f"{left}_vs_{right}"
        result[pair_key] = {}
        for field in fields:
            left_values = {record["fingerprints"][field] for record in all_records[left]}
            right_values = {record["fingerprints"][field] for record in all_records[right]}
            result[pair_key][field] = len(left_values & right_values)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.project_root.resolve() / "data"
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = {}
    summaries = {}
    for split in SPLITS:
        records[split], summaries[split] = build_split(split, data_root, output_root)

    report = {
        "schema_version": "1.0.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "splits": summaries,
        "cross_split_overlap": overlap_summary(records),
        "label_policy": {
            "cleaned_vader": "weak_candidate_only",
            "refined_llm": "weak_candidate_only",
            "verified_fields": "null_until_manual_or_validated_annotation",
        },
    }
    (output_root / "manifest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_root / "manifest_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
