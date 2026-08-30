#!/usr/bin/env python3
"""Read-only inventory audit for the task2 experiment assets."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


JSONL_NAMES = (
    "StickerConv_Cleaned_Train.jsonl",
    "StickerConv_Cleaned_Vail.jsonl",
    "StickerConv_Cleaned_Test.jsonl",
    "StickerConv_Refined_Train.jsonl",
    "StickerConv_Refined_Vail.jsonl",
    "StickerConv_Refined_Test.jsonl",
)


def safe_command(args: list[str]) -> dict:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {
            "command": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # environment probe must not abort data audit
        return {"command": args, "error": repr(exc)}


def count_files(root: Path) -> dict:
    extensions = collections.Counter()
    total_bytes = 0
    file_count = 0
    if not root.exists():
        return {"exists": False, "file_count": 0, "total_bytes": 0, "extensions": {}}
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            file_count += 1
            extensions[path.suffix.lower() or "<none>"] += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    return {
        "exists": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "extensions": dict(extensions.most_common()),
    }


def normalize_sticker_path(data_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    value = raw_path.replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]
    return data_root / value


def audit_jsonl(path: Path, data_root: Path) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "lines": 0,
        "valid_json": 0,
        "invalid_json": 0,
        "duplicate_session_ids": 0,
        "duplicate_session_id_examples": [],
        "unique_session_ids": 0,
        "out_of_order_transitions": 0,
        "out_of_order_examples": [],
        "session_conflict": collections.Counter(),
        "turns": 0,
        "turns_with_sticker": 0,
        "missing_sticker_paths": 0,
        "text_sentiment": collections.Counter(),
        "sticker_emotion": collections.Counter(),
        "top_level_keys": collections.Counter(),
        "turn_keys": collections.Counter(),
        "error_examples": [],
    }
    if not path.exists():
        return result

    session_ids = set()
    ordered_session_ids = []
    for line_number, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        result["lines"] += 1
        try:
            item = json.loads(line)
        except Exception as exc:
            result["invalid_json"] += 1
            if len(result["error_examples"]) < 5:
                result["error_examples"].append({"line": line_number, "error": repr(exc)})
            continue

        result["valid_json"] += 1
        result["top_level_keys"].update(item.keys())
        session_id = item.get("session_id")
        if session_id in session_ids:
            result["duplicate_session_ids"] += 1
            if len(result["duplicate_session_id_examples"]) < 10:
                result["duplicate_session_id_examples"].append(session_id)
        else:
            session_ids.add(session_id)
        if ordered_session_ids and session_id is not None and session_id < ordered_session_ids[-1]:
            result["out_of_order_transitions"] += 1
            if len(result["out_of_order_examples"]) < 10:
                result["out_of_order_examples"].append([ordered_session_ids[-1], session_id])
        ordered_session_ids.append(session_id)
        result["session_conflict"][str(item.get("is_session_conflict"))] += 1

        for turn in item.get("dialogue", []):
            result["turns"] += 1
            result["turn_keys"].update(turn.keys())
            result["text_sentiment"][str(turn.get("text_sentiment"))] += 1
            emotion = turn.get("sticker_emotion", turn.get("emotion_label"))
            if emotion is not None:
                result["sticker_emotion"][str(emotion)] += 1
            sticker_path = turn.get("sticker_path")
            if sticker_path:
                result["turns_with_sticker"] += 1
                resolved = normalize_sticker_path(data_root, sticker_path)
                if resolved is None or not resolved.exists():
                    result["missing_sticker_paths"] += 1

    result["unique_session_ids"] = len(session_ids)
    result["_session_ids"] = sorted(str(value) for value in session_ids)
    for key in (
        "session_conflict",
        "text_sentiment",
        "sticker_emotion",
        "top_level_keys",
        "turn_keys",
    ):
        result[key] = dict(result[key].most_common())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_root = project_root / "data"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "audit_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only_scope": str(project_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "disk_usage": dict(zip(("total", "used", "free"), shutil.disk_usage(project_root))),
            "nvidia_smi": safe_command([
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ]),
        },
        "assets": {
            "project_top_level": sorted(p.name for p in project_root.iterdir()),
            "sticker_images": count_files(data_root / "SER_Dataset" / "Images"),
            "annotations": count_files(data_root / "SER_Dataset" / "Annotations"),
        },
        "jsonl": {},
        "cleaned_refined_comparisons": {},
    }

    for name in JSONL_NAMES:
        report["jsonl"][name] = audit_jsonl(data_root / name, data_root)

    for split in ("Train", "Vail", "Test"):
        clean_name = f"StickerConv_Cleaned_{split}.jsonl"
        refined_name = f"StickerConv_Refined_{split}.jsonl"
        clean_ids = set(report["jsonl"][clean_name].get("_session_ids", []))
        refined_ids = set(report["jsonl"][refined_name].get("_session_ids", []))
        report["cleaned_refined_comparisons"][split] = {
            "missing_from_refined_count": len(clean_ids - refined_ids),
            "missing_from_refined_examples": sorted(clean_ids - refined_ids)[:20],
            "unexpected_in_refined_count": len(refined_ids - clean_ids),
            "unexpected_in_refined_examples": sorted(refined_ids - clean_ids)[:20],
        }

    for item in report["jsonl"].values():
        item.pop("_session_ids", None)

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
