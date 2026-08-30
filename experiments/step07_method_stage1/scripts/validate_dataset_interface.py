#!/usr/bin/env python3
"""验证三个正式划分能否无损进入方法数据接口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spchconvsti.data import iter_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    filenames = {
        "train": "spchconvsti_train.jsonl",
        "validation": "spchconvsti_validation.jsonl",
        "test": "spchconvsti_test.jsonl",
    }
    split_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    conflict_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    duplicate_ids = 0
    audio_present = 0
    for expected_split, filename in filenames.items():
        for record in iter_records(args.dataset_dir / filename, args.audio_root):
            if record.split != expected_split:
                raise ValueError(f"{record.sample_id}: split={record.split}, 文件={expected_split}")
            if record.sample_id in sample_ids:
                duplicate_ids += 1
            sample_ids.add(record.sample_id)
            split_counts[record.split] += 1
            emotion_counts[record.sticker_emotion] += 1
            conflict_counts[record.conflict_label] += 1
            audio_present += int(record.audio_path is not None and record.audio_path.exists())
    report = {
        "status": "passed" if duplicate_ids == 0 and len(sample_ids) == 12930 else "failed",
        "samples": len(sample_ids),
        "duplicate_sample_ids": duplicate_ids,
        "split_counts": dict(split_counts),
        "emotion_counts": dict(emotion_counts),
        "conflict_counts": dict(conflict_counts),
        "audio_present_at_validation_time": audio_present,
        "audio_missing_at_validation_time": len(sample_ids) - audio_present,
        "note": "音频仍在后台生成；缺失数不作为本次数据接口失败条件。",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
