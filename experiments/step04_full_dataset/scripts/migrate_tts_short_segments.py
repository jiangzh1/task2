"""保留既有成功音频，仅将未成功的长文本分段迁移为更短的可生成分段。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, max_words: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences or [text.strip()]:
        words = sentence.split()
        while len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if words and current and current_words + len(words) > max_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        if words:
            current.append(" ".join(words))
            current_words += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text.strip()]


def link_if_missing(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    os.link(source, destination)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-artifacts", type=Path, required=True)
    parser.add_argument("--old-audio", type=Path, required=True)
    parser.add_argument("--new-artifacts", type=Path, required=True)
    parser.add_argument("--new-audio", type=Path, required=True)
    parser.add_argument("--max-words", type=int, default=20)
    args = parser.parse_args()

    old_manifest = load_jsonl(args.old_artifacts / "full_tts_segments.jsonl")
    old_index = load_jsonl(args.old_artifacts / "full_tts_sample_index.jsonl")
    database = sqlite3.connect(args.old_artifacts / "state.sqlite3")
    status = {key: value for key, value in database.execute("SELECT key,status FROM units")}
    old_records = {item["key"]: item for item in old_manifest}
    old_segments = args.old_audio / "segments_16k"
    new_segments = args.new_audio / "segments_16k"

    mapping: dict[str, list[str]] = {}
    new_records: list[dict] = []
    retained = resplit = 0
    for old_key, item in old_records.items():
        old_wav = old_segments / f"{old_key}.wav"
        if status.get(old_key) == "succeeded" and old_wav.is_file() and old_wav.stat().st_size > 44:
            link_if_missing(old_wav, new_segments / old_wav.name)
            mapping[old_key] = [old_key]
            new_records.append(item)
            retained += 1
            continue
        keys = []
        for position, chunk in enumerate(split_text(item["target_text"], args.max_words)):
            key = f"{old_key}__r{position:03d}"
            child = {**item, "key": key, "source_text": chunk, "target_text": chunk}
            child["metadata"] = {**item["metadata"], "parent_key": old_key, "repair_index": position}
            new_records.append(child)
            keys.append(key)
        mapping[old_key] = keys
        resplit += 1

    new_index = []
    for item in old_index:
        segment_keys = [key for old_key in item["segment_keys"] for key in mapping[old_key]]
        new_index.append({**item, "segment_keys": segment_keys, "tts_segmentation": {"version": "v2_short_pending", "max_words": args.max_words}})

    args.new_artifacts.mkdir(parents=True, exist_ok=True)
    args.new_audio.mkdir(parents=True, exist_ok=True)
    with (args.new_artifacts / "full_tts_segments.jsonl").open("w", encoding="utf-8") as handle:
        for item in new_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.new_artifacts / "full_tts_sample_index.jsonl").open("w", encoding="utf-8") as handle:
        for item in new_index:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 完整样本已由旧分段拼接完成，可硬链接复用；不移动原文件。
    retained_final = 0
    old_final = args.old_audio / "final_16k"
    for source in old_final.glob("*/*.wav"):
        link_if_missing(source, args.new_audio / "final_16k" / source.parent.name / source.name)
        retained_final += 1

    lengths = [len(item["target_text"].split()) for item in new_records]
    summary = {
        "schema_version": "2.0.0",
        "purpose": "仅重新切分旧队列中未成功的语音分段；成功音频以硬链接复用且原文件保持不动。",
        "max_words_for_resplit_pending": args.max_words,
        "old_segments": len(old_records),
        "new_segments": len(new_records),
        "retained_successful_segments": retained,
        "resplit_unsuccessful_parent_segments": resplit,
        "retained_complete_samples": retained_final,
        "new_length_statistics": {"min": min(lengths), "max": max(lengths), "over_max": sum(length > args.max_words for length in lengths)},
        "new_emotion_counts": dict(Counter(item["metadata"]["sticker_emotion_official"] for item in new_records)),
    }
    (args.new_artifacts / "migration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
