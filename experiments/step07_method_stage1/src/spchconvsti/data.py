"""正式版本 B 的数据契约与清单读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


VALID_EMOTIONS = {"Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"}
VALID_CONFLICT = {"Consistent", "Conflict"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    current_text: str
    context: tuple[dict, ...]
    sticker_emotion: str
    conflict_label: str
    audio_path: Path | None


def iter_records(dataset_jsonl: Path, audio_root: Path | None = None) -> Iterator[SampleRecord]:
    with dataset_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            labels = row["derived_labels"]
            emotion = labels["sticker_emotion_official"]
            conflict = labels["conflict_label"]
            if emotion not in VALID_EMOTIONS:
                raise ValueError(f"第 {line_number} 行存在非法七类情感: {emotion}")
            if conflict not in VALID_CONFLICT:
                raise ValueError(f"第 {line_number} 行存在非法冲突标签: {conflict}")
            split = row["split"]
            audio_path = audio_root / split / f"{row['sample_id']}.wav" if audio_root else None
            yield SampleRecord(
                sample_id=row["sample_id"],
                split=split,
                current_text=row["current"]["text"],
                context=tuple(row["context"]),
                sticker_emotion=emotion,
                conflict_label=conflict,
                audio_path=audio_path,
            )
