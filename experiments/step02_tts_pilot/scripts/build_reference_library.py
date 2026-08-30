#!/usr/bin/env python3
"""Extract and quality-check neutral reference speech from RAVDESS and CREMA-D."""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def quality(audio: np.ndarray, sample_rate: int) -> dict:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sample_rate if sample_rate else 0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    silence = float(np.mean(np.abs(audio) < 0.005)) if len(audio) else 1.0
    return {"sample_rate": sample_rate, "channels": 1, "duration_seconds": round(duration, 4), "rms": round(rms, 7), "peak": round(peak, 7), "silence_ratio": round(silence, 6)}


def accepted(metrics: dict) -> bool:
    return 1.2 <= metrics["duration_seconds"] <= 12.0 and metrics["rms"] >= 0.008 and metrics["peak"] <= 1.0 and metrics["silence_ratio"] <= 0.65


def preprocess_reference(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """转单声道、裁剪首尾留白并进行保守峰值归一化。

    RAVDESS 原始文件带较长首尾留白且整体电平偏低；直接按整段 RMS/静音率
    会把有效语音误判为低质量。这里只裁首尾，不移除句中自然停顿。
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio):
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    threshold = max(0.001, peak * 0.03)
    active = np.flatnonzero(np.abs(audio) >= threshold)
    if len(active):
        pad = int(0.15 * sample_rate)
        start = max(0, int(active[0]) - pad)
        end = min(len(audio), int(active[-1]) + pad + 1)
        audio = audio[start:end]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio * min(0.85 / peak, 12.0)
    return audio


def save_wav(destination: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="PCM_16")


def ravdess(zip_path: Path, output: Path) -> list[dict]:
    result = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".wav"):
                continue
            filename = Path(name).name
            fields = filename.removesuffix(".wav").split("-")
            if len(fields) != 7 or fields[2] != "01":  # RAVDESS emotion 01 = neutral
                continue
            actor = fields[6]
            audio, sr = sf.read(io.BytesIO(archive.read(name)), dtype="float32", always_2d=False)
            original_metrics = quality(audio, sr)
            audio = preprocess_reference(audio, sr)
            metrics = quality(audio, sr)
            destination = output / "RAVDESS" / f"actor_{actor}" / filename
            save_wav(destination, audio, sr)
            result.append({"source": "RAVDESS", "speaker_id": f"ravdess_{actor}", "gender": "male" if int(actor) % 2 else "female", "emotion": "neutral", "path": str(destination), "accepted": accepted(metrics), "original_metrics": original_metrics, **metrics})
    return result


def crema(paths: list[Path], output: Path) -> list[dict]:
    result = []
    for path in paths:
        table = pq.read_table(path, columns=["audio", "actor_id", "emotion_code", "source_file"])
        for row in table.to_pylist():
            if row.get("emotion_code") != "NEU":
                continue
            audio_obj = row["audio"]
            raw = audio_obj.get("bytes") if isinstance(audio_obj, dict) else None
            if not raw:
                continue
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            original_metrics = quality(audio, sr)
            audio = preprocess_reference(audio, sr)
            metrics = quality(audio, sr)
            actor = str(row["actor_id"])
            destination = output / "CREMA-D" / f"actor_{actor}" / row["source_file"]
            save_wav(destination, audio, sr)
            result.append({"source": "CREMA-D", "speaker_id": f"cremad_{actor}", "gender": "unknown", "emotion": "neutral", "path": str(destination), "accepted": accepted(metrics), "original_metrics": original_metrics, **metrics})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess-zip", type=Path, required=True)
    parser.add_argument("--crema-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = ravdess(args.ravdess_zip, args.output_dir)
    crema_paths = sorted(path for path in args.crema_dir.glob("*.parquet") if path.stat().st_size > 0)
    if crema_paths:
        records.extend(crema(crema_paths, args.output_dir))
    manifest = args.output_dir / "reference_library.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    accepted_rows = [row for row in records if row["accepted"]]
    speakers = collections.Counter(row["source"] for row in accepted_rows)
    summary = {
        "total_neutral_files": len(records),
        "accepted_files": len(accepted_rows),
        "rejected_files": len(records) - len(accepted_rows),
        "accepted_by_source": dict(speakers),
        "unique_accepted_speakers": len({row["speaker_id"] for row in accepted_rows}),
        "crema_status": "included" if crema_paths else "pending_download",
        "preprocessing": "mono; DC removal; trim leading/trailing silence with 150 ms margin; conservative peak normalization capped at 12x",
        "quality_rule": "post-preprocessing duration 1.2-12s; RMS >= 0.008; peak <= 1.0; silence ratio <= 0.65",
    }
    (args.output_dir / "reference_library_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
