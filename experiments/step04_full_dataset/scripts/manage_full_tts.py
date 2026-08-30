#!/usr/bin/env python3
"""生成队列、增量转码、拼接和状态统计。仅依赖 TTS 虚拟环境已有包。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS units (key TEXT PRIMARY KEY, payload TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending', last_error TEXT)")
    return db


def load_manifest(db: sqlite3.Connection, manifest: Path) -> None:
    with manifest.open(encoding="utf-8") as handle:
        db.executemany(
            "INSERT OR IGNORE INTO units(key,payload) VALUES(?,?)",
            ((json.loads(line)["key"], line.rstrip("\n")) for line in handle if line.strip()),
        )
    db.commit()


def resample_file(source: Path, destination: Path) -> dict:
    audio, sr = sf.read(source, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tensor = torch.from_numpy(np.asarray(audio)).unsqueeze(0)
    if sr != 16000:
        tensor = AF.resample(tensor, sr, 16000)
    result = tensor.squeeze(0).numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.wav")
    sf.write(temp, result, 16000, subtype="PCM_16")
    temp.replace(destination)
    duration = len(result) / 16000
    rms = float(np.sqrt(np.mean(np.square(result)))) if len(result) else 0.0
    peak = float(np.max(np.abs(result))) if len(result) else 0.0
    return {"duration_seconds": round(duration, 4), "rms": round(rms, 7), "peak": round(peak, 7)}


def reconcile(db: sqlite3.Connection, raw_dir: Path, segment_dir: Path) -> dict:
    converted = 0
    for source in raw_dir.glob("*.wav") if raw_dir.exists() else []:
        destination = segment_dir / source.name
        if not destination.exists() or destination.stat().st_size == 0:
            resample_file(source, destination)
            converted += 1
    completed = {path.stem for path in segment_dir.glob("*.wav") if path.stat().st_size > 44}
    db.executemany("UPDATE units SET status='succeeded',last_error=NULL WHERE key=?", ((key,) for key in completed))
    db.commit()
    return {"newly_converted": converted, "completed_segments": len(completed)}


def make_batch(db: sqlite3.Connection, output: Path, limit: int, max_attempts: int) -> int:
    rows = db.execute(
        "SELECT key,payload FROM units WHERE status!='succeeded' AND attempts<? ORDER BY rowid LIMIT ?",
        (max_attempts, limit),
    ).fetchall()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(payload + "\n" for _, payload in rows), encoding="utf-8")
    return len(rows)


def mark_missing_attempt(db: sqlite3.Connection, batch: Path, segment_dir: Path, message: str) -> None:
    for line in batch.open(encoding="utf-8"):
        if not line.strip():
            continue
        key = json.loads(line)["key"]
        if not (segment_dir / f"{key}.wav").exists():
            db.execute("UPDATE units SET attempts=attempts+1,status='pending',last_error=? WHERE key=?", (message, key))
    db.commit()


def assemble(index_path: Path, segment_dir: Path, final_dir: Path, silence_ms: int) -> dict:
    final_dir.mkdir(parents=True, exist_ok=True)
    assembled = 0
    for line in index_path.open(encoding="utf-8"):
        item = json.loads(line)
        destination = final_dir / item["split"] / f"{item['sample_id']}.wav"
        if destination.exists() and destination.stat().st_size > 44:
            continue
        sources = [segment_dir / f"{key}.wav" for key in item["segment_keys"]]
        if not all(path.exists() and path.stat().st_size > 44 for path in sources):
            continue
        pieces = []
        for position, source in enumerate(sources):
            audio, sr = sf.read(source, dtype="float32", always_2d=False)
            if sr != 16000:
                raise ValueError(f"unexpected sample rate {sr}: {source}")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            pieces.append(audio)
            if position + 1 < len(sources):
                pieces.append(np.zeros(int(16000 * silence_ms / 1000), dtype=np.float32))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".tmp.wav")
        sf.write(temp, np.concatenate(pieces), 16000, subtype="PCM_16")
        temp.replace(destination)
        assembled += 1
    total = sum(1 for _ in final_dir.glob("*/*.wav"))
    return {"newly_assembled": assembled, "completed_samples": total}


def summary(db: sqlite3.Connection, index_path: Path, final_dir: Path, max_attempts: int) -> dict:
    total_segments = db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    succeeded = db.execute("SELECT COUNT(*) FROM units WHERE status='succeeded'").fetchone()[0]
    retryable = db.execute("SELECT COUNT(*) FROM units WHERE status!='succeeded' AND attempts<?", (max_attempts,)).fetchone()[0]
    exhausted = db.execute("SELECT COUNT(*) FROM units WHERE status!='succeeded' AND attempts>=?", (max_attempts,)).fetchone()[0]
    total_samples = sum(1 for line in index_path.open(encoding="utf-8") if line.strip())
    completed_samples = sum(1 for _ in final_dir.glob("*/*.wav")) if final_dir.exists() else 0
    return {
        "total_segments": total_segments,
        "succeeded_segments": succeeded,
        "retryable_segments": retryable,
        "exhausted_segments": exhausted,
        "total_samples": total_samples,
        "completed_samples": completed_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "reconcile", "make-batch", "mark-missing", "assemble", "summary"])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--segment-dir", type=Path)
    parser.add_argument("--final-dir", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--silence-ms", type=int, default=180)
    parser.add_argument("--message", default="模型未产出音频")
    parser.add_argument("--status-json", type=Path)
    args = parser.parse_args()
    db = connect(args.db)
    if args.command == "init":
        load_manifest(db, args.manifest)
        result = {"initialized": db.execute("SELECT COUNT(*) FROM units").fetchone()[0]}
    elif args.command == "reconcile":
        result = reconcile(db, args.raw_dir, args.segment_dir)
    elif args.command == "make-batch":
        result = {"batch_size": make_batch(db, args.batch, args.limit, args.max_attempts)}
    elif args.command == "mark-missing":
        mark_missing_attempt(db, args.batch, args.segment_dir, args.message)
        result = {"marked": True}
    elif args.command == "assemble":
        result = assemble(args.index, args.segment_dir, args.final_dir, args.silence_ms)
    else:
        result = summary(db, args.index, args.final_dir, args.max_attempts)
    if args.status_json:
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        temp = args.status_json.with_suffix(".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.status_json)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
