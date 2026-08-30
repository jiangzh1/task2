#!/usr/bin/env python3
"""将 hash-safe 数据、VAE latent 与正式语音路径汇总为可断点刷新的训练索引。"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path


SPLITS = ("train", "validation", "test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"splits": {}, "total_samples": 0, "latent_ready": 0, "audio_ready": 0, "fully_ready": 0}
    for split in SPLITS:
        counts = collections.Counter()
        destination = args.output_dir / f"training_index_{split}.jsonl"
        temporary = destination.with_suffix(".tmp.jsonl")
        with temporary.open("w", encoding="utf-8") as output:
            for line in (args.dataset_dir / f"spchconvsti_{split}.jsonl").open(encoding="utf-8"):
                row = json.loads(line)
                latent = args.latent_dir / f"{row['sticker']['image_sha256']}.pt"
                audio = args.audio_root / split / f"{row['sample_id']}.wav"
                latent_ready = latent.is_file() and latent.stat().st_size > 0
                audio_ready = audio.is_file() and audio.stat().st_size > 44
                item = {
                    "sample_id": row["sample_id"], "split": split,
                    "current_text": row["current"]["text"], "context": row["context"],
                    "sticker_emotion_official": row["derived_labels"]["sticker_emotion_official"],
                    "conflict_label": row["derived_labels"]["conflict_label"],
                    "image_sha256": row["sticker"]["image_sha256"],
                    "latent_path": str(latent), "audio_path": str(audio),
                    "latent_ready": latent_ready, "audio_ready": audio_ready,
                }
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts["samples"] += 1
                counts["latent_ready"] += latent_ready
                counts["audio_ready"] += audio_ready
                counts["fully_ready"] += latent_ready and audio_ready
        os.replace(temporary, destination)
        summary["splits"][split] = dict(counts)
        for key in ("samples", "latent_ready", "audio_ready", "fully_ready"):
            target = "total_samples" if key == "samples" else key
            summary[target] += counts[key]
    summary["image_side_complete"] = summary["latent_ready"] == summary["total_samples"]
    summary["training_ready"] = summary["fully_ready"] == summary["total_samples"]
    (args.output_dir / "training_index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["image_side_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
