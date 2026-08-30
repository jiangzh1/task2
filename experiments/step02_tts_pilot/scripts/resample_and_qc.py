#!/usr/bin/env python3
"""将 EmoVoice 的 22.05 kHz 输出转为论文统一使用的 16 kHz，并做基础质检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sorted(args.input_dir.glob("*.wav")):
        audio, sr = sf.read(source, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        tensor = torch.from_numpy(np.asarray(audio)).unsqueeze(0)
        if sr != 16000:
            tensor = AF.resample(tensor, sr, 16000)
        result = tensor.squeeze(0).numpy()
        destination = args.output_dir / source.name
        sf.write(destination, result, 16000, subtype="PCM_16")
        duration = len(result) / 16000
        peak = float(np.max(np.abs(result))) if len(result) else 0.0
        rms = float(np.sqrt(np.mean(np.square(result)))) if len(result) else 0.0
        rows.append({
            "file": source.name,
            "input_sample_rate": sr,
            "output_sample_rate": 16000,
            "duration_seconds": round(duration, 4),
            "rms": round(rms, 7),
            "peak": round(peak, 7),
            "basic_qc_pass": 0.5 <= duration <= 30 and rms >= 0.003 and peak <= 1.0,
        })
    report = {
        "generated_files": len(rows),
        "basic_qc_passed": sum(row["basic_qc_pass"] for row in rows),
        "all_basic_qc_passed": bool(rows) and all(row["basic_qc_pass"] for row in rows),
        "files": rows,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
