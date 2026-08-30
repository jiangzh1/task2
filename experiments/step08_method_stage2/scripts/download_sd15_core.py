#!/usr/bin/env python3
"""可恢复下载 Stable Diffusion v1.5 核心组件，不含 safety checker/ONNX/Flax。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.output_dir,
        allow_patterns=[
            "model_index.json",
            "scheduler/*.json",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder/pytorch_model.bin",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    )
    required = [
        args.output_dir / "model_index.json",
        args.output_dir / "unet" / "config.json",
        args.output_dir / "unet" / "diffusion_pytorch_model.safetensors",
        args.output_dir / "vae" / "config.json",
        args.output_dir / "vae" / "diffusion_pytorch_model.safetensors",
    ]
    report = {
        "repo": args.repo,
        "snapshot_path": str(path),
        "required_files_present": all(item.exists() for item in required),
        "required_files": [{"path": str(item), "exists": item.exists(), "bytes": item.stat().st_size if item.exists() else 0} for item in required],
    }
    (args.output_dir / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["required_files_present"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
