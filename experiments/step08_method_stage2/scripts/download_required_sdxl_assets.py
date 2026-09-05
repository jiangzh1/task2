#!/usr/bin/env python3
"""以单文件重试方式补齐 SDXL 运行所需的 VAE 与 Latent-CLIP 资产。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


REQUIRED = (
    ("madebyollin/sdxl-vae-fp16-fix", "config.json", "sdxl_vae"),
    ("madebyollin/sdxl-vae-fp16-fix", "diffusion_pytorch_model.safetensors", "sdxl_vae"),
    ("wendlerc/latent-clip-b-4-512-plus-34b-80k", "Latent-ViT-B-4-512-plus.json", "latent_clip"),
    ("wendlerc/latent-clip-b-4-512-plus-34b-80k", "checkpoints/epoch_34.pt", "latent_clip"),
)


def download_with_retry(repo_id: str, filename: str, destination: Path, attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        try:
            print(f"[{attempt}/{attempts}] {repo_id}/{filename}", flush=True)
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=destination)
            if not Path(path).is_file() or Path(path).stat().st_size == 0:
                raise RuntimeError(f"下载后文件为空: {path}")
            print(f"completed {path}", flush=True)
            return
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(300, 15 * 2 ** (attempt - 1))
            print(f"download failed: {error!r}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    for repo_id, filename, directory in REQUIRED:
        download_with_retry(repo_id, filename, args.output_root / directory, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
