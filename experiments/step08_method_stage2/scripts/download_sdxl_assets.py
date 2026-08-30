#!/usr/bin/env python3
"""下载 SDXL/Latent-CLIP 运行资产；资产目录不纳入 Git。"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ASSETS = {
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_vae": "madebyollin/sdxl-vae-fp16-fix",
    "latent_clip": "wendlerc/latent-clip-b-4-512-plus-34b-80k",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only", choices=[*ASSETS, "all"], default="all")
    args = parser.parse_args()
    selected = ASSETS.items() if args.only == "all" else [(args.only, ASSETS[args.only])]
    for name, repo_id in selected:
        destination = args.output_root / name
        print(f"downloading {repo_id} -> {destination}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=destination, resume_download=True)
        print(f"completed {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
