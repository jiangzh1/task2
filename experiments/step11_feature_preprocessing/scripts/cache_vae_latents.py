#!/usr/bin/env python3
"""将目标 sticker 确定性编码为 SD1.5 VAE latent，支持断点续跑。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps


def prepare_image(path: Path, size: int, resize_mode: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        rgb = background.convert("RGB")
        if resize_mode == "pad":
            contained = ImageOps.contain(rgb, (size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            offset = ((size - contained.width) // 2, (size - contained.height) // 2)
            canvas.paste(contained, offset)
            rgb = canvas
        else:
            rgb = ImageOps.fit(rgb, (size, size), method=Image.Resampling.LANCZOS)
        data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        tensor = data.reshape(size, size, 3).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True, help="本地 diffusers SD1.5 目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--resize-mode", choices=["pad", "center_crop"], default="pad")
    parser.add_argument("--latent-scale", type=float, default=0.18215)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(args.device).eval()
    vae.requires_grad_(False)
    completed = skipped = failed = 0
    failures = []
    for line in args.manifest.open(encoding="utf-8"):
        row = json.loads(line)
        destination = args.output_dir / f"{row['sha256']}.pt"
        if destination.exists() and destination.stat().st_size > 0:
            skipped += 1
            continue
        try:
            image = prepare_image(Path(row["image_path"]), args.image_size, args.resize_mode)
            with torch.inference_mode():
                latent = vae.encode(image.unsqueeze(0).to(args.device, dtype=dtype)).latent_dist.mode()
                latent = (latent * args.latent_scale).squeeze(0).cpu()
            temporary = destination.with_suffix(".tmp.pt")
            torch.save({
                "latent": latent,
                "sha256": row["sha256"],
                "source_image": row["image_path"],
                "image_size": args.image_size,
                "resize_mode": args.resize_mode,
                "latent_scale": args.latent_scale,
            }, temporary)
            os.replace(temporary, destination)
            completed += 1
        except Exception as error:
            failed += 1
            failures.append({"image_path": row["image_path"], "error": repr(error)})
    report = {"completed": completed, "skipped": skipped, "failed": failed, "failures": failures}
    (args.output_dir / "cache_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
