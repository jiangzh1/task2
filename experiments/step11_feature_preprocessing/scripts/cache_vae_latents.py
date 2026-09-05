#!/usr/bin/env python3
"""将目标 sticker 确定性编码为扩散模型 VAE latent，支持断点续跑。

SDXL 工作分支默认使用 SDXL Base 的 VAE 及其配置中的 scaling_factor。
"""

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
    parser.add_argument("--model", type=Path, default=None, help="包含 vae 子目录的本地 diffusers 模型目录（兼容旧用法）")
    parser.add_argument("--vae", type=Path, default=None, help="独立的 SDXL VAE 目录；与 --model 二选一")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--resize-mode", choices=["pad", "center_crop"], default="pad")
    parser.add_argument("--latent-scale", type=float, default=None, help="默认读取 VAE config.scaling_factor")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if (args.model is None) == (args.vae is None):
        raise ValueError("必须且只能指定 --model 或 --vae")
    vae_source = args.vae if args.vae is not None else args.model
    vae_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if args.vae is None:
        vae_kwargs["subfolder"] = "vae"
    vae = AutoencoderKL.from_pretrained(str(vae_source), **vae_kwargs).to(args.device).eval()
    vae.requires_grad_(False)
    latent_scale = args.latent_scale if args.latent_scale is not None else getattr(vae.config, "scaling_factor", None)
    if latent_scale is None:
        raise ValueError("VAE 未声明 scaling_factor，请通过 --latent-scale 显式指定")
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
                latent = (latent * latent_scale).squeeze(0).cpu()
            temporary = destination.with_suffix(".tmp.pt")
            torch.save({
                "latent": latent,
                "sha256": row["sha256"],
                "source_image": row["image_path"],
                "image_size": args.image_size,
                "resize_mode": args.resize_mode,
                "latent_scale": latent_scale,
                "vae_scaling_factor": getattr(vae.config, "scaling_factor", None),
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
