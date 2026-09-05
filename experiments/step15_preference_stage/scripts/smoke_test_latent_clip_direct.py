#!/usr/bin/env python3
"""验证公开 Latent-CLIP 对 SDXL latent 的直接、可微编码。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "step15_preference_stage/src"))
from spchconvsti_preference.latent_clip import load_official_latent_clip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    encoder = load_official_latent_clip(
        source_dir=args.source, checkpoint_path=args.checkpoint, vae_path=args.vae,
        device=device, precision="fp16" if dtype == torch.float16 else "fp32",
    )
    latent = torch.randn(1, 4, 64, 64, device=device, dtype=dtype, requires_grad=True)
    embedding = encoder(latent)
    embedding.float().square().mean().backward()
    checks = {
        "direct_latent_input": list(latent.shape) == [1, 4, 64, 64],
        "embedding_shape": list(embedding.shape) == [1, 640],
        "embedding_finite": bool(torch.isfinite(embedding).all()),
        "latent_gradient": latent.grad is not None and bool(torch.isfinite(latent.grad).all()),
        "frozen_encoder": all(not parameter.requires_grad for parameter in encoder.parameters()),
    }
    result = {
        "passed": all(checks.values()), "checks": checks,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2) if device.type == "cuda" else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
