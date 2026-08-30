#!/usr/bin/env python3
"""在完整 SD1.5 U-Net 权重上验证适配器恒等性、梯度和显存。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

ROOT = Path("/data/jzh/2026/task2/experiments")
sys.path.insert(0, str(ROOT / "step07_method_stage1/src"))
sys.path.insert(0, str(ROOT / "step08_method_stage2/src"))
from spchconvsti_diffusion.unet_adapter import ConflictAwareUNetAdapter


def main() -> int:
    model_dir = ROOT / "step08_method_stage2/assets/stable-diffusion-v1-5"
    artifact = ROOT / "step08_method_stage2/artifacts/sd15_full_smoke.json"
    if torch.cuda.device_count() < 2:
        raise RuntimeError("完整 SD1.5 冒烟测试要求第二张空闲 GPU")
    device = torch.device("cuda:1")
    dtype = torch.float16
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", torch_dtype=dtype).to(device).eval()
    unet.requires_grad_(False)
    adapter = ConflictAwareUNetAdapter(
        unet, content_dim=256, style_dim=64, conflict_dim=128,
        num_content_tokens=4, layer_paths=("mid_block",),
    ).to(device=device, dtype=dtype)
    content = torch.randn(1, 256, device=device, dtype=dtype)
    style = torch.randn(1, 64, device=device, dtype=dtype)
    conflict = torch.randn(1, 128, device=device, dtype=dtype)
    latent = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device)
    tokens = adapter.content_adapter(content)
    with torch.no_grad():
        baseline = unet(latent, timestep, encoder_hidden_states=tokens).sample
        zero_init = adapter(latent, timestep, content, style, conflict).sample
    max_diff = float((baseline - zero_init).abs().max())

    content = content.detach().requires_grad_(True)
    style = style.detach().requires_grad_(True)
    conflict = conflict.detach().requires_grad_(True)
    prediction = adapter(latent, timestep, content, style, conflict).sample
    prediction.float().square().mean().backward()
    checks = {
        "full_sd15_loaded": True,
        "output_shape": list(prediction.shape) == [1, 4, 64, 64],
        "output_finite": bool(torch.isfinite(prediction).all()),
        "zero_init_exact_identity": max_diff == 0.0,
        "content_gradient": content.grad is not None and bool(torch.isfinite(content.grad).all()),
        "style_gradient": style.grad is not None and bool(torch.isfinite(style.grad).all()),
        "conflict_gradient": conflict.grad is not None and bool(torch.isfinite(conflict.grad).all()),
        "cafm_parameter_gradient": adapter.cafm_layers["mid_block"].gamma_base.weight.grad is not None,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "zero_init_max_abs_diff": max_diff,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        "dtype": str(dtype),
        "latent_shape": list(latent.shape),
    }
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
