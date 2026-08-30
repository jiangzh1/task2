#!/usr/bin/env python3
"""在 SDXL Base 权重上验证条件注入的恒等性和输入梯度。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "step07_method_stage1/src"))
sys.path.insert(0, str(ROOT / "step08_method_stage2/src"))
from spchconvsti_diffusion.sdxl_conditions import build_empty_sdxl_conditions
from spchconvsti_diffusion.unet_adapter import ConflictAwareUNetAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.eval().requires_grad_(False)
    adapter = ConflictAwareUNetAdapter(
        pipeline.unet, content_dim=256, style_dim=64, conflict_dim=128,
        num_content_tokens=4, layer_paths=("mid_block",),
    ).to(device=device, dtype=dtype)
    latent = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device)
    content = torch.randn(1, 256, device=device, dtype=dtype)
    style = torch.randn(1, 64, device=device, dtype=dtype)
    conflict = torch.randn(1, 128, device=device, dtype=dtype)
    conditions = build_empty_sdxl_conditions(pipeline, 1, device, dtype)
    tokens = adapter.content_adapter(content, conditions.encoder_hidden_states)
    with torch.no_grad():
        baseline = pipeline.unet(latent, timestep, encoder_hidden_states=tokens, added_cond_kwargs=conditions.added_cond_kwargs).sample
        adapted = adapter(latent, timestep, content, style, conflict, base_encoder_hidden_states=conditions.encoder_hidden_states, added_cond_kwargs=conditions.added_cond_kwargs).sample
    content = content.detach().requires_grad_(True)
    style = style.detach().requires_grad_(True)
    conflict = conflict.detach().requires_grad_(True)
    predicted = adapter(latent, timestep, content, style, conflict, base_encoder_hidden_states=conditions.encoder_hidden_states, added_cond_kwargs=conditions.added_cond_kwargs).sample
    predicted.float().square().mean().backward()
    checks = {
        "output_shape": list(predicted.shape) == [1, 4, 64, 64],
        "output_finite": bool(torch.isfinite(predicted).all()),
        "zero_init_exact_identity": float((baseline - adapted).abs().max()) == 0.0,
        "content_gradient": content.grad is not None and bool(torch.isfinite(content.grad).all()),
        "style_gradient": style.grad is not None and bool(torch.isfinite(style.grad).all()),
        "conflict_gradient": conflict.grad is not None and bool(torch.isfinite(conflict.grad).all()),
    }
    result = {"passed": all(checks.values()), "checks": checks, "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
