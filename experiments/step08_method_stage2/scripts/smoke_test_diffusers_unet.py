#!/usr/bin/env python3
"""用小型真实 diffusers UNet2DConditionModel 验证适配器 API。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from diffusers import UNet2DConditionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "step07_method_stage1" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from spchconvsti_diffusion import ConflictAwareUNetAdapter


def main() -> int:
    torch.manual_seed(2026)
    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=24,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    adapter = ConflictAwareUNetAdapter(
        unet,
        content_dim=16,
        style_dim=10,
        conflict_dim=14,
        num_content_tokens=4,
        layer_paths=("mid_block",),
    )
    sample = torch.randn(1, 4, 16, 16)
    content = torch.randn(1, 16, requires_grad=True)
    style = torch.randn(1, 10, requires_grad=True)
    conflict = torch.randn(1, 14, requires_grad=True)
    output = adapter(sample, torch.tensor(10), content, style, conflict).sample
    output.square().mean().backward()
    checks = {
        "output_shape": list(output.shape) == [1, 4, 16, 16],
        "output_finite": bool(torch.isfinite(output).all()),
        "content_gradient": content.grad is not None and bool(torch.isfinite(content.grad).all()),
        "style_path_registered": len(adapter.cafm_layers) == 1,
        "hook_registered": len(adapter._hook_handles) == 1,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "diffusers_version": __import__("diffusers").__version__,
        "checks": checks,
        "unet_parameters": sum(parameter.numel() for parameter in unet.parameters()),
    }
    report_path = ROOT / "artifacts" / "diffusers_unet_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
