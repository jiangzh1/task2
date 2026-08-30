#!/usr/bin/env python3
"""不下载权重的 U-Net 条件注入与 CA-FM hook 回归测试。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "step07_method_stage1" / "src"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "src"))

from spchconvsti_diffusion import ConflictAwareUNetAdapter


class FakeConditionalUNet(nn.Module):
    def __init__(self, channels: int, cross_attention_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(cross_attention_dim=cross_attention_dim)
        self.condition_projection = nn.Linear(cross_attention_dim, channels)
        self.mid_block = nn.Conv2d(channels, channels, 3, padding=1)
        self.mid_block.out_channels = channels
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, sample, timestep, encoder_hidden_states):
        condition = self.condition_projection(encoder_hidden_states.mean(dim=1))[:, :, None, None]
        hidden = self.mid_block(sample + condition)
        return SimpleNamespace(sample=self.output(hidden))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "unet_adapter_smoke.json")
    args = parser.parse_args()
    torch.manual_seed(2026)
    batch, channels = 2, 12
    fake = FakeConditionalUNet(channels, cross_attention_dim=20)
    adapter = ConflictAwareUNetAdapter(
        fake,
        content_dim=16,
        style_dim=10,
        conflict_dim=14,
        num_content_tokens=4,
    )
    sample = torch.randn(batch, channels, 8, 8)
    content = torch.randn(batch, 16, requires_grad=True)
    style = torch.randn(batch, 10, requires_grad=True)
    conflict = torch.randn(batch, 14, requires_grad=True)
    base = torch.randn(batch, 5, 20)

    baseline = fake(sample, 10, adapter.content_adapter(content, base)).sample
    conditioned = adapter(sample, 10, content, style, conflict, base).sample
    identity_ok = torch.allclose(conditioned, baseline)

    cafm = adapter.cafm_layers["mid_block"]
    with torch.no_grad():
        cafm.gamma_conflict.weight.fill_(0.01)
        cafm.beta_conflict.weight.fill_(0.01)
    changed = adapter(sample, 10, content, style, conflict, base).sample
    changed_ok = not torch.allclose(changed, baseline)
    changed.square().mean().backward()
    checks = {
        "content_token_shape": list(adapter.content_adapter(content).shape) == [batch, 4, 20],
        "append_token_shape": list(adapter.content_adapter(content, base).shape) == [batch, 9, 20],
        "zero_init_identity": bool(identity_ok),
        "nonzero_cafm_changes_output": bool(changed_ok),
        "content_gradient": content.grad is not None and bool(torch.isfinite(content.grad).all()),
        "style_gradient": style.grad is not None and bool(torch.isfinite(style.grad).all()),
        "conflict_gradient": conflict.grad is not None and bool(torch.isfinite(conflict.grad).all()),
        "hook_count": len(adapter._hook_handles) == 1,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "output_shape": list(changed.shape),
        "trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
