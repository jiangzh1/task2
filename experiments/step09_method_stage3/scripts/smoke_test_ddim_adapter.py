#!/usr/bin/env python3
"""用真实 diffusers DDIMScheduler 验证零修正等价性和非零修正。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from diffusers import DDIMScheduler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "step07_method_stage1" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from spchconvsti.stage3 import MultiDimensionalLatentReward
from spchconvsti_sampling import DiffusersDDIMCorrectionAdapter


class TinyLatentEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(latent.mean(dim=(-2, -1)))


def build_evaluator() -> MultiDimensionalLatentReward:
    return MultiDimensionalLatentReward(
        TinyLatentEncoder(4, 16),
        visual_dim=16,
        semantic_dim=12,
        emotion_dim=10,
        atmosphere_dim=8,
    )


def main() -> int:
    torch.manual_seed(2026)
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(10)
    timestep = scheduler.timesteps[0]
    sample = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(sample)
    semantic = torch.randn(2, 12)
    emotion = torch.randn(2, 10)
    atmosphere = torch.randn(2, 8)

    zero_adapter = DiffusersDDIMCorrectionAdapter(scheduler, build_evaluator(), 10, eta_zero=0.0)
    zero = zero_adapter.step(noise, timestep, sample, 10, semantic, emotion, atmosphere)
    official = scheduler.step(noise, timestep, sample, eta=0.0).prev_sample

    active_adapter = DiffusersDDIMCorrectionAdapter(scheduler, build_evaluator(), 10, eta_zero=0.1)
    active = active_adapter.step(noise, timestep, sample, 10, semantic, emotion, atmosphere)
    checks = {
        "zero_correction_matches_diffusers_ddim": bool(torch.allclose(zero.corrected_latent, official, atol=1e-5)),
        "active_correction_changes_path": bool(not torch.allclose(active.corrected_latent, active.temporary_latent)),
        "gradient_finite": bool(torch.isfinite(active.guidance_gradient).all()),
        "reward_shape": list(active.reward.dimension_scores.shape) == [2, 3],
        "latent_shape": list(active.corrected_latent.shape) == [2, 4, 8, 8],
        "latent_encoder_frozen": all(not p.requires_grad for p in active_adapter.evaluator.latent_encoder.parameters()),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "diffusers_version": __import__("diffusers").__version__,
        "training_timestep": int(timestep),
        "checks": checks,
        "max_zero_difference": float((zero.corrected_latent - official).abs().max()),
    }
    report_path = ROOT / "artifacts" / "ddim_adapter_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
