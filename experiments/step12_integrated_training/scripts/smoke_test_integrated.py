#!/usr/bin/env python3
"""用小型真实 diffusers U-Net 验证完整第一阶段前向和反向传播。"""

from __future__ import annotations

import os
import sys

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

ROOT = "/data/jzh/2026/task2/experiments"
for path in (
    f"{ROOT}/step07_method_stage1/src",
    f"{ROOT}/step08_method_stage2/src",
    f"{ROOT}/step12_integrated_training/src",
):
    sys.path.insert(0, path)

from spchconvsti.contracts import MultimodalFeatures
from spchconvsti.stage1 import SpeechTextConflictReasoner
from spchconvsti.stage2 import ConflictAwareConditioner
from spchconvsti_diffusion.unet_adapter import ConflictAwareUNetAdapter
from spchconvsti_integrated import SpchConvStiStageOne


def main() -> int:
    torch.manual_seed(7)
    batch, words, frames, context = 2, 4, 8, 5
    reasoner = SpeechTextConflictReasoner(
        text_dim=16, acoustic_dim=12, prosody_dim=4, context_dim=16,
        model_dim=16, projection_dim=8, emotion_dim=8, local_conflict_dim=8,
        joint_dim=16, num_heads=4, fusion_layers=1, dropout=0.0,
    )
    conditioner = ConflictAwareConditioner(16, 12, 8, codebook_size=7)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=12, attention_head_dim=4,
        norm_num_groups=8,
    )
    adapted = ConflictAwareUNetAdapter(unet, content_dim=12, style_dim=8, conflict_dim=16)
    scheduler = DDPMScheduler(num_train_timesteps=20, prediction_type="epsilon")
    model = SpchConvStiStageOne(reasoner, conditioner, adapted, scheduler)
    spans = torch.tensor([[[0,2],[2,4],[4,6],[6,8]]] * batch)
    features = MultimodalFeatures(
        text=torch.randn(batch, words, 16),
        acoustic=torch.randn(batch, frames, 12),
        prosody=torch.randn(batch, frames, 4),
        context=torch.randn(batch, context, 16),
        word_frame_spans=spans,
        text_mask=torch.ones(batch, words, dtype=torch.bool),
        speech_mask=torch.ones(batch, frames, dtype=torch.bool),
        context_mask=torch.ones(batch, context, dtype=torch.bool),
    )
    latents = torch.randn(batch, 4, 16, 16)
    noise = torch.randn_like(latents)
    out = model(features, latents, torch.tensor([3, 12]), noise=noise)
    checks = {
        "prediction_shape": tuple(out.predicted_noise.shape) == tuple(latents.shape),
        "loss_finite": bool(torch.isfinite(out.total_loss)),
        "loss_components_positive": all(float(v.detach()) >= 0 for v in (out.content_loss, out.codebook_loss, out.alignment_loss)),
    }
    out.total_loss.backward()
    checks["reasoner_gradient"] = reasoner.fusion_seed.weight.grad is not None
    checks["conditioner_gradient"] = conditioner.projector.content_projection[1].weight.grad is not None
    checks["unet_gradient"] = next(unet.parameters()).grad is not None
    checks["cafm_gradient"] = adapted.cafm_layers["mid_block"].gamma_base.weight.grad is not None
    print({"checks": checks, "loss": float(out.total_loss.detach())})
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
