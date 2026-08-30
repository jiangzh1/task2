#!/usr/bin/env python3
"""在完整 SD1.5 U-Net 上验证第一阶段总计算图，使用 GPU 1。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

ROOT = Path("/data/jzh/2026/task2/experiments")
for path in (ROOT / "step07_method_stage1/src", ROOT / "step08_method_stage2/src", ROOT / "step12_integrated_training/src"):
    sys.path.insert(0, str(path))
from spchconvsti.contracts import MultimodalFeatures
from spchconvsti.stage1 import SpeechTextConflictReasoner
from spchconvsti.stage2 import ConflictAwareConditioner
from spchconvsti_diffusion.unet_adapter import ConflictAwareUNetAdapter
from spchconvsti_integrated import SpchConvStiStageOne


def main() -> int:
    torch.manual_seed(11)
    device, dtype = torch.device("cuda:1"), torch.float16
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    batch, words, frames, contexts = 2, 4, 8, 5
    reasoner = SpeechTextConflictReasoner(
        text_dim=32, acoustic_dim=24, prosody_dim=8, context_dim=32,
        model_dim=32, projection_dim=16, emotion_dim=64, local_conflict_dim=64,
        joint_dim=256, num_heads=8, fusion_layers=1, dropout=0.0,
    )
    conditioner = ConflictAwareConditioner(256, 256, 64, codebook_size=32)
    model_dir = ROOT / "step08_method_stage2/assets/stable-diffusion-v1-5"
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", torch_dtype=dtype).eval()
    unet.requires_grad_(False)
    adapted = ConflictAwareUNetAdapter(unet, 256, 64, 128, layer_paths=("mid_block",))
    model = SpchConvStiStageOne(reasoner, conditioner, adapted, DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon"))
    model.to(device=device, dtype=dtype)
    spans = torch.tensor([[[0,2],[2,4],[4,6],[6,8]]] * batch, device=device)
    features = MultimodalFeatures(
        text=torch.randn(batch, words, 32, device=device, dtype=dtype),
        acoustic=torch.randn(batch, frames, 24, device=device, dtype=dtype),
        prosody=torch.randn(batch, frames, 8, device=device, dtype=dtype),
        context=torch.randn(batch, contexts, 32, device=device, dtype=dtype),
        word_frame_spans=spans,
        text_mask=torch.ones(batch, words, dtype=torch.bool, device=device),
        speech_mask=torch.ones(batch, frames, dtype=torch.bool, device=device),
        context_mask=torch.ones(batch, contexts, dtype=torch.bool, device=device),
    )
    latents = torch.randn(batch, 4, 64, 64, device=device, dtype=dtype)
    out = model(features, latents, torch.tensor([100, 700], device=device))
    out.total_loss.backward()
    checks = {
        "total_loss_finite": bool(torch.isfinite(out.total_loss)),
        "prediction_finite": bool(torch.isfinite(out.predicted_noise).all()),
        "prediction_shape": list(out.predicted_noise.shape) == [2, 4, 64, 64],
        "reasoner_gradient": reasoner.fusion_seed.weight.grad is not None,
        "conditioner_gradient": conditioner.projector.content_projection[1].weight.grad is not None,
        "content_adapter_gradient": adapted.content_adapter.projection[1].weight.grad is not None,
        "cafm_gradient": adapted.cafm_layers["mid_block"].gamma_base.weight.grad is not None,
        "unet_frozen": all(parameter.grad is None for parameter in unet.parameters()),
    }
    report = {
        "passed": all(checks.values()), "checks": checks,
        "loss": float(out.total_loss.detach()),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        "batch": batch, "latent_shape": list(latents.shape), "dtype": str(dtype),
    }
    artifact = ROOT / "step12_integrated_training/artifacts"
    artifact.mkdir(exist_ok=True)
    (artifact / "sd15_integrated_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
