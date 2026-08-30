#!/usr/bin/env python3
"""CPU 单元测试：论文模块一公式、模块二核心、损失和反向传播。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spchconvsti.contracts import MultimodalFeatures
from spchconvsti.losses import diffusion_content_loss, preference_margin_loss, stage_one_total_loss
from spchconvsti.stage1 import SpeechTextConflictReasoner, WordTimestampAligner, seconds_to_frame_spans
from spchconvsti.stage2 import ConflictAwareConditioner, ConflictAwareFeatureModulation
from spchconvsti.stage3 import (
    MultiDimensionalLatentReward,
    constant_noise_trajectory_correction,
    dynamic_reward_weights,
)


class DummyLatentEncoder(torch.nn.Module):
    def __init__(self, channels: int, visual_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(channels, visual_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(latent.mean(dim=(-2, -1)))


def build_features(config: dict) -> MultimodalFeatures:
    batch = config["batch_size"]
    text_steps = config["text_steps"]
    speech_steps = config["speech_steps"]
    context_steps = config["context_steps"]
    text_mask = torch.ones(batch, text_steps, dtype=torch.bool)
    text_mask[1, -2:] = False
    text_mask[2, -4:] = False
    context_mask = torch.ones(batch, context_steps, dtype=torch.bool)
    context_mask[2, -2:] = False
    speech_mask = torch.ones(batch, speech_steps, dtype=torch.bool)
    timestamps = torch.zeros(batch, text_steps, 2)
    for word_index in range(text_steps):
        timestamps[:, word_index, 0] = word_index * 0.08
        timestamps[:, word_index, 1] = (word_index + 1) * 0.08
    spans = seconds_to_frame_spans(timestamps, text_mask, frame_rate=50.0, num_frames=speech_steps)
    return MultimodalFeatures(
        text=torch.randn(batch, text_steps, config["text_dim"]),
        acoustic=torch.randn(batch, speech_steps, config["acoustic_dim"]),
        prosody=torch.randn(batch, speech_steps, config["prosody_dim"]),
        context=torch.randn(batch, context_steps, config["context_dim"]),
        word_frame_spans=spans,
        text_mask=text_mask,
        speech_mask=speech_mask,
        context_mask=context_mask,
    )


def test_exact_timestamp_mean() -> bool:
    speech = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    spans = torch.tensor([[[0, 2], [2, 5], [-1, -1]]])
    mask = torch.tensor([[True, True, False]])
    aligned = WordTimestampAligner()(speech, spans, mask)
    expected = torch.stack([speech[0, 0:2].mean(0), speech[0, 2:5].mean(0), torch.zeros(2)])
    return bool(torch.allclose(aligned[0], expected))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.json")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "smoke_test_v2.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    features = build_features(config)

    stage1_keys = (
        "text_dim", "acoustic_dim", "prosody_dim", "context_dim", "model_dim", "projection_dim",
        "emotion_dim", "local_conflict_dim", "joint_dim", "num_heads", "fusion_layers", "dropout",
        "alignment_temperature", "local_temperature",
    )
    module1 = SpeechTextConflictReasoner(**{key: config[key] for key in stage1_keys})
    output1 = module1(features)
    alignment_loss = module1.alignment_loss(output1)

    conditioner = ConflictAwareConditioner(
        joint_dim=config["joint_dim"],
        content_dim=config["content_dim"],
        emotion_dim=config["style_dim"],
        codebook_size=config["codebook_size"],
    )
    output2 = conditioner(output1.h_joint)
    conflict_dim = config["emotion_dim"] + config["local_conflict_dim"]
    cafm = ConflictAwareFeatureModulation(config["style_dim"], conflict_dim, config["unet_channels"])
    feature_map = torch.randn(config["batch_size"], config["unet_channels"], 8, 8)
    cafm_output = cafm(feature_map, output2.emotion_quantized, output1.delta)

    predicted_noise = torch.randn_like(feature_map, requires_grad=True)
    target_noise = torch.randn_like(feature_map)
    content_loss = diffusion_content_loss(predicted_noise, target_noise)
    total_loss = stage_one_total_loss(content_loss, output2.codebook_loss, alignment_loss, 1.0, 0.1)
    total_loss.backward()

    positive = torch.tensor([[0.8, 0.7, 0.6], [0.9, 0.7, 0.5]])
    negative = torch.tensor([[0.2, 0.3, 0.1], [0.3, 0.1, 0.2]])
    preference = preference_margin_loss(positive, negative, margin=0.2)
    visual_dim = 36
    reward_evaluator = MultiDimensionalLatentReward(
        latent_encoder=DummyLatentEncoder(config["unet_channels"], visual_dim),
        visual_dim=visual_dim,
        semantic_dim=config["model_dim"],
        emotion_dim=config["style_dim"],
        atmosphere_dim=config["model_dim"],
    )
    timestep = torch.tensor([50.0, 25.0, 1.0])
    correction = constant_noise_trajectory_correction(
        z_t=torch.randn_like(feature_map),
        predicted_noise=torch.randn_like(feature_map),
        alpha_bar_t=torch.tensor([0.25, 0.5, 0.9]),
        alpha_bar_previous=torch.tensor([0.3, 0.55, 0.92]),
        timestep=timestep,
        total_steps=50,
        eta_zero=0.1,
        evaluator=reward_evaluator,
        semantic_reference=output1.completed_semantic.detach(),
        emotion_reference=output2.emotion_quantized.detach(),
        atmosphere_reference=output1.pooled_context.detach(),
    )
    schedule = dynamic_reward_weights(torch.tensor([0.0, 25.0, 50.0]), 50)
    parameters = list(module1.parameters()) + list(conditioner.parameters()) + list(cafm.parameters())
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters)
    local_weight_sums = output1.local_weights.sum(dim=1)
    checks = {
        "timestamp_exact_mean": test_exact_timestamp_mean(),
        "local_weights_sum_to_one": bool(torch.allclose(local_weight_sums, torch.ones_like(local_weight_sums), atol=1e-5)),
        "padding_weights_are_zero": bool((output1.local_weights[~features.text_mask] == 0).all()),
        "arbitration_is_bounded": bool(((output1.arbitration >= 0) & (output1.arbitration <= 1)).all()),
        "ca_fm_zero_initialization_is_identity": bool(torch.allclose(cafm_output.feature_map, feature_map)),
        "style_indices_valid": bool(((output2.style_indices >= 0) & (output2.style_indices < config["codebook_size"])).all()),
        "preference_loss_nonnegative": bool(preference >= 0),
        "reward_weights_sum_to_one": bool(torch.allclose(schedule.sum(dim=-1), torch.ones(3))),
        "reward_weights_nonnegative": bool((schedule >= 0).all()),
        "constant_noise_gradient_finite": bool(torch.isfinite(correction.guidance_gradient).all()),
        "trajectory_correction_changes_latent": bool(not torch.allclose(correction.corrected_latent, correction.temporary_latent)),
        "latent_encoder_is_frozen": bool(all(not parameter.requires_grad for parameter in reward_evaluator.latent_encoder.parameters())),
        "gradients_finite": bool(gradients_finite),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "device": "cpu",
        "checks": checks,
        "shapes": {
            "h_joint": list(output1.h_joint.shape),
            "aligned_text": list(output1.aligned_text.shape),
            "aligned_speech": list(output1.aligned_speech.shape),
            "delta_global": list(output1.delta_global.shape),
            "delta_local": list(output1.delta_local.shape),
            "delta": list(output1.delta.shape),
            "content_condition": list(output2.content.shape),
            "style_condition": list(output2.emotion_quantized.shape),
            "ca_fm_feature": list(cafm_output.feature_map.shape),
            "reward_scores": list(correction.reward.dimension_scores.shape),
            "corrected_latent": list(correction.corrected_latent.shape),
        },
        "losses": {
            "alignment": round(alignment_loss.detach().item(), 6),
            "codebook": round(output2.codebook_loss.detach().item(), 6),
            "content": round(content_loss.detach().item(), 6),
            "stage_one_total": round(total_loss.detach().item(), 6),
            "preference": round(preference.detach().item(), 6),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
