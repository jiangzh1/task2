"""SpchConvSti 第一阶段训练的端到端编排。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spchconvsti.contracts import MultimodalFeatures
from spchconvsti.losses import diffusion_content_loss, stage_one_total_loss
from spchconvsti.stage1 import Module1Output, SpeechTextConflictReasoner
from spchconvsti.stage2 import ConflictAwareConditioner, Module2Condition
from spchconvsti_diffusion.unet_adapter import ConflictAwareUNetAdapter


@dataclass
class TrainingStepOutput:
    total_loss: torch.Tensor
    content_loss: torch.Tensor
    codebook_loss: torch.Tensor
    alignment_loss: torch.Tensor
    predicted_noise: torch.Tensor
    module1: Module1Output
    module2: Module2Condition


class SpchConvStiStageOne(nn.Module):
    """依次执行模块一、双流条件、前向加噪 U-Net 与论文第一阶段总损失。"""

    def __init__(
        self,
        reasoner: SpeechTextConflictReasoner,
        conditioner: ConflictAwareConditioner,
        unet_adapter: ConflictAwareUNetAdapter,
        noise_scheduler,
        lambda_code: float = 1.0,
        lambda_align: float = 0.1,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.conditioner = conditioner
        self.unet_adapter = unet_adapter
        self.noise_scheduler = noise_scheduler
        self.lambda_code = lambda_code
        self.lambda_align = lambda_align

    def forward(
        self,
        features: MultimodalFeatures,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
        base_encoder_hidden_states: torch.Tensor | None = None,
    ) -> TrainingStepOutput:
        if clean_latents.ndim != 4:
            raise ValueError("clean_latents 必须为 [B,C,H,W]")
        if timesteps.shape != (clean_latents.shape[0],):
            raise ValueError("timesteps 必须为 [B]")
        noise = torch.randn_like(clean_latents) if noise is None else noise
        if noise.shape != clean_latents.shape:
            raise ValueError("noise 与 clean_latents 形状必须一致")

        module1 = self.reasoner(features)
        module2 = self.conditioner(module1.h_joint)
        noisy_latents = self.noise_scheduler.add_noise(clean_latents, noise, timesteps)
        prediction = self.unet_adapter(
            sample=noisy_latents,
            timestep=timesteps,
            content=module2.content,
            style=module2.emotion_quantized,
            conflict=module1.delta_modulated,
            base_encoder_hidden_states=base_encoder_hidden_states,
        )
        predicted_noise = prediction.sample if hasattr(prediction, "sample") else prediction[0]
        content_loss = diffusion_content_loss(predicted_noise, noise)
        alignment_loss = self.reasoner.alignment_loss(module1)
        total_loss = stage_one_total_loss(
            content_loss,
            module2.codebook_loss,
            alignment_loss,
            self.lambda_code,
            self.lambda_align,
        )
        return TrainingStepOutput(
            total_loss=total_loss,
            content_loss=content_loss,
            codebook_loss=module2.codebook_loss,
            alignment_loss=alignment_loss,
            predicted_noise=predicted_noise,
            module1=module1,
            module2=module2,
        )
