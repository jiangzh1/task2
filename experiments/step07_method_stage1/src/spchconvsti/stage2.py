"""模块二独立核心：双流投影、情感风格量化与 CA-FM。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class DualStreamConditionProjector(nn.Module):
    def __init__(self, joint_dim: int, content_dim: int, emotion_dim: int) -> None:
        super().__init__()
        self.content_projection = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, content_dim))
        self.emotion_projection = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, emotion_dim))

    def forward(self, h_joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.content_projection(h_joint), self.emotion_projection(h_joint)


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    quantized_straight_through: torch.Tensor
    indices: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    total_loss: torch.Tensor


class EmotionStyleQuantizer(nn.Module):
    """最近邻风格码本及正文 Eq. L_code。"""

    def __init__(self, codebook_size: int, emotion_dim: int, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, emotion_dim)
        self.commitment_weight = commitment_weight
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, emotion: torch.Tensor) -> QuantizerOutput:
        distances = (
            emotion.square().sum(dim=-1, keepdim=True)
            + self.codebook.weight.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * emotion @ self.codebook.weight.transpose(0, 1)
        )
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices)
        codebook_loss = F.mse_loss(quantized, emotion.detach())
        commitment_loss = F.mse_loss(emotion, quantized.detach())
        total_loss = codebook_loss + self.commitment_weight * commitment_loss
        straight_through = emotion + (quantized - emotion).detach()
        return QuantizerOutput(quantized, straight_through, indices, codebook_loss, commitment_loss, total_loss)


@dataclass
class CAFMOutput:
    feature_map: torch.Tensor
    gamma_base: torch.Tensor
    beta_base: torch.Tensor
    gamma_adapt: torch.Tensor
    beta_adapt: torch.Tensor


class ConflictAwareFeatureModulation(nn.Module):
    """实现正文 Eq. gamma/beta 与 Eq. F'，输入特征格式为 `[B,C,H,W]`。"""

    def __init__(self, emotion_dim: int, conflict_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma_base = nn.Linear(emotion_dim, channels)
        self.beta_base = nn.Linear(emotion_dim, channels)
        self.gamma_conflict = nn.Linear(conflict_dim, channels, bias=False)
        self.beta_conflict = nn.Linear(conflict_dim, channels, bias=False)
        nn.init.zeros_(self.gamma_base.weight)
        nn.init.zeros_(self.gamma_base.bias)
        nn.init.zeros_(self.beta_base.weight)
        nn.init.zeros_(self.beta_base.bias)
        nn.init.zeros_(self.gamma_conflict.weight)
        nn.init.zeros_(self.beta_conflict.weight)

    def forward(self, feature_map: torch.Tensor, style: torch.Tensor, conflict: torch.Tensor) -> CAFMOutput:
        if feature_map.ndim != 4:
            raise ValueError("CA-FM 特征图必须为 [B,C,H,W]")
        gamma_base = self.gamma_base(style)
        beta_base = self.beta_base(style)
        gamma_adapt = gamma_base + self.gamma_conflict(conflict)
        beta_adapt = beta_base + self.beta_conflict(conflict)
        gamma = gamma_adapt[:, :, None, None]
        beta = beta_adapt[:, :, None, None]
        modulated = (1.0 + gamma) * feature_map + beta
        return CAFMOutput(modulated, gamma_base, beta_base, gamma_adapt, beta_adapt)


@dataclass
class Module2Condition:
    content: torch.Tensor
    emotion_continuous: torch.Tensor
    emotion_quantized: torch.Tensor
    style_indices: torch.Tensor
    codebook_loss: torch.Tensor


class ConflictAwareConditioner(nn.Module):
    """不绑定 diffusers 版本的模块二条件生成核心。"""

    def __init__(
        self,
        joint_dim: int,
        content_dim: int,
        emotion_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.projector = DualStreamConditionProjector(joint_dim, content_dim, emotion_dim)
        self.quantizer = EmotionStyleQuantizer(codebook_size, emotion_dim, commitment_weight)

    def forward(self, h_joint: torch.Tensor) -> Module2Condition:
        content, emotion = self.projector(h_joint)
        quantized = self.quantizer(emotion)
        return Module2Condition(
            content=content,
            emotion_continuous=emotion,
            emotion_quantized=quantized.quantized_straight_through,
            style_indices=quantized.indices,
            codebook_loss=quantized.total_loss,
        )
