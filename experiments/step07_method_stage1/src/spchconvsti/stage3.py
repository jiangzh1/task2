"""模块三：潜空间一致性奖励与扩散采样轨迹自校正。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _broadcast_schedule(value: torch.Tensor | float, target: torch.Tensor) -> torch.Tensor:
    schedule = torch.as_tensor(value, dtype=target.dtype, device=target.device)
    while schedule.ndim < target.ndim:
        schedule = schedule.unsqueeze(-1)
    return schedule


def tweedie_endpoint_estimate(
    z_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    alpha_bar_t: torch.Tensor | float,
) -> torch.Tensor:
    """正文 Eq. z0：由当前噪声状态估计最终无噪潜变量。"""
    alpha = _broadcast_schedule(alpha_bar_t, z_t)
    return (z_t - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)


def ddim_temporary_step(
    estimated_z0: torch.Tensor,
    predicted_noise: torch.Tensor,
    alpha_bar_previous: torch.Tensor | float,
) -> torch.Tensor:
    """正文 Eq. z'_(t-1)。"""
    alpha_previous = _broadcast_schedule(alpha_bar_previous, estimated_z0)
    return torch.sqrt(alpha_previous) * estimated_z0 + torch.sqrt(1.0 - alpha_previous) * predicted_noise


def dynamic_reward_weights(timestep: torch.Tensor, total_steps: int) -> torch.Tensor:
    """返回 `[lambda_sem, lambda_emo, lambda_atm]`，严格实现正文调度公式。"""
    if total_steps <= 0:
        raise ValueError("total_steps 必须大于 0")
    if ((timestep < 0) | (timestep > total_steps)).any():
        raise ValueError("timestep 必须位于 [0, total_steps]")
    ratio = timestep.float() / float(total_steps)
    emotion = 4.0 * ratio * (1.0 - ratio)
    semantic = ratio * (1.0 - emotion)
    atmosphere = (1.0 - ratio) * (1.0 - emotion)
    return torch.stack([semantic, emotion, atmosphere], dim=-1)


class RewardProjectionHead(nn.Module):
    def __init__(self, reference_dim: int, visual_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(reference_dim, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        return self.network(reference)


@dataclass
class RewardOutput:
    visual_embedding: torch.Tensor
    dimension_scores: torch.Tensor
    weights: torch.Tensor
    total_reward: torch.Tensor


class MultiDimensionalLatentReward(nn.Module):
    """语义、情感风格、社会氛围三个潜空间余弦奖励头。"""

    def __init__(
        self,
        latent_encoder: nn.Module,
        visual_dim: int,
        semantic_dim: int,
        emotion_dim: int,
        atmosphere_dim: int,
        freeze_latent_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.latent_encoder = latent_encoder
        if freeze_latent_encoder:
            self.latent_encoder.requires_grad_(False)
        self.semantic_head = RewardProjectionHead(semantic_dim, visual_dim)
        self.emotion_head = RewardProjectionHead(emotion_dim, visual_dim)
        self.atmosphere_head = RewardProjectionHead(atmosphere_dim, visual_dim)

    def forward(
        self,
        estimated_z0: torch.Tensor,
        semantic_reference: torch.Tensor,
        emotion_reference: torch.Tensor,
        atmosphere_reference: torch.Tensor,
        timestep: torch.Tensor,
        total_steps: int,
    ) -> RewardOutput:
        visual = self.latent_encoder(estimated_z0)
        references = (
            self.semantic_head(semantic_reference),
            self.emotion_head(emotion_reference),
            self.atmosphere_head(atmosphere_reference),
        )
        scores = torch.stack([F.cosine_similarity(visual, reference, dim=-1) for reference in references], dim=-1)
        weights = dynamic_reward_weights(timestep, total_steps).to(scores.dtype)
        total = (weights * scores).sum(dim=-1)
        return RewardOutput(visual, scores, weights, total)


class StageTwoProjectionLatentReward(nn.Module):
    """以阶段二训练完成的三组双塔头实现正文 3.3.1 的奖励。

    三个参考严格为：上下文补全语义 ``y_bar_prime``、量化情感风格
    ``e_q``、全局上下文 ``c_bar``。视觉侧始终由冻结的 ``E_lat`` 提取。
    """

    def __init__(self, latent_encoder: nn.Module, projection_heads: tuple[nn.Module, nn.Module, nn.Module]) -> None:
        super().__init__()
        self.latent_encoder = latent_encoder.requires_grad_(False)
        self.projection_heads = nn.ModuleList(projection_heads)

    def forward(self, estimated_z0, semantic_reference, emotion_reference, atmosphere_reference, timestep, total_steps):
        visual = self.latent_encoder(estimated_z0)
        references = (semantic_reference, emotion_reference, atmosphere_reference)
        scores = []
        for head, reference in zip(self.projection_heads, references):
            if not hasattr(head, "forward_condition") or not hasattr(head, "forward_latent"):
                raise TypeError("阶段三必须使用阶段二训练的双塔投影头")
            scores.append(F.cosine_similarity(head.forward_latent(visual), head.forward_condition(reference), dim=-1))
        scores = torch.stack(scores, dim=-1)
        weights = dynamic_reward_weights(timestep, total_steps).to(scores.dtype)
        return RewardOutput(visual, scores, weights, (weights * scores).sum(dim=-1))


@dataclass
class CorrectionOutput:
    corrected_latent: torch.Tensor
    temporary_latent: torch.Tensor
    estimated_z0: torch.Tensor
    reward: RewardOutput
    guidance_gradient: torch.Tensor


def constant_noise_trajectory_correction(
    z_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    alpha_bar_t: torch.Tensor | float,
    alpha_bar_previous: torch.Tensor | float,
    timestep: torch.Tensor,
    total_steps: int,
    eta_zero: float,
    evaluator: MultiDimensionalLatentReward,
    semantic_reference: torch.Tensor,
    emotion_reference: torch.Tensor,
    atmosphere_reference: torch.Tensor,
) -> CorrectionOutput:
    """实现正文 Eq. g_t 与 Eq. z_(t-1)，噪声预测在求导时显式 detach。"""
    current = z_t.detach().requires_grad_(True)
    constant_noise = predicted_noise.detach()
    estimated = tweedie_endpoint_estimate(current, constant_noise, alpha_bar_t)
    reward = evaluator(
        estimated,
        semantic_reference,
        emotion_reference,
        atmosphere_reference,
        timestep,
        total_steps,
    )
    gradient = torch.autograd.grad(reward.total_reward.sum(), current, create_graph=False)[0]
    temporary = ddim_temporary_step(estimated.detach(), constant_noise, alpha_bar_previous)
    alpha = _broadcast_schedule(alpha_bar_t, current)
    eta_t = eta_zero * torch.sqrt(1.0 - alpha)
    corrected = temporary + eta_t * gradient
    return CorrectionOutput(corrected, temporary, estimated, reward, gradient)
