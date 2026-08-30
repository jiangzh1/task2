"""严格对应论文第二阶段的高斯加噪偏好损失，不定义未在正文给出的正负配对规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def freeze_modules(modules: Iterable[nn.Module]) -> None:
    """冻结第一阶段已训练网络；调用方仅将评分投影头交给优化器。"""
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


@dataclass
class PreferenceNoisyPair:
    positive_noisy: torch.Tensor
    negative_noisy: torch.Tensor
    positive_noise: torch.Tensor
    negative_noise: torch.Tensor
    timesteps: torch.Tensor


class StageTwoPreferenceObjective(nn.Module):
    """构造同 timestep 的正负 noisy latent，并计算三维独立 margin 偏好损失。

    评分函数必须返回形状为 ``[B, 3]`` 的 ``[sem, emo, atm]`` 分数。
    ``pair_source``（如何判定正负表情包）由正式偏好标注或明确的数据构造规则提供；
    本模块刻意不从七类情感标签臆造偏好关系。
    """

    def __init__(self, noise_scheduler, margin: float) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin 必须非负")
        self.noise_scheduler = noise_scheduler
        self.margin = float(margin)

    def add_noise_pair(
        self,
        positive_latents: torch.Tensor,
        negative_latents: torch.Tensor,
        timesteps: torch.Tensor,
        positive_noise: torch.Tensor | None = None,
        negative_noise: torch.Tensor | None = None,
    ) -> PreferenceNoisyPair:
        if positive_latents.shape != negative_latents.shape:
            raise ValueError("正负 latent 形状必须一致")
        if timesteps.shape != (positive_latents.shape[0],):
            raise ValueError("timesteps 必须为 [B]")
        positive_noise = torch.randn_like(positive_latents) if positive_noise is None else positive_noise
        negative_noise = torch.randn_like(negative_latents) if negative_noise is None else negative_noise
        if positive_noise.shape != positive_latents.shape or negative_noise.shape != negative_latents.shape:
            raise ValueError("高斯噪声与对应 latent 形状必须一致")
        return PreferenceNoisyPair(
            positive_noisy=self.noise_scheduler.add_noise(positive_latents, positive_noise, timesteps),
            negative_noisy=self.noise_scheduler.add_noise(negative_latents, negative_noise, timesteps),
            positive_noise=positive_noise,
            negative_noise=negative_noise,
            timesteps=timesteps,
        )

    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        if positive_scores.shape != negative_scores.shape or positive_scores.ndim != 2 or positive_scores.shape[1] != 3:
            raise ValueError("正负评分均必须为 [B,3]，三维顺序为 sem/emo/atm")
        return torch.relu(self.margin - positive_scores + negative_scores).sum(dim=-1).mean()

    def in_batch_negative_loss(self, score_matrix: torch.Tensor) -> torch.Tensor:
        """按用户定义构造 batch 内负样本损失。

        ``score_matrix[i, j, d]`` 表示以第 i 个样本的条件参考评价第 j 个
        sticker latent 时，第 d 个维度的评分。对角线 ``i==j`` 是该样本自身的
        正样本；同批其余 ``j!=i`` 全部是负样本。
        """
        if score_matrix.ndim != 3 or score_matrix.shape[0] != score_matrix.shape[1] or score_matrix.shape[2] != 3:
            raise ValueError("score_matrix 必须为 [B,B,3]，且 B 为同一批次的样本数")
        batch_size = score_matrix.shape[0]
        if batch_size < 2:
            raise ValueError("batch 内负样本要求 batch_size 至少为 2")
        positive = score_matrix.diagonal(dim1=0, dim2=1).transpose(0, 1)  # [B,3]
        pairwise_margin = torch.relu(self.margin - positive.unsqueeze(1) + score_matrix)
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=score_matrix.device)
        return pairwise_margin[negative_mask].sum(dim=-1).mean()
