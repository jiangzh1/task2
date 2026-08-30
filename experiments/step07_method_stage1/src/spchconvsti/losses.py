"""论文两阶段训练使用的损失函数。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def diffusion_content_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """正文 Eq. L_content。"""
    return F.mse_loss(predicted_noise, target_noise)


def stage_one_total_loss(
    content_loss: torch.Tensor,
    codebook_loss: torch.Tensor,
    alignment_loss: torch.Tensor,
    lambda_code: float,
    lambda_align: float,
) -> torch.Tensor:
    """正文 Eq. L_total。"""
    return content_loss + lambda_code * codebook_loss + lambda_align * alignment_loss


def preference_margin_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """正文 Eq. L_preference；最后一维对应 sem/emo/atm。"""
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("正负样本评分形状必须一致")
    return torch.relu(margin - positive_scores + negative_scores).sum(dim=-1).mean()
