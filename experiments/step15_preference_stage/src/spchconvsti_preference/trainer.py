"""第二阶段训练器：批内“自身为正、其余为负”的可执行实现。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .stage_two import StageTwoPreferenceObjective, freeze_modules


class TwoTowerProjectionHead(nn.Module):
    """阶段二单个评分维度的条件/latent 双塔投影。

    阶段一条件 ``h_joint`` 与 Latent-CLIP 视觉嵌入维度不同，二者各自经两层
    GELU MLP 映射到同一评分空间，再计算余弦相似度。
    """

    def __init__(self, condition_dim: int, latent_dim: int, projection_dim: int) -> None:
        super().__init__()
        def tower(input_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(input_dim, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))
        self.condition = tower(condition_dim)
        self.latent = tower(latent_dim)

    def forward_condition(self, value: torch.Tensor) -> torch.Tensor:
        return self.condition(value)

    def forward_latent(self, value: torch.Tensor) -> torch.Tensor:
        return self.latent(value)


class InBatchPreferenceTrainer(nn.Module):
    """将论文第二阶段的冻结、加噪、三维评分和批内负样本连接起来。

    ``condition_encoder`` 为第一阶段冻结网络，输入 batch 并返回 ``[B,D]`` 条件表示；
    ``latent_encoder`` 为论文指定的预训练 ``E_lat``，输入 noisy sticker latent 并返回
    ``[B,D]`` 视觉表示。三个 ``projection_heads`` 均为论文的二层 GELU 投影头。
    本类不伪造 ``E_lat`` 权重：实际训练前必须提供可验证的预训练实现。
    """

    def __init__(
        self,
        condition_encoder: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        latent_encoder: nn.Module,
        projection_heads: Sequence[nn.Module],
        objective: StageTwoPreferenceObjective,
        frozen_stage_one: Sequence[nn.Module] = (),
    ) -> None:
        super().__init__()
        if len(projection_heads) != 3:
            raise ValueError("projection_heads 必须恰好包含 sem/emo/atm 三个头")
        self.condition_encoder = condition_encoder
        self.latent_encoder = latent_encoder
        self.projection_heads = nn.ModuleList(projection_heads)
        self.objective = objective
        freeze_modules(frozen_stage_one)

    def score_matrix(self, condition: torch.Tensor, noisy_latents: torch.Tensor) -> torch.Tensor:
        """返回 ``[B,B,3]``：行是条件样本，列是候选 sticker 样本。"""
        latent_features = self.latent_encoder(noisy_latents)
        if condition.ndim != 2 or latent_features.ndim != 2 or condition.shape[0] != latent_features.shape[0]:
            raise ValueError("condition 与 E_lat 输出必须为同一 batch 的二维张量")
        scores = []
        for head in self.projection_heads:
            # 兼容旧的同维合成测试；正式 SDXL 训练使用 TwoTowerProjectionHead。
            if isinstance(head, TwoTowerProjectionHead):
                condition_value = head.forward_condition(condition)
                latent_value = head.forward_latent(latent_features)
            else:
                condition_value = head(condition)
                latent_value = head(latent_features)
            cond_projection = F.normalize(condition_value, dim=-1)
            latent_projection = F.normalize(latent_value, dim=-1)
            scores.append(cond_projection @ latent_projection.T)
        return torch.stack(scores, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor], sticker_latents: torch.Tensor, timesteps: torch.Tensor) -> dict[str, torch.Tensor]:
        # 正样本就是本行 sticker；批内其它行均作为负样本，故只需对一个候选矩阵统一加噪。
        with torch.no_grad():
            condition = self.condition_encoder(batch)
        noisy = self.objective.noise_scheduler.add_noise(sticker_latents, torch.randn_like(sticker_latents), timesteps)
        matrix = self.score_matrix(condition, noisy)
        loss = self.objective.in_batch_negative_loss(matrix)
        return {"loss": loss, "score_matrix": matrix, "noisy_latents": noisy}
