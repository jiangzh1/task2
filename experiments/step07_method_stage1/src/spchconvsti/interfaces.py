"""模块二、模块三的接口边界；公式核对前不固定内部实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class GenerationConditioner(nn.Module, ABC):
    """双流条件投影、风格量化与 CA-FM 的统一接口。"""

    @abstractmethod
    def forward(self, joint_representation: torch.Tensor, conflict_signal: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class TrajectoryCorrector(nn.Module, ABC):
    """推理阶段一致性校验与扩散轨迹自校正接口。"""

    @abstractmethod
    def correct(self, latents: torch.Tensor, condition: dict[str, torch.Tensor], timestep: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
