"""diffusers DDIMScheduler 与论文轨迹修正公式之间的适配器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spchconvsti.stage3 import (
    CorrectionOutput,
    MultiDimensionalLatentReward,
    constant_noise_trajectory_correction,
)


@dataclass
class SchedulerCoordinates:
    training_timestep: int
    previous_training_timestep: int
    alpha_bar_t: torch.Tensor
    alpha_bar_previous: torch.Tensor


class DiffusersDDIMCorrectionAdapter:
    """使用 scheduler 的 alpha 累积表执行论文的确定性 DDIM+奖励修正。

    `schedule_position` 与 `total_guidance_steps` 表示论文动态奖励中的 t/T；它们与
    DDPM 训练时间索引明确分开，避免把 0..999 的训练索引误当成 50 步推理序号。
    """

    def __init__(
        self,
        scheduler,
        evaluator: MultiDimensionalLatentReward,
        total_guidance_steps: int,
        eta_zero: float,
    ) -> None:
        if total_guidance_steps <= 0:
            raise ValueError("total_guidance_steps 必须大于 0")
        if getattr(scheduler.config, "prediction_type", "epsilon") != "epsilon":
            raise ValueError("当前论文公式只支持 epsilon prediction scheduler")
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.total_guidance_steps = total_guidance_steps
        self.eta_zero = eta_zero

    def coordinates(self, timestep: torch.Tensor | int, device: torch.device, dtype: torch.dtype) -> SchedulerCoordinates:
        index = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        if hasattr(self.scheduler, "previous_timestep"):
            previous = int(self.scheduler.previous_timestep(index))
        else:
            if self.scheduler.num_inference_steps is None:
                raise ValueError("调用 step 前必须先执行 scheduler.set_timesteps")
            step_ratio = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            previous = index - step_ratio
        alpha_t = self.scheduler.alphas_cumprod[index].to(device=device, dtype=dtype)
        if previous >= 0:
            alpha_previous = self.scheduler.alphas_cumprod[previous].to(device=device, dtype=dtype)
        else:
            alpha_previous = torch.as_tensor(
                self.scheduler.final_alpha_cumprod,
                device=device,
                dtype=dtype,
            )
        return SchedulerCoordinates(index, previous, alpha_t, alpha_previous)

    def step(
        self,
        predicted_noise: torch.Tensor,
        timestep: torch.Tensor | int,
        sample: torch.Tensor,
        schedule_position: int,
        semantic_reference: torch.Tensor,
        emotion_reference: torch.Tensor,
        atmosphere_reference: torch.Tensor,
    ) -> CorrectionOutput:
        if not 0 <= schedule_position <= self.total_guidance_steps:
            raise ValueError("schedule_position 必须位于 [0,total_guidance_steps]")
        coordinates = self.coordinates(timestep, sample.device, sample.dtype)
        batch_timestep = torch.full(
            (sample.shape[0],),
            float(schedule_position),
            device=sample.device,
            dtype=sample.dtype,
        )
        return constant_noise_trajectory_correction(
            z_t=sample,
            predicted_noise=predicted_noise,
            alpha_bar_t=coordinates.alpha_bar_t,
            alpha_bar_previous=coordinates.alpha_bar_previous,
            timestep=batch_timestep,
            total_steps=self.total_guidance_steps,
            eta_zero=self.eta_zero,
            evaluator=self.evaluator,
            semantic_reference=semantic_reference,
            emotion_reference=emotion_reference,
            atmosphere_reference=atmosphere_reference,
        )
