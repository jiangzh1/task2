"""单机双卡 DDP、混合精度与可复现运行环境。"""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def seed_everything(seed: int, rank: int = 0, deterministic: bool = False) -> int:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return effective_seed


@dataclass
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, force_cpu: bool = False, backend: str | None = None) -> "DistributedRuntime":
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_cuda = torch.cuda.is_available() and not force_cpu
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        initialized_here = False
        if world_size > 1 and not dist.is_initialized():
            selected_backend = backend or ("nccl" if use_cuda else "gloo")
            dist.init_process_group(backend=selected_backend, init_method="env://")
            initialized_here = True
        return cls(rank, local_rank, world_size, device, initialized_here)

    def wrap_model(self, model: nn.Module, find_unused_parameters: bool = False) -> nn.Module:
        model = model.to(self.device)
        if not self.distributed:
            return model
        kwargs = {"find_unused_parameters": find_unused_parameters}
        if self.device.type == "cuda":
            kwargs.update(device_ids=[self.local_rank], output_device=self.local_rank)
        return DistributedDataParallel(model, **kwargs)

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


class MixedPrecisionRuntime:
    def __init__(self, device: torch.device, precision: str = "fp16") -> None:
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision 仅支持 fp32/fp16/bf16")
        self.device = device
        self.precision = precision
        self.enabled = precision != "fp32"
        if precision == "fp16" and device.type != "cuda":
            self.enabled = False
        self.dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled and precision == "fp16")

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters,
        max_grad_norm: float | None = None,
    ) -> float | None:
        self.scaler.scale(loss).backward()
        grad_norm = None
        if max_grad_norm is not None:
            self.scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm))
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return grad_norm
