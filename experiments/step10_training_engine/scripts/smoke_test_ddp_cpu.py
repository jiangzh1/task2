#!/usr/bin/env python3
"""以双进程 gloo 验证 torchrun/DDP 路径，不占用两张 GPU。"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, "/data/jzh/2026/task2/experiments/step10_training_engine/src")
from spchconvsti_training.runtime import DistributedRuntime, seed_everything


def main() -> int:
    runtime = DistributedRuntime.initialize(force_cpu=True, backend="gloo")
    seed_everything(20260824, runtime.rank)
    model = runtime.wrap_model(nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(3, 4) + runtime.rank
    target = torch.randn(3, 2)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    flat = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(runtime.world_size)]
    dist.all_gather(gathered, flat)
    equal = all(torch.equal(gathered[0], item) for item in gathered[1:])
    passed = runtime.world_size == 2 and equal and torch.isfinite(loss)
    if runtime.is_main_process:
        print({"world_size": runtime.world_size, "weights_identical": equal, "loss_finite": bool(torch.isfinite(loss)), "passed": bool(passed)})
    runtime.barrier()
    runtime.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
