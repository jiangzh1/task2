"""不依赖真实音频或 E_lat 权重的第二阶段批内配对连通性测试。"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spchconvsti_preference import InBatchPreferenceTrainer, StageTwoPreferenceObjective, TwoTowerProjectionHead


class Scheduler:
    def add_noise(self, x, noise, timestep):
        return x + noise * timestep.float().view(-1, 1, 1, 1) / 10


class Condition(nn.Module):
    def forward(self, batch):
        return batch["condition"]


torch.manual_seed(7)
b, condition_dim, latent_dim = 3, 8, 10
trainer = InBatchPreferenceTrainer(
    Condition(), nn.Sequential(nn.Flatten(), nn.Linear(4, latent_dim)),
    [TwoTowerProjectionHead(condition_dim, latent_dim, 6) for _ in range(3)],
    StageTwoPreferenceObjective(Scheduler(), margin=0.2), frozen_stage_one=[Condition()],
)
result = trainer({"condition": torch.randn(b, condition_dim)}, torch.randn(b, 1, 2, 2), torch.tensor([1, 2, 3]))
assert result["score_matrix"].shape == (b, b, 3)
assert torch.isfinite(result["loss"])
result["loss"].backward()
assert all(p.grad is not None for h in trainer.projection_heads for p in h.parameters())
print({"status": "passed", "score_matrix_shape": list(result["score_matrix"].shape), "loss": float(result["loss"])})
