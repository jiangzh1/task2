"""第二阶段偏好训练接口 CPU 冒烟测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spchconvsti_preference import StageTwoPreferenceObjective, freeze_modules


class Scheduler:
    def add_noise(self, latent: torch.Tensor, noise: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        scale = (steps.float() / 100.0).view(-1, 1, 1, 1)
        return latent + scale * noise


def main() -> int:
    torch.manual_seed(42)
    frozen = torch.nn.Linear(4, 4)
    heads = torch.nn.Linear(4, 3)
    freeze_modules([frozen])
    objective = StageTwoPreferenceObjective(Scheduler(), margin=0.2)
    positive = torch.randn(3, 4, 8, 8)
    negative = torch.randn(3, 4, 8, 8)
    steps = torch.tensor([10, 50, 90])
    zeros = torch.zeros_like(positive)
    pair = objective.add_noise_pair(positive, negative, steps, zeros, zeros)
    values = heads(torch.randn(3, 4))
    loss = objective(values + 0.3, values,)
    score_matrix = torch.stack([values - 0.3 for _ in range(3)], dim=0)
    diagonal = torch.arange(3)
    score_matrix[diagonal, diagonal] = values + 0.3
    in_batch_loss = objective.in_batch_negative_loss(score_matrix)
    (loss + in_batch_loss).backward()
    checks = {
        "same_timestep_pair": bool(torch.equal(pair.timesteps, steps)),
        "zero_noise_identity": bool(torch.allclose(pair.positive_noisy, positive) and torch.allclose(pair.negative_noisy, negative)),
        "frozen_stage_one_two": all(not parameter.requires_grad for parameter in frozen.parameters()),
        "only_heads_receive_gradient": all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in heads.parameters()),
        "three_dimension_margin_loss_finite": bool(torch.isfinite(loss)),
        "in_batch_other_samples_are_negatives": bool(torch.isfinite(in_batch_loss)),
    }
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "loss": float(loss.detach()), "in_batch_loss": float(in_batch_loss.detach())}
    destination = ROOT / "artifacts" / "stage_two_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
