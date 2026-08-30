#!/usr/bin/env python3
"""CPU 验证单进程运行、梯度更新和 checkpoint 完整恢复。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spchconvsti_training import CheckpointManager, DistributedRuntime, MixedPrecisionRuntime, seed_everything


def main() -> int:
    runtime = DistributedRuntime.initialize(force_cpu=True)
    seed_everything(2026, runtime.rank)
    model = runtime.wrap_model(nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    precision = MixedPrecisionRuntime(runtime.device, "fp32")
    inputs = torch.randn(3, 8)
    targets = torch.randn(3, 4)
    with precision.autocast():
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
    grad_norm = precision.backward_and_step(loss, optimizer, model.parameters(), max_grad_norm=1.0)
    scheduler.step()
    expected = model(inputs).detach().clone()

    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifact_dir) as temporary:
        manager = CheckpointManager(Path(temporary))
        saved = manager.save(
            "step_000001",
            model,
            optimizer,
            scheduler,
            precision.scaler,
            epoch=2,
            global_step=17,
            config={"precision": "fp32", "world_size": 1},
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        altered = model(inputs).detach().clone()
        payload = manager.load(None, model, optimizer, scheduler, precision.scaler)
        restored = model(inputs).detach().clone()
        checks = {
            "single_process_cpu": runtime.world_size == 1 and runtime.device.type == "cpu",
            "checkpoint_exists": saved.exists(),
            "weights_were_altered": bool(not torch.allclose(altered, expected)),
            "weights_restored": bool(torch.allclose(restored, expected)),
            "epoch_restored": payload["epoch"] == 2,
            "global_step_restored": payload["global_step"] == 17,
            "optimizer_restored": payload["optimizer"] is not None,
            "scheduler_restored": payload["scheduler"] is not None,
            "rng_state_present": payload["rng_state"] is not None,
            "gradient_norm_finite": grad_norm is not None and bool(torch.isfinite(torch.tensor(grad_norm))),
        }
    runtime.close()
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    report_path = ROOT / "artifacts" / "training_engine_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
