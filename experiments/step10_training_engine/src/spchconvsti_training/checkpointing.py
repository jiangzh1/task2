"""包含 RNG、优化器、scheduler 和 AMP 状态的原子 checkpoint。"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.json"

    def save(
        self,
        name: str,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        scaler=None,
        epoch: int = 0,
        global_step: int = 0,
        config: dict | None = None,
        extra: dict | None = None,
    ) -> Path:
        destination = self.directory / f"{name}.pt"
        temporary = self.directory / f".{name}.tmp.pt"
        payload = {
            "schema_version": "1.0.0",
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": config or {},
            "extra": extra or {},
            "rng_state": capture_rng_state(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        latest_temp = self.directory / ".latest.tmp.json"
        latest_temp.write_text(
            json.dumps({"checkpoint": destination.name, "epoch": epoch, "global_step": global_step}, indent=2),
            encoding="utf-8",
        )
        os.replace(latest_temp, self.latest_path)
        return destination

    def resolve_latest(self) -> Path:
        if not self.latest_path.exists():
            raise FileNotFoundError("latest.json 不存在")
        metadata = json.loads(self.latest_path.read_text(encoding="utf-8"))
        path = self.directory / metadata["checkpoint"]
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def load(
        self,
        path: Path | None,
        model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        scaler=None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
        strict: bool = True,
    ) -> dict:
        checkpoint_path = path or self.resolve_latest()
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        unwrap_model(model).load_state_dict(payload["model"], strict=strict)
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if restore_rng and payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
        return payload
