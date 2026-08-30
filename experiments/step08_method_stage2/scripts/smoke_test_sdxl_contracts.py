#!/usr/bin/env python3
"""不下载权重的 SDXL 迁移接口烟雾测试。"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "step07_method_stage1/src"))
sys.path.insert(0, str(ROOT / "step08_method_stage2/src"))
from spchconvsti.stage3 import tweedie_endpoint_estimate
from spchconvsti_diffusion.sdxl_conditions import SDXLConditionInputs

def main() -> int:
    z0 = tweedie_endpoint_estimate(torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64), 0.5)
    condition = SDXLConditionInputs(torch.randn(2, 77, 2048), {"text_embeds": torch.randn(2, 1280), "time_ids": torch.ones(2, 6)})
    checks = {"sdxl_latent_shape": z0.shape == (2, 4, 64, 64), "sdxl_cross_attention_shape": condition.encoder_hidden_states.shape == (2, 77, 2048), "sdxl_added_text_shape": condition.added_cond_kwargs["text_embeds"].shape == (2, 1280), "sdxl_time_ids_shape": condition.added_cond_kwargs["time_ids"].shape == (2, 6)}
    print({"passed": all(checks.values()), "checks": checks})
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
