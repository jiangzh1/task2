#!/usr/bin/env python3
"""仅验证 SDXL 离线资产可装载；不读取训练样本、不进行训练或采样。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "step07_method_stage1/src"))
sys.path.insert(0, str(ROOT / "step08_method_stage2/src"))
from spchconvsti_diffusion.sdxl_loader import load_sdxl_base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    pipeline = load_sdxl_base(
        checkpoint_path=args.checkpoint, vae_path=args.vae, torch_dtype=torch.float16,
    ).to(args.device)
    checks = {
        "unet_cross_attention_dim": pipeline.unet.config.cross_attention_dim == 2048,
        "vae_scaling_factor": float(pipeline.vae.config.scaling_factor) > 0,
        "dual_tokenizers": pipeline.tokenizer is not None and pipeline.tokenizer_2 is not None,
        "dual_text_encoders": pipeline.text_encoder is not None and pipeline.text_encoder_2 is not None,
    }
    result = {"passed": all(checks.values()), "checks": checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
