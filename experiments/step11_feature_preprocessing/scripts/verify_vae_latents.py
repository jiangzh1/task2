#!/usr/bin/env python3
"""核验 VAE latent 文件数量、命名、形状、有限值与元数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    files = sorted(args.cache_dir.glob("*.pt"))
    errors = []
    for path in files:
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
            latent = item["latent"]
            if item["sha256"] != path.stem:
                errors.append({"file": path.name, "error": "sha256 与文件名不一致"})
            elif tuple(latent.shape) != (4, 64, 64):
                errors.append({"file": path.name, "error": f"形状错误 {tuple(latent.shape)}"})
            elif not torch.isfinite(latent).all():
                errors.append({"file": path.name, "error": "存在非有限值"})
            elif item["image_size"] != 512 or item["resize_mode"] != "pad" or item["latent_scale"] != 0.18215:
                errors.append({"file": path.name, "error": "预处理元数据不一致"})
        except Exception as error:
            errors.append({"file": path.name, "error": repr(error)})
    report = {
        "passed": len(files) == args.expected and not errors,
        "expected_files": args.expected,
        "actual_files": len(files),
        "valid_files": len(files) - len(errors),
        "errors": errors,
    }
    (args.cache_dir / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
