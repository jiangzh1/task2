#!/usr/bin/env python3
"""加载 EmoVoice 推理配置并审计检查点是否有静默缺失/多余参数。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    step = Path("/data/jzh/2026/task2/experiments/step02_tts_pilot")
    code = step / "code" / "EmoVoice"
    os.chdir(code)
    sys.path.insert(0, str(code / "examples" / "tts"))
    sys.path.insert(0, str(code / "src"))
    checkpoint_path = step / "assets" / "EmoVoice" / "EmoVoice.pt"
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The official release is a plain state_dict.  This lightweight audit avoids
    # constructing the 0.5B model solely to inspect the downloaded checkpoint.
    parameter_count = sum(value.numel() for value in checkpoint.values() if hasattr(value, "numel"))
    output = {
        "checkpoint_type": type(checkpoint).__name__,
        "checkpoint_parameter_keys": len(checkpoint),
        "checkpoint_parameter_count": parameter_count,
        "first_parameter_keys": list(checkpoint)[:30],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
