#!/usr/bin/env python3
"""自动指标逐样本计算层的 CPU 回归测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    script = Path(__file__).with_name("normalize_objective_annotations.py")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "annotation.jsonl", root / "metrics.jsonl"
        source.write_text(json.dumps({"sample_id": "a", "target_emotion": "Anger", "predicted_emotion": "Anger", "visual_integrity": 4, "sticker_style": 5, "visual_clarity": 3, "cua": 4, "dca": 2}) + "\n", encoding="utf-8")
        subprocess.run([sys.executable, str(script), "--annotations", str(source), "--output", str(output)], check=True)
        row = json.loads(output.read_text(encoding="utf-8"))
        checks = {"ega_percent": row["metrics"]["EGA"] == 100.0, "vqa_mean": row["metrics"]["VQ-a"] == 4.0, "cua": row["metrics"]["CUA"] == 4.0, "dca": row["metrics"]["DCA"] == 2.0}
        print(json.dumps({"passed": all(checks.values()), "checks": checks}, ensure_ascii=False))
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
