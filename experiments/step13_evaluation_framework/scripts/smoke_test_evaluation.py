#!/usr/bin/env python3
"""构建微型数据，检查完整性拒绝、分组数与置信区间。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    script = Path(__file__).with_name("aggregate_metrics.py")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset, predictions, output = root / "data.jsonl", root / "pred.jsonl", root / "out"
        rows = []
        for index, (emotion, conflict) in enumerate((("Happiness", "Consistent"), ("Happiness", "Conflict"), ("Anger", "Conflict"), ("Anger", "Consistent"))):
            rows.append({"sample_id": f"s{index}", "split": "test", "derived_labels": {"sticker_emotion_official": emotion, "conflict_label": conflict}})
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        predictions.write_text("".join(json.dumps({"sample_id": row["sample_id"], "metrics": {"EGA": index / 10, "VQ-a": 1-index/10}}) + "\n" for index, row in enumerate(rows)), encoding="utf-8")
        subprocess.run([sys.executable, str(script), "--dataset", str(dataset), "--predictions", str(predictions), "--output-dir", str(output), "--bootstrap", "100", "--seed", "7"], check=True)
        report = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
        checks = {
            "complete": report["complete"],
            "sample_count": report["evaluated_samples"] == 4,
            "metric_names": report["metrics"] == ["EGA", "VQ-a"],
            "overall_n": all(row["n"] == 4 for row in report["results"] if row["group_type"] == "overall"),
            "ci_order": all(row["ci95_low"] <= row["mean"] <= row["ci95_high"] for row in report["results"]),
        }
        print(json.dumps({"passed": all(checks.values()), "checks": checks}, ensure_ascii=False))
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
