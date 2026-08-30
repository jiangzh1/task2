#!/usr/bin/env python3
"""严格按正式数据标签聚合逐样本指标，并计算可复现 bootstrap 置信区间。"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap(values: list[float], iterations: int, rng: random.Random) -> dict[str, float | int]:
    if not values:
        raise ValueError("不能聚合空指标")
    mean = sum(values) / len(values)
    draws = []
    for _ in range(iterations):
        draws.append(sum(rng.choice(values) for _ in values) / len(values))
    return {"n": len(values), "mean": mean, "ci95_low": percentile(draws, 0.025), "ci95_high": percentile(draws, 0.975)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="正式某个 split 的 JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="逐样本指标 JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    for line in args.dataset.open(encoding="utf-8"):
        row = json.loads(line)
        labels[row["sample_id"]] = {
            "split": row["split"],
            "emotion": row["derived_labels"]["sticker_emotion_official"],
            "conflict": row["derived_labels"]["conflict_label"],
        }
    predictions = {}
    metric_names = None
    for line_number, line in enumerate(args.predictions.open(encoding="utf-8"), 1):
        row = json.loads(line)
        sample_id = row["sample_id"]
        if sample_id in predictions:
            raise ValueError(f"重复预测 sample_id: {sample_id}")
        if sample_id not in labels:
            raise ValueError(f"预测不属于目标数据集: {sample_id}")
        metrics = row["metrics"]
        current_names = tuple(sorted(metrics))
        metric_names = current_names if metric_names is None else metric_names
        if current_names != metric_names:
            raise ValueError(f"第 {line_number} 行指标字段不一致")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"第 {line_number} 行存在非有限指标")
        predictions[sample_id] = {name: float(value) for name, value in metrics.items()}
    missing = sorted(set(labels) - set(predictions))
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"缺少 {len(missing)} 个样本的指标；如仅调试可显式使用 --allow-incomplete")

    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for sample_id in predictions:
        label = labels[sample_id]
        groups[("overall", "All")].append(sample_id)
        groups[("conflict", label["conflict"])].append(sample_id)
        groups[("emotion", label["emotion"])].append(sample_id)
        groups[("emotion_x_conflict", f"{label['emotion']}|{label['conflict']}")].append(sample_id)

    result_rows = []
    rng = random.Random(args.seed)
    for (group_type, group_name), sample_ids in sorted(groups.items()):
        for metric in metric_names or ():
            stats = bootstrap([predictions[item][metric] for item in sample_ids], args.bootstrap, rng)
            result_rows.append({"group_type": group_type, "group_name": group_name, "metric": metric, **stats})
    report = {
        "dataset_samples": len(labels), "evaluated_samples": len(predictions), "missing_samples": len(missing),
        "complete": not missing, "bootstrap_iterations": args.bootstrap, "seed": args.seed,
        "metrics": list(metric_names or ()), "results": result_rows,
    }
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_type", "group_name", "metric", "n", "mean", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps({key: report[key] for key in report if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
