#!/usr/bin/env python3
"""Create a leakage-safe, group-aware, label-stratified version-B split."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


SOURCE_SPLITS = ("train", "validation", "test")
TARGET_SPLITS = ("train", "validation", "test")
EMOTIONS = ("Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral")


def choose_quotas(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in fractions.items()}
    result = {name: int(value) for name, value in raw.items()}
    for name in sorted(fractions, key=lambda item: (raw[item] - result[item], item), reverse=True)[: total - sum(result.values())]:
        result[name] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups, strata_total = collections.defaultdict(list), collections.Counter()
    for source_split in SOURCE_SPLITS:
        with (args.source_dir / f"spchconvsti_{source_split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["sticker"]["origin_anno"], row["derived_labels"]["conflict_label"])
                groups[row["sticker"]["image_path"]].append((key, row))
                strata_total[key] += 1
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    global_target = choose_quotas(sum(strata_total.values()), fractions)
    stratum_target = {key: choose_quotas(count, fractions) for key, count in strata_total.items()}
    # All non-empty strata contain at least five distinct target-image groups.
    # Reserve one group in both validation and test so no table cell is empty.
    assignment, assigned_counts, assigned_totals = {}, {split: collections.Counter() for split in TARGET_SPLITS}, collections.Counter()
    group_keys = {image: collections.Counter(key for key, _ in rows) for image, rows in groups.items()}
    rarity = {key: (strata_total[key], sum(1 for values in group_keys.values() if key in values)) for key in strata_total}
    for split in ("validation", "test"):
        for key in sorted(strata_total, key=lambda value: rarity[value]):
            if assigned_counts[split][key] > 0:
                continue
            candidates = [image for image, counts in group_keys.items() if image not in assignment and counts[key] > 0]
            if not candidates:
                raise RuntimeError(f"Unable to reserve {key} in {split}")
            image = min(candidates, key=lambda item: (len(groups[item]), item))
            assignment[image] = split
            assigned_totals[split] += len(groups[image])
            assigned_counts[split].update(group_keys[image])
    rng = random.Random(args.seed)
    remaining = [image for image in groups if image not in assignment]
    rng.shuffle(remaining)
    remaining.sort(key=lambda image: (-len(groups[image]), image))
    for image in remaining:
        counts = group_keys[image]
        def score(split: str) -> tuple[float, float, float, float]:
            quota_gain = sum(min(value, max(stratum_target[key][split] - assigned_counts[split][key], 0)) for key, value in counts.items())
            quota_over = sum(max(assigned_counts[split][key] + value - stratum_target[key][split], 0) for key, value in counts.items())
            total_gain = min(len(groups[image]), max(global_target[split] - assigned_totals[split], 0))
            total_over = max(assigned_totals[split] + len(groups[image]) - global_target[split], 0)
            return (quota_gain * 20 + total_gain * 3 - quota_over * 8 - total_over * 2, -total_over, -assigned_totals[split] / global_target[split], -len(split))
        split = max(TARGET_SPLITS, key=score)
        assignment[image] = split
        assigned_totals[split] += len(groups[image])
        assigned_counts[split].update(counts)
    output_handles = {split: (args.output_dir / f"spchconvsti_{split}.jsonl").open("w", encoding="utf-8") for split in TARGET_SPLITS}
    try:
        for image in sorted(groups):
            split = assignment[image]
            for _, row in groups[image]:
                original_split = row["split"]
                row["split"] = split
                row["source"]["resplit_policy"] = "version_b_group_aware_stratified_8_1_1"
                row["source"]["original_official_split"] = original_split
                output_handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in output_handles.values():
            handle.close()
    summary = {
        "split_policy": "group-aware by target image path; stratified greedily by official emotion × version-B binary label",
        "seed": args.seed,
        "global_target": global_target,
        "actual_samples": dict(assigned_totals),
        "target_image_overlap": 0,
        "strata": {
            f"{emotion}|{label}": {"total": total, **{split: assigned_counts[split][(emotion, label)] for split in TARGET_SPLITS}}
            for (emotion, label), total in sorted(strata_total.items())
        },
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
