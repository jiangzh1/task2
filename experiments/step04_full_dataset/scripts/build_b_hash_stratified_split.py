#!/usr/bin/env python3
"""按目标表情包图像内容哈希分组，构建版本 B 的 8:1:1 分层划分。"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


SPLITS = ("train", "validation", "test")


def choose_quotas(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in fractions.items()}
    result = {name: int(value) for name, value in raw.items()}
    missing = total - sum(result.values())
    order = sorted(fractions, key=lambda name: (raw[name] - result[name], name), reverse=True)
    for name in order[:missing]:
        result[name] += 1
    return result


def resolve_image(root: Path, image_path: str) -> Path:
    normalized = image_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return root / normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hash_cache: dict[str, str] = {}
    groups: dict[str, list[tuple[tuple[str, str], dict]]] = collections.defaultdict(list)
    strata_total: collections.Counter = collections.Counter()
    path_to_hash: dict[str, str] = {}
    for source_split in SPLITS:
        source_file = args.source_dir / f"spchconvsti_{source_split}.jsonl"
        with source_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                image_path = row["sticker"]["image_path"]
                if image_path not in hash_cache:
                    resolved = resolve_image(args.image_root, image_path)
                    if not resolved.is_file():
                        raise FileNotFoundError(f"目标图像不存在: {resolved}")
                    hash_cache[image_path] = hashlib.sha256(resolved.read_bytes()).hexdigest()
                digest = hash_cache[image_path]
                path_to_hash[image_path] = digest
                key = (row["sticker"]["origin_anno"], row["derived_labels"]["conflict_label"])
                groups[digest].append((key, row))
                strata_total[key] += 1

    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    global_target = choose_quotas(sum(strata_total.values()), fractions)
    stratum_target = {key: choose_quotas(count, fractions) for key, count in strata_total.items()}
    group_keys = {digest: collections.Counter(key for key, _ in rows) for digest, rows in groups.items()}
    rarity = {key: (strata_total[key], sum(key in values for values in group_keys.values())) for key in strata_total}
    assignment: dict[str, str] = {}
    assigned_counts = {split: collections.Counter() for split in SPLITS}
    assigned_totals: collections.Counter = collections.Counter()

    # 为验证集、测试集的每个非空分层单元预留至少一个哈希组，避免出现 0 单元格。
    for split in ("validation", "test"):
        for key in sorted(strata_total, key=lambda value: rarity[value]):
            if assigned_counts[split][key] > 0:
                continue
            candidates = [digest for digest, counts in group_keys.items() if digest not in assignment and counts[key] > 0]
            if not candidates:
                raise RuntimeError(f"无法为 {split} 的 {key} 预留独立图像哈希组")
            digest = min(candidates, key=lambda item: (len(groups[item]), item))
            assignment[digest] = split
            assigned_totals[split] += len(groups[digest])
            assigned_counts[split].update(group_keys[digest])

    rng = random.Random(args.seed)
    remaining = [digest for digest in groups if digest not in assignment]
    rng.shuffle(remaining)
    remaining.sort(key=lambda digest: -len(groups[digest]))
    for digest in remaining:
        counts = group_keys[digest]

        def score(split: str) -> tuple[float, float, float]:
            gain = sum(min(value, max(stratum_target[key][split] - assigned_counts[split][key], 0)) for key, value in counts.items())
            over = sum(max(assigned_counts[split][key] + value - stratum_target[key][split], 0) for key, value in counts.items())
            total_over = max(assigned_totals[split] + len(groups[digest]) - global_target[split], 0)
            total_gap = abs(global_target[split] - (assigned_totals[split] + len(groups[digest])))
            return (gain * 20 - over * 8 - total_over * 3, -total_gap, -assigned_totals[split])

        chosen = max(SPLITS, key=score)
        assignment[digest] = chosen
        assigned_totals[chosen] += len(groups[digest])
        assigned_counts[chosen].update(counts)

    handles = {split: (args.output_dir / f"spchconvsti_{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    moved = 0
    try:
        for digest in sorted(groups):
            split = assignment[digest]
            for _, row in groups[digest]:
                old_split = row["split"]
                moved += old_split != split
                row["split"] = split
                row["source"]["resplit_policy"] = "version_b_sha256_group_stratified_8_1_1"
                row["source"]["pre_hashsafe_split"] = old_split
                row["sticker"]["image_sha256"] = digest
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    path_groups = collections.defaultdict(set)
    for path, digest in path_to_hash.items():
        path_groups[digest].add(path)
    summary = {
        "split_policy": "按目标表情包文件 SHA-256 分组；按官方七类情感×版本B二分类标签贪心分层",
        "seed": args.seed,
        "global_target": global_target,
        "actual_samples": dict(assigned_totals),
        "unique_image_paths": len(path_to_hash),
        "unique_image_hashes": len(groups),
        "duplicate_hash_groups": sum(len(paths) > 1 for paths in path_groups.values()),
        "cross_split_sha256_overlap": 0,
        "moved_from_previous_split": moved,
        "strata": {
            f"{emotion}|{label}": {"total": total, **{split: assigned_counts[split][(emotion, label)] for split in SPLITS}}
            for (emotion, label), total in sorted(strata_total.items())
        },
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
