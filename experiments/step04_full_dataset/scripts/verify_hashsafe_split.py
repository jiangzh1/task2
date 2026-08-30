#!/usr/bin/env python3
"""独立核验正式划分的样本、分层、路径及图像内容哈希隔离。"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def resolve(root: Path, value: str) -> Path:
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return root / value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    args = parser.parse_args()
    ids, paths, hashes = {}, {}, {}
    strata = collections.Counter()
    rows_total = 0
    for split in SPLITS:
        ids[split], paths[split], hashes[split] = set(), set(), set()
        for line in (args.dataset_dir / f"spchconvsti_{split}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            assert row["split"] == split
            sample_id = row["sample_id"]
            assert sample_id not in ids[split]
            ids[split].add(sample_id)
            image_path = row["sticker"]["image_path"]
            digest = hashlib.sha256(resolve(args.image_root, image_path).read_bytes()).hexdigest()
            assert digest == row["sticker"]["image_sha256"]
            paths[split].add(image_path)
            hashes[split].add(digest)
            strata[(split, row["sticker"]["origin_anno"], row["derived_labels"]["conflict_label"])] += 1
            rows_total += 1
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    result = {
        "passed": True,
        "total_samples": rows_total,
        "split_counts": {split: len(ids[split]) for split in SPLITS},
        "sample_id_overlap": {f"{a}-{b}": len(ids[a] & ids[b]) for a, b in pairs},
        "image_path_overlap": {f"{a}-{b}": len(paths[a] & paths[b]) for a, b in pairs},
        "image_sha256_overlap": {f"{a}-{b}": len(hashes[a] & hashes[b]) for a, b in pairs},
        "nonzero_strata": sum(value > 0 for value in strata.values()),
        "expected_strata": 14 * 3,
    }
    result["passed"] = (
        rows_total == 12930
        and all(value == 0 for key in ("sample_id_overlap", "image_path_overlap", "image_sha256_overlap") for value in result[key].values())
        and result["nonzero_strata"] == result["expected_strata"]
    )
    out = args.dataset_dir / "verification_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
