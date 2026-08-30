"""验证 step14 真实缓存到模块一数据契约的批量拼接。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--stage1-src", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage1_src))
    from spchconvsti.real_features import CachedFeatureDataset, collate_cached_features

    paths = sorted(args.feature_dir.glob("*.pt"))[: args.limit]
    dataset = CachedFeatureDataset(paths)
    batch = collate_cached_features([dataset[index] for index in range(len(dataset))])
    features = batch.features
    valid_spans = features.word_frame_spans[features.text_mask]
    print(
        {
            "status": "passed",
            "sample_ids": list(batch.sample_ids),
            "text": list(features.text.shape),
            "context": list(features.context.shape),
            "acoustic": list(features.acoustic.shape),
            "prosody": list(features.prosody.shape),
            "word_frame_spans": list(features.word_frame_spans.shape),
            "valid_word_spans": valid_spans.tolist(),
            "all_spans_within_acoustic_frames": bool((valid_spans[:, 1] <= features.acoustic.shape[1]).all()),
        }
    )


if __name__ == "__main__":
    main()
