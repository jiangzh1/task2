#!/usr/bin/env python3
"""为 EmoVoice 输入分配跨划分互斥的中性参考说话人。"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def stable_index(value: str, length: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % length


def split_speakers(speakers: list[str]) -> dict[str, list[str]]:
    """按说话人切分约 8:1:1，且在小型来源中保证验证/测试均非空。"""
    speakers = sorted(speakers, key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    n = len(speakers)
    if n < 3:
        raise ValueError(f"可用说话人不足 3 名：{n}")
    n_val = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    if n_val + n_test >= n:
        n_val = n_test = 1
    n_train = n - n_val - n_test
    return {
        "train": speakers[:n_train],
        "val": speakers[n_train : n_train + n_val],
        "test": speakers[n_train + n_val :],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    refs = [json.loads(line) for line in args.reference_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in refs:
        raw_path = Path(row["path"])
        if not raw_path.is_absolute():
            # 构建脚本可能从实验根目录运行并记录相对路径；推理脚本随后会切换
            # 到 EmoVoice 源码目录，因此必须在写入推理 JSONL 前固定为绝对路径。
            cwd_candidate = (Path.cwd() / raw_path).resolve()
            manifest_candidate = (args.reference_manifest.parent.parent / raw_path).resolve()
            row["path"] = str(cwd_candidate if cwd_candidate.exists() else manifest_candidate)
    refs = [row for row in refs if row.get("accepted")]
    by_source_speaker: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in refs:
        by_source_speaker[row["source"]][row["speaker_id"]].append(row)
    if not by_source_speaker:
        raise ValueError("参考库中没有通过质检的中性语音")

    source_pools: dict[str, dict[str, list[str]]] = {}
    wav_lookup: dict[str, list[dict]] = {}
    for source, speaker_rows in sorted(by_source_speaker.items()):
        source_pools[source] = split_speakers(list(speaker_rows))
        wav_lookup.update(speaker_rows)

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    assigned = []
    counts = collections.Counter()
    for row in rows:
        split = row.get("metadata", {}).get("dataset_split", "train").lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"未知数据划分：{split}")
        usable_sources = [source for source in sorted(source_pools) if source_pools[source][split]]
        source = usable_sources[stable_index(row["key"] + ":source", len(usable_sources))]
        speakers = source_pools[source][split]
        speaker = speakers[stable_index(row["key"] + ":speaker", len(speakers))]
        wavs = sorted(wav_lookup[speaker], key=lambda x: x["path"])
        ref = wavs[stable_index(row["key"] + ":wav", len(wavs))]
        row["neutral_speaker_wav"] = ref["path"]
        # 官方加载器在 inference_mode 下仍会调用目标音频 token 的整理函数；
        # 空列表会被转换成仅含 EOA/填充的合法占位，生成时不会作为条件输入。
        row.setdefault("answer_cosyvoice_speech_token", [])
        row.setdefault("metadata", {})["reference_source"] = source
        row["metadata"]["reference_speaker_id"] = speaker
        assigned.append(row)
        counts[(split, source)] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in assigned:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.smoke_output:
        args.smoke_output.write_text(json.dumps(assigned[0], ensure_ascii=False) + "\n", encoding="utf-8")

    speaker_sets = {
        split: sorted({speaker for source in source_pools.values() for speaker in source[split]})
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(set(speaker_sets["train"]) & set(speaker_sets["val"])),
        "train_test": sorted(set(speaker_sets["train"]) & set(speaker_sets["test"])),
        "val_test": sorted(set(speaker_sets["val"]) & set(speaker_sets["test"])),
    }
    summary = {
        "input_samples": len(rows),
        "assigned_samples": len(assigned),
        "accepted_reference_files": len(refs),
        "speaker_pool_sizes": {split: len(values) for split, values in speaker_sets.items()},
        "speaker_overlap": overlaps,
        "sample_assignment_by_split_source": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
