"""真实特征缓存到模块一 ``MultimodalFeatures`` 的可验证批量接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .contracts import MultimodalFeatures
from .stage1 import seconds_to_frame_spans


REQUIRED_TENSORS = ("text", "context", "acoustic", "prosody_lld", "word_timestamps")


def load_cached_feature(path: Path) -> dict:
    """安全读取由 step14 生成的特征缓存，不允许 pickle 反序列化。"""
    value = torch.load(path, map_location="cpu", weights_only=True)
    missing = [key for key in REQUIRED_TENSORS if key not in value]
    if missing:
        raise ValueError(f"{path} 缺少特征字段: {missing}")
    if value["text"].ndim != 2 or value["context"].ndim != 2 or value["acoustic"].ndim != 2 or value["prosody_lld"].ndim != 2:
        raise ValueError(f"{path} 的序列特征必须为二维张量")
    if value["word_timestamps"].shape != (value["text"].shape[0], 2):
        raise ValueError(f"{path} 的词级时间戳与文本长度不一致")
    if value["acoustic"].shape[0] != value["prosody_lld"].shape[0]:
        raise ValueError(f"{path} 的 WavLM 与 LLD 帧数不一致")
    return value


class CachedFeatureDataset(Dataset[dict]):
    """只读取已验证的真实特征缓存；不在训练循环内重复运行 ASR/声学模型。"""

    def __init__(self, feature_paths: Sequence[Path]) -> None:
        if not feature_paths:
            raise ValueError("至少提供一个真实特征缓存文件")
        self.feature_paths = tuple(Path(path) for path in feature_paths)

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> dict:
        return load_cached_feature(self.feature_paths[index])


def _pad_sequences(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """把 ``[L_i,D]`` 序列零填充为 ``[B,L_max,D]``，同时返回布尔有效掩码。"""
    if not values or any(value.ndim != 2 for value in values):
        raise ValueError("待填充的特征必须是非空的二维张量序列")
    if len({value.shape[1] for value in values}) != 1:
        raise ValueError("同一模态的特征维数必须一致")
    max_length = max(value.shape[0] for value in values)
    batch = values[0].new_zeros((len(values), max_length, values[0].shape[1]))
    mask = torch.zeros((len(values), max_length), dtype=torch.bool)
    for index, value in enumerate(values):
        batch[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return batch, mask


@dataclass
class CachedFeatureBatch:
    sample_ids: tuple[str, ...]
    features: MultimodalFeatures


def collate_cached_features(items: Sequence[dict]) -> CachedFeatureBatch:
    """生成模块一所需的完整、带 mask 的真实数据 batch。"""
    if not items:
        raise ValueError("不能对空 batch 进行拼接")
    texts, text_mask = _pad_sequences([item["text"].float() for item in items])
    contexts, context_mask = _pad_sequences([item["context"].float() for item in items])
    acoustic, speech_mask = _pad_sequences([item["acoustic"].float() for item in items])
    prosody, prosody_mask = _pad_sequences([item["prosody_lld"].float() for item in items])
    if not torch.equal(speech_mask, prosody_mask):
        raise ValueError("声学与韵律序列的有效帧掩码不一致")

    timestamps = texts.new_zeros((*text_mask.shape, 2))
    for index, item in enumerate(items):
        length = item["word_timestamps"].shape[0]
        timestamps[index, :length] = item["word_timestamps"].float()
    frame_rate = {float(item.get("metadata", {}).get("wavlm_frame_rate", 50.0)) for item in items}
    if len(frame_rate) != 1:
        raise ValueError("同一 batch 的 WavLM 帧率必须一致")
    spans = seconds_to_frame_spans(timestamps, text_mask, frame_rate.pop(), acoustic.shape[1])
    features = MultimodalFeatures(
        text=texts,
        acoustic=acoustic,
        prosody=prosody,
        context=contexts,
        word_frame_spans=spans,
        text_mask=text_mask,
        speech_mask=speech_mask,
        context_mask=context_mask,
    )
    features.validate()
    return CachedFeatureBatch(tuple(item["sample_id"] for item in items), features)
