"""模块一各预处理器之间的张量契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultimodalFeatures:
    """RoBERTa、WavLM、OpenSMILE 与 Whisper 时间戳的批量输出。

    `word_frame_spans[b, i] = [start, end)` 表示第 i 个词覆盖的语音帧区间；
    padding 词必须填为 `[-1, -1]`。
    """

    text: torch.Tensor
    acoustic: torch.Tensor
    prosody: torch.Tensor
    context: torch.Tensor
    word_frame_spans: torch.Tensor
    text_mask: torch.Tensor
    speech_mask: torch.Tensor
    context_mask: torch.Tensor

    def validate(self) -> None:
        if self.text.ndim != 3 or self.acoustic.ndim != 3 or self.prosody.ndim != 3 or self.context.ndim != 3:
            raise ValueError("四种特征都必须是 [B, L, D] 三维张量")
        batch = self.text.shape[0]
        if any(tensor.shape[0] != batch for tensor in (self.acoustic, self.prosody, self.context)):
            raise ValueError("所有模态的 batch 维必须一致")
        if self.text_mask.shape != self.text.shape[:2]:
            raise ValueError("text_mask 形状错误")
        if self.speech_mask.shape != self.acoustic.shape[:2]:
            raise ValueError("speech_mask 形状错误")
        if self.context_mask.shape != self.context.shape[:2]:
            raise ValueError("context_mask 形状错误")
        if self.acoustic.shape[:2] != self.prosody.shape[:2]:
            raise ValueError("WavLM 与 OpenSMILE 特征必须共享语音时间轴")
        if (self.text_mask.sum(dim=1) == 0).any():
            raise ValueError("每个样本至少需要一个有效文本词")
        if (self.speech_mask.sum(dim=1) == 0).any():
            raise ValueError("每个样本至少需要一个有效语音帧")
        if (self.context_mask.sum(dim=1) == 0).any():
            raise ValueError("每个样本至少需要一个有效上下文 token")
        if self.word_frame_spans.shape != (*self.text.shape[:2], 2):
            raise ValueError("word_frame_spans 必须为 [B, L_y, 2]")
        if self.word_frame_spans.dtype not in (torch.int32, torch.int64):
            raise ValueError("word_frame_spans 必须使用整数帧索引")
        valid_spans = self.word_frame_spans[self.text_mask]
        if valid_spans.numel():
            starts, ends = valid_spans.unbind(dim=-1)
            if (starts < 0).any() or (ends <= starts).any() or (ends > self.acoustic.shape[1]).any():
                raise ValueError("有效词的帧区间必须满足 0 <= start < end <= L_f")
        padded_spans = self.word_frame_spans[~self.text_mask]
        if padded_spans.numel() and (padded_spans != -1).any():
            raise ValueError("padding 词的帧区间必须为 [-1, -1]")
