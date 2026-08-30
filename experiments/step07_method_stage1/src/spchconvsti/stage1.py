"""模块一：上下文协同的意图推理与语义补全。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import MultimodalFeatures


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def seconds_to_frame_spans(
    word_timestamps: torch.Tensor,
    word_mask: torch.Tensor,
    frame_rate: float,
    num_frames: int,
) -> torch.Tensor:
    """把 Whisper 的秒级 `[start, end]` 转为 WavLM 的 `[start, end)` 帧索引。"""

    if word_timestamps.shape != (*word_mask.shape, 2):
        raise ValueError("word_timestamps 必须为 [B, L_y, 2]")
    if frame_rate <= 0 or num_frames <= 0:
        raise ValueError("frame_rate 和 num_frames 必须大于 0")
    starts = torch.floor(word_timestamps[..., 0] * frame_rate).long()
    ends = torch.ceil(word_timestamps[..., 1] * frame_rate).long()
    starts = starts.clamp(0, max(num_frames - 1, 0))
    ends = ends.clamp(1, num_frames)
    ends = torch.maximum(ends, starts + 1).clamp_max(num_frames)
    spans = torch.stack([starts, ends], dim=-1)
    return torch.where(word_mask.unsqueeze(-1), spans, torch.full_like(spans, -1))


class ProsodyEnhancedSpeechAdapter(nn.Module):
    """实现正文中的 `FFN(LN([A;R]W1+b1)) + [A;R]W2`。"""

    def __init__(self, acoustic_dim: int, prosody_dim: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = acoustic_dim + prosody_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.residual_projection = nn.Linear(input_dim, model_dim, bias=False)

    def forward(self, acoustic: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([acoustic, prosody], dim=-1)
        return self.ffn(self.norm(self.input_projection(combined))) + self.residual_projection(combined)


class ProjectionHead(nn.Module):
    """正文中的两层 MLP 投影头 g_text / g_speech。"""

    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def bidirectional_info_nce(z_text: torch.Tensor, z_speech: torch.Tensor, temperature: float) -> torch.Tensor:
    """正文 Eq. L_align；两个方向的交叉熵取和，与原式一致。"""

    if z_text.shape != z_speech.shape or z_text.ndim != 2:
        raise ValueError("InfoNCE 输入必须是形状相同的 [M, d_p] 张量")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    text = F.normalize(z_text, dim=-1)
    speech = F.normalize(z_speech, dim=-1)
    logits = text @ speech.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)


class WordTimestampAligner(nn.Module):
    """按照 Whisper 词级时间窗，对 WavLM/OpenSMILE 帧做均值池化。"""

    def forward(
        self,
        speech: torch.Tensor,
        word_frame_spans: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, word_count = text_mask.shape
        aligned = speech.new_zeros(batch_size, word_count, speech.shape[-1])
        for batch_index in range(batch_size):
            for word_index in torch.nonzero(text_mask[batch_index], as_tuple=False).flatten().tolist():
                start, end = word_frame_spans[batch_index, word_index].tolist()
                aligned[batch_index, word_index] = speech[batch_index, start:end].mean(dim=0)
        return aligned


@dataclass
class Module1Output:
    h_joint: torch.Tensor
    aligned_text: torch.Tensor
    aligned_speech: torch.Tensor
    z_text: torch.Tensor
    z_speech: torch.Tensor
    delta_global: torch.Tensor
    delta_local: torch.Tensor
    delta: torch.Tensor
    arbitration: torch.Tensor
    delta_modulated: torch.Tensor
    local_similarity: torch.Tensor
    local_weights: torch.Tensor
    completed_semantic: torch.Tensor
    pooled_speech: torch.Tensor
    pooled_context: torch.Tensor


class SpeechTextConflictReasoner(nn.Module):
    """按论文 3.1 节实现的上下文协同双模态不一致推理网络。"""

    def __init__(
        self,
        text_dim: int,
        acoustic_dim: int,
        prosody_dim: int,
        context_dim: int,
        model_dim: int = 256,
        projection_dim: int = 128,
        emotion_dim: int = 64,
        local_conflict_dim: int = 64,
        joint_dim: int = 256,
        num_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        alignment_temperature: float = 0.07,
        local_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.alignment_temperature = alignment_temperature
        self.local_temperature = local_temperature
        self.text_adapter = nn.Linear(text_dim, model_dim) if text_dim != model_dim else nn.Identity()
        self.context_adapter = nn.Linear(context_dim, model_dim) if context_dim != model_dim else nn.Identity()
        self.speech_adapter = ProsodyEnhancedSpeechAdapter(acoustic_dim, prosody_dim, model_dim, dropout)
        self.text_projection = ProjectionHead(model_dim, projection_dim)
        self.speech_projection = ProjectionHead(model_dim, projection_dim)
        self.word_aligner = WordTimestampAligner()
        self.text_emotion_projection = nn.Linear(model_dim, emotion_dim, bias=False)
        self.speech_emotion_projection = nn.Linear(model_dim, emotion_dim, bias=False)
        self.local_projection = nn.Linear(model_dim, local_conflict_dim)
        delta_dim = emotion_dim + local_conflict_dim
        self.context_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.context_semantic_norm = nn.LayerNorm(model_dim)
        self.context_arbitration = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, delta_dim),
            nn.Sigmoid(),
        )
        self.fusion_seed = nn.Linear(model_dim * 2 + delta_dim, joint_dim)
        self.semantic_token = nn.Linear(model_dim, joint_dim)
        self.speech_token = nn.Linear(model_dim, joint_dim)
        self.conflict_token = nn.Linear(delta_dim, joint_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=joint_dim,
            nhead=num_heads,
            dim_feedforward=joint_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.output_norm = nn.LayerNorm(joint_dim)

    def forward(self, features: MultimodalFeatures) -> Module1Output:
        features.validate()
        text = self.text_adapter(features.text)
        context = self.context_adapter(features.context)
        speech = self.speech_adapter(features.acoustic, features.prosody)

        pooled_text_global = masked_mean(text, features.text_mask)
        pooled_speech_global = masked_mean(speech, features.speech_mask)
        z_text = self.text_projection(pooled_text_global)
        z_speech = self.speech_projection(pooled_speech_global)

        aligned_speech = self.word_aligner(speech, features.word_frame_spans, features.text_mask)
        aligned_text = text * features.text_mask.unsqueeze(-1).to(text.dtype)
        pooled_text = masked_mean(aligned_text, features.text_mask)
        pooled_speech = masked_mean(aligned_speech, features.text_mask)

        emotion_text = self.text_emotion_projection(pooled_text)
        emotion_speech = self.speech_emotion_projection(pooled_speech)
        delta_global = emotion_text - emotion_speech

        local_difference = aligned_text - aligned_speech
        local_similarity = F.cosine_similarity(aligned_text, aligned_speech, dim=-1, eps=1e-8)
        conflict_logits = (1.0 - local_similarity) / self.local_temperature
        conflict_logits = conflict_logits.masked_fill(~features.text_mask, torch.finfo(conflict_logits.dtype).min)
        local_weights = conflict_logits.softmax(dim=-1)
        local_weights = local_weights * features.text_mask.to(local_weights.dtype)
        delta_local = (local_weights.unsqueeze(-1) * self.local_projection(local_difference)).sum(dim=1)
        delta = torch.cat([delta_global, delta_local], dim=-1)

        completed_text, _ = self.context_attention(
            query=aligned_text,
            key=context,
            value=context,
            key_padding_mask=~features.context_mask,
            need_weights=False,
        )
        completed_text = self.context_semantic_norm(aligned_text + completed_text)
        completed_text = completed_text * features.text_mask.unsqueeze(-1).to(completed_text.dtype)
        completed_semantic = masked_mean(completed_text, features.text_mask)
        pooled_context = masked_mean(context, features.context_mask)
        arbitration = self.context_arbitration(pooled_context)
        delta_modulated = arbitration * delta

        fusion_vector = torch.cat([completed_semantic, pooled_speech, delta_modulated], dim=-1)
        tokens = torch.stack(
            [
                self.fusion_seed(fusion_vector),
                self.semantic_token(completed_semantic),
                self.speech_token(pooled_speech),
                self.conflict_token(delta_modulated),
            ],
            dim=1,
        )
        h_joint = self.output_norm(self.fusion_transformer(tokens)[:, 0])
        return Module1Output(
            h_joint=h_joint,
            aligned_text=aligned_text,
            aligned_speech=aligned_speech,
            z_text=z_text,
            z_speech=z_speech,
            delta_global=delta_global,
            delta_local=delta_local,
            delta=delta,
            arbitration=arbitration,
            delta_modulated=delta_modulated,
            local_similarity=local_similarity,
            local_weights=local_weights,
            completed_semantic=completed_semantic,
            pooled_speech=pooled_speech,
            pooled_context=pooled_context,
        )

    def alignment_loss(self, output: Module1Output) -> torch.Tensor:
        return bidirectional_info_nce(output.z_text, output.z_speech, self.alignment_temperature)
