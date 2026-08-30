"""SDXL U-Net 所需的空文本附加条件。

论文的语义条件由内容 token 注入器提供；此处仅构造 SDXL 架构不可省略的双文本编码
和微条件，不替代论文模块一或模块二。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SDXLConditionInputs:
    encoder_hidden_states: torch.Tensor
    added_cond_kwargs: dict[str, torch.Tensor]


@torch.inference_mode()
def build_empty_sdxl_conditions(pipeline, batch_size: int, device: torch.device, dtype: torch.dtype, image_size: int = 512) -> SDXLConditionInputs:
    """用空字符串构造 SDXL 的结构性基础条件，供内容 token 在其后追加。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    required = ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2")
    missing = [name for name in required if getattr(pipeline, name, None) is None]
    if missing:
        raise ValueError(f"SDXL pipeline 缺少组件: {missing}")
    prompts = [""] * batch_size
    token_1 = pipeline.tokenizer(prompts, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
    token_2 = pipeline.tokenizer_2(prompts, padding="max_length", max_length=pipeline.tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
    hidden_1 = pipeline.text_encoder(token_1, output_hidden_states=True).hidden_states[-2]
    encoded_2 = pipeline.text_encoder_2(token_2, output_hidden_states=True)
    hidden_2 = encoded_2.hidden_states[-2]
    prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1).to(dtype=dtype)
    time_ids = torch.tensor([image_size, image_size, 0, 0, image_size, image_size], device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    return SDXLConditionInputs(prompt_embeds, {"text_embeds": encoded_2[0].to(dtype=dtype), "time_ids": time_ids})
