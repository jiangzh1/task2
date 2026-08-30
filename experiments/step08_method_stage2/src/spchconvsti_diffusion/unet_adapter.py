"""将内容条件注入 cross-attention，并用 forward hook 挂载 CA-FM。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, Sequence

import torch
from torch import nn

from spchconvsti.stage2 import ConflictAwareFeatureModulation


def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    current: nn.Module = root
    for part in dotted_path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def infer_output_channels(module: nn.Module) -> int:
    for attribute in ("out_channels", "channels"):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            return value
    resnets = getattr(module, "resnets", None)
    if resnets and hasattr(resnets[-1], "out_channels"):
        return int(resnets[-1].out_channels)
    raise ValueError(f"无法从 {module.__class__.__name__} 推断输出通道，请显式传入 layer_channels")


class ContentConditionTokenAdapter(nn.Module):
    """把单个内容向量转换为 U-Net cross-attention token。"""

    def __init__(self, content_dim: int, cross_attention_dim: int, num_tokens: int = 4) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens 必须大于 0")
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.projection = nn.Sequential(
            nn.LayerNorm(content_dim),
            nn.Linear(content_dim, num_tokens * cross_attention_dim),
        )

    def forward(
        self,
        content: torch.Tensor,
        base_encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.projection(content).reshape(content.shape[0], self.num_tokens, self.cross_attention_dim)
        if base_encoder_hidden_states is None:
            return tokens
        if base_encoder_hidden_states.shape[0] != content.shape[0]:
            raise ValueError("基础文本条件与内容条件 batch 不一致")
        if base_encoder_hidden_states.shape[-1] != self.cross_attention_dim:
            raise ValueError("基础文本条件的 cross_attention_dim 不一致")
        return torch.cat([base_encoder_hidden_states, tokens], dim=1)


@dataclass
class HookState:
    style: torch.Tensor
    conflict: torch.Tensor


class ConflictAwareUNetAdapter(nn.Module):
    """包装 diffusers `UNet2DConditionModel`，不绑定具体 diffusers 小版本。

    默认只在 `mid_block` 注入 CA-FM。其他输出为 Tensor 的 U-Net 子模块可通过
    `layer_paths` 配置；若模块输出不是 Tensor，立即报错，避免静默修改错误对象。
    """

    def __init__(
        self,
        unet: nn.Module,
        content_dim: int,
        style_dim: int,
        conflict_dim: int,
        num_content_tokens: int = 4,
        layer_paths: Sequence[str] = ("mid_block",),
        layer_channels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.unet = unet
        cross_attention_dim = getattr(getattr(unet, "config", None), "cross_attention_dim", None)
        if isinstance(cross_attention_dim, (tuple, list)):
            if len(set(cross_attention_dim)) != 1:
                raise ValueError("当前适配器要求所有 U-Net block 使用相同 cross_attention_dim")
            cross_attention_dim = cross_attention_dim[0]
        if not isinstance(cross_attention_dim, int):
            raise ValueError("U-Net config.cross_attention_dim 必须是整数")
        self.content_adapter = ContentConditionTokenAdapter(content_dim, cross_attention_dim, num_content_tokens)
        self.layer_paths = tuple(layer_paths)
        supplied = layer_channels or {}
        self.cafm_layers = nn.ModuleDict()
        for path in self.layer_paths:
            module = resolve_module(unet, path)
            channels = supplied.get(path, infer_output_channels(module))
            self.cafm_layers[self._safe_key(path)] = ConflictAwareFeatureModulation(style_dim, conflict_dim, channels)
        self._hook_state: HookState | None = None
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    @staticmethod
    def _safe_key(path: str) -> str:
        return path.replace(".", "__")

    def _register_hooks(self) -> None:
        if self._hook_handles:
            raise RuntimeError("CA-FM hook 已注册，拒绝重复注册")
        for path in self.layer_paths:
            target = resolve_module(self.unet, path)
            cafm = self.cafm_layers[self._safe_key(path)]

            def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor, cafm_layer: nn.Module = cafm) -> torch.Tensor:
                if self._hook_state is None:
                    return output
                if not isinstance(output, torch.Tensor):
                    raise TypeError("CA-FM 目标层必须直接输出 Tensor；请改用更具体的子模块路径")
                return cafm_layer(output, self._hook_state.style, self._hook_state.conflict).feature_map

            self._hook_handles.append(target.register_forward_hook(hook))

    @contextmanager
    def conditioned(self, style: torch.Tensor, conflict: torch.Tensor) -> Iterator[None]:
        if self._hook_state is not None:
            raise RuntimeError("不允许嵌套 CA-FM 条件上下文")
        self._hook_state = HookState(style, conflict)
        try:
            yield
        finally:
            self._hook_state = None

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        content: torch.Tensor,
        style: torch.Tensor,
        conflict: torch.Tensor,
        base_encoder_hidden_states: torch.Tensor | None = None,
        **unet_kwargs,
    ):
        encoder_hidden_states = self.content_adapter(content, base_encoder_hidden_states)
        with self.conditioned(style, conflict):
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **unet_kwargs,
            )
