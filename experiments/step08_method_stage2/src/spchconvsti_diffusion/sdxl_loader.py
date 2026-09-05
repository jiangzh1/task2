"""本项目 SDXL Base 1.0 的离线加载器。

SDXL 的单文件基础 checkpoint 与 fp16-fix VAE 分开保存；本模块显式组合二者，
不依赖不完整的 Hub snapshot，也不会在运行时请求网络。
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline


def load_sdxl_base(
    *, checkpoint_path: str | Path, vae_path: str | Path,
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionXLPipeline:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    vae_dir = Path(vae_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"未找到 SDXL Base checkpoint: {checkpoint}")
    if not (vae_dir / "config.json").is_file():
        raise FileNotFoundError(f"未找到 SDXL VAE 目录: {vae_dir}")
    vae = AutoencoderKL.from_pretrained(
        str(vae_dir), torch_dtype=torch_dtype, local_files_only=True,
    )
    # 已缓存的 Base 目录含有 UNet、双文本编码器与 tokenizer；优先用它组装，
    # 以避免 diffusers 0.27 单文件加载器在离线模式额外查询 Hub 的 tokenizer。
    model_dir = checkpoint.parent
    if (model_dir / "model_index.json").is_file():
        return StableDiffusionXLPipeline.from_pretrained(
            str(model_dir), vae=vae, variant="fp16", torch_dtype=torch_dtype,
            local_files_only=True, safety_checker=None, requires_safety_checker=False,
        )
    return StableDiffusionXLPipeline.from_single_file(
        str(checkpoint), vae=vae, torch_dtype=torch_dtype,
        local_files_only=True, safety_checker=None, requires_safety_checker=False,
    )
