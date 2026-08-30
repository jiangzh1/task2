"""论文阶段三使用的官方 Latent-CLIP 直接潜变量编码器。

该封装只接收 SDXL VAE 的 NCHW latent，不调用 VAE decoder，也不生成中间图像。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

DEFAULT_MODEL_ID = "hf-hub:wendlerc/latent-clip-b-4-512-plus-34b-80k"


class LatentCLIPEncoder(nn.Module):
    """将官方 Latent-CLIP 适配为阶段三 ``E_lat`` 的视觉嵌入函数。"""

    def __init__(self, model: nn.Module, *, normalize: bool = True) -> None:
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[1:] != (4, 64, 64):
            raise ValueError("Latent-CLIP 要求 SDXL 512px latent，形状必须为 [B,4,64,64]")
        return self.model.encode_image(latents, normalize=self.normalize)


def load_official_latent_clip(
    *, source_dir: str | Path, model_id: str = DEFAULT_MODEL_ID,
    device: str | torch.device = "cuda", precision: str = "fp16",
) -> LatentCLIPEncoder:
    """加载作者公开的 Latent-CLIP 权重并冻结参数。

    ``source_dir`` 指向作者仓库根目录；模型权重由其 ``hf-hub:`` 加载逻辑管理。
    参数冻结不影响输入 latent 的梯度，因而可用于阶段三的奖励梯度修正。
    """
    source = Path(source_dir).expanduser().resolve()
    package_root = source / "src"
    if not (package_root / "open_clip").is_dir():
        raise FileNotFoundError(f"未找到作者 Latent-CLIP 源码: {package_root / 'open_clip'}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    import open_clip  # type: ignore[import-not-found]

    model, _, _ = open_clip.create_model_and_transforms(model_id, precision=precision, device=device)
    model.eval()
    model.requires_grad_(False)
    return LatentCLIPEncoder(model)
