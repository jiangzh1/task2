"""论文阶段三使用的官方 Latent-CLIP 直接潜变量编码器。

该封装只接收 SDXL VAE 的 NCHW latent，不调用 VAE decoder，也不生成中间图像。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

DEFAULT_MODEL_NAME = "Latent-ViT-B-4-512-plus"


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
    *, source_dir: str | Path, checkpoint_path: str | Path,
    vae_path: str | Path,
    model_name: str = DEFAULT_MODEL_NAME, device: str | torch.device = "cuda",
    precision: str = "fp16",
) -> LatentCLIPEncoder:
    """加载作者公开的 Latent-CLIP 权重并冻结参数。

    ``source_dir`` 指向作者仓库根目录，``checkpoint_path`` 指向其公开的
    ``epoch_34.pt``。不使用 Hugging Face Hub 模型别名，避免把不存在的
    OpenCLIP Hub 配置误当成官方权重。
    参数冻结不影响输入 latent 的梯度，因而可用于阶段三的奖励梯度修正。
    """
    source = Path(source_dir).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    vae_dir = Path(vae_path).expanduser().resolve()
    package_root = source / "src"
    if not (package_root / "open_clip").is_dir():
        raise FileNotFoundError(f"未找到作者 Latent-CLIP 源码: {package_root / 'open_clip'}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"未找到官方 Latent-CLIP checkpoint: {checkpoint}")
    if not (vae_dir / "config.json").is_file():
        raise FileNotFoundError(f"未找到 SDXL VAE 目录: {vae_dir}")

    # Latent-CLIP 发布时依赖的 Diffusers 模块路径已在新版中迁移。仅在
    # 当前 Python 进程注册兼容别名，不修改或 fork 作者源码。
    try:
        import diffusers.models.vae  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        from diffusers.models.autoencoders import vae as modern_vae  # type: ignore[import-not-found]

        sys.modules.setdefault("diffusers.models.vae", modern_vae)
    # 官方模型构造函数硬编码 Hub 名称。运行时将这一次加载定向到已验证的
    # 本地 fp16-fix VAE；传入四通道 latent 时其 encoder/decoder 路径均不参与
    # forward，因此这不会生成或重建任何中间图像。
    from diffusers import AutoencoderKL  # type: ignore[import-not-found]

    original_from_pretrained = AutoencoderKL.from_pretrained

    def _local_sdxl_vae(pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path) == "madebyollin/sdxl-vae-fp16-fix":
            kwargs["local_files_only"] = True
            return original_from_pretrained(str(vae_dir), *args, **kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoencoderKL.from_pretrained = _local_sdxl_vae
    import open_clip  # type: ignore[import-not-found]

    try:
        model = open_clip.create_model(model_name, precision=precision, device=device)
    finally:
        AutoencoderKL.from_pretrained = original_from_pretrained
    open_clip.load_checkpoint(model, str(checkpoint))
    model.eval()
    model.requires_grad_(False)
    return LatentCLIPEncoder(model)
