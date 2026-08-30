# Latent-CLIP 公开复用性核查（2026-08-29）

## 结论

此前“没有可验证的公开代码/权重”的结论不正确，现已更正。论文作者 Chris Wendler 已公开 Latent-CLIP 源码及 Hugging Face checkpoint；本报告只核查可复用性，未下载模型、未安装依赖、未替换项目的 `E_lat`，也未修改论文。

## 已验证的公开产物

- 源码：`https://github.com/wendlerc/latent_clip` 的 `latent-clip` 分支；该仓库包含模型配置、训练代码及 `Minimum Usage.ipynb`。
- checkpoint：`https://huggingface.co/wendlerc/latent-clip-b-4-512-plus-34b-80k`；模型仓库页面显示总大小约 2.26 GB，并包含 `checkpoints` 与 `Latent-ViT-B-4-512-plus.json`。
- 模型配置：`Latent-ViT-B-4-512-plus`，`embed_dim=640`、4 latent channels、512 图像尺度、patch size 4，使用 `madebyollin/sdxl-vae-fp16-fix`。
- 官方最小用法注释明确其输入可以是 SDXL latent，形状为 `[B,64,64,4]`（channel-last）；调用 `model.encode_image(latents)` 得到视觉表示，不需 VAE decode。

## 与当前工程的关系

当前工程缓存的是 SD 1.5 VAE latent（channel-first `[B,4,64,64]`）。虽然形状可通过置换变为 channel-last，但 SD 1.5 VAE latent 与该预训练模型使用的 SDXL VAE latent 不是同一分布，不能把置换维度当作兼容。

若要严格保留论文流程图中“Latent-CLIP Visual Encoder 直接评分中间 latent、不渲染中间图像”的设计，技术上最自洽的路径是迁移扩散骨干与 VAE 到 SDXL VAE latent 空间，然后复用上述冻结的公开 Latent-CLIP。该迁移不改变 Step 1 语音文本推理和 Step 2 双流/VQ/冲突注入的算法结构；但需要重建 VAE latent 缓存、重做 SDXL 的 U-Net/条件接口和 DDIM 验证。

## 尚未决定、不得擅自执行的事项

1. 是否将工程骨干从 SD 1.5 改为 SDXL；
2. 采用 SDXL 1.0 多步扩散还是 SDXL-Turbo（后者与 Latent-CLIP 原论文生成实验更接近，但和当前 DDIM 多步流程不完全相同）；
3. 下载 checkpoint、安装作者代码、进行 shape/梯度冒烟；
4. 修改论文正文、流程图、引用或实验配置。

以上均需用户确认后再做。
