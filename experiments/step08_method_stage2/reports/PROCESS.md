# 模块二真实集成过程报告

## 本轮目标

在不等待全量语音、且不干扰 GPU 0 TTS 的条件下，将模块二数学核心接到 diffusers U-Net 的真实调用接口。

## 环境审计

- diffusers 0.27.2
- transformers 4.43.4
- accelerate 0.34.2
- PyTorch 2.4.1+cu121
- GPU 0 正在运行 EmoVoice-PP；GPU 1 空闲。
- 服务器尚无 Stable Diffusion v1.5 权重。

## 当前设计

- 内容向量被投影为可配置数量的 cross-attention tokens。
- 若存在原始文本 tokens，可将内容 tokens 追加到原始条件后；否则单独作为条件。
- CA-FM 通过 PyTorch forward hook 注册到明确输出 Tensor 的 U-Net 子模块。
- 默认只挂载 `mid_block`，避免在未做层级消融前擅自同时修改全部 U-Net block。
- hook 遇到 tuple/非 Tensor 输出会立即报错，不静默修改错误对象。
- CA-FM 零初始化时保持预训练 U-Net 原始行为。

## 权重下载策略与结果

使用 Hugging Face 仓库 `stable-diffusion-v1-5/stable-diffusion-v1-5`，只下载 U-Net、VAE、文本编码器、tokenizer、scheduler 和 model index；不下载 safety checker、ONNX、Flax 或重复 fp16 权重。官方下载端点连接反复重置，改用可连通的 `hf-mirror.com` 后完成断点下载。必需文件全部通过存在性检查。

完整 SD1.5 U-Net 已在 GPU 1 完成 FP16 冒烟测试，峰值已分配显存约 2409.47 MiB；CA-FM 零初始化相对原模型最大绝对差为 0，全部条件分支梯度正常。机器结果保存在 `artifacts/sd15_full_smoke.json`。
