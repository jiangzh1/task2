# SDXL 与 Latent-CLIP 迁移状态（2026-08-30）

## 已完成的结构迁移

- 在独立目录和 `sdxl-work` 分支中开发，未修改运行中的 SD 1.5 目录；`sd15-baseline` 分支完整保留原实现。
- VAE latent 缓存脚本不再固定 SD 1.5 的 `0.18215`，默认读取加载 VAE 的 `config.scaling_factor`，并把实际尺度写入缓存元数据。
- 新增 SDXL 双文本编码和尺寸微条件的构造接口。论文的语义条件仍由模块二产生的内容 token 注入，空文本仅满足 SDXL U-Net 的架构输入要求。
- 新增官方 Latent-CLIP 封装：直接将 `[B,4,64,64]` SDXL latent 传给 `encode_image`，不调用 VAE decoder，也不输出中间图像。编码器参数冻结，但对输入潜变量保留梯度，供阶段三奖励梯度修正使用。

## 已验证与待验证

- 新增代码已通过 Python 语法编译和 Git 空白检查。
- 已复用既有 TTS 虚拟环境，在 CPU 上通过 SDXL 张量接口烟雾测试：512px latent 为 `[B,4,64,64]`，双文本交叉注意力为 `[B,77,2048]`，附加 `text_embeds` 与 `time_ids` 形状正确。
- 已在 CPU 上通过既有端到端集成烟雾测试；原训练路径的预测形状、损失有限性及模块一、模块二、U-Net、CA-FM 梯度均正常。
- SDXL Base、匹配 VAE 和官方 Latent-CLIP 权重已于 2026-08-31 启动后台下载到 `/data/jzh/2026/task2_sdxl_assets`，该目录不纳入 Git，也不占 GPU。
- 下载完成后需要在 GPU 1 上验证：SDXL U-Net 前向、Tweedie `z0` 到 `E_lat` 的梯度、DDIM 修正步以及 VAE latent 缓存形状和尺度。

## 论文影响

不需要改变三阶段方法架构或流程图中的 Step 3 连线；文字中将基础扩散骨干由 SD 1.5 改为 SDXL Base，并把 `E_lat` 明确为公开预训练 Latent-CLIP（SDXL VAE latent 域）。该组件为复用的预训练评价器，不应作为本文创新点；本文创新仍是冲突感知条件注入与三维潜空间奖励驱动的轨迹修正。
