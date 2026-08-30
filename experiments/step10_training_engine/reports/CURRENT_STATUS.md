# 训练工程当前状态

- CPU 单进程 checkpoint/恢复 10 项检查全部通过；
- 双进程 `torchrun`+gloo 检查通过，world size=2，更新后参数逐元素一致；
- fp32/fp16/bf16、GradScaler、梯度裁剪、随机种子、原子 checkpoint 均已实现。

正式双 RTX 3090 的 NCCL/显存压力测试需等待 GPU 0 的全量 TTS 完成，当前结果不得写成正式双卡训练实验。
