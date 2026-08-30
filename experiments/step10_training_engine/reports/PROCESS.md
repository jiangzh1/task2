# 训练工程过程报告

## 目标

在不启动正式训练的情况下，先完成论文要求的双卡 DDP、混合精度、checkpoint 与断点续训基础设施。

## 实现范围

- 从 `torchrun` 的 RANK/LOCAL_RANK/WORLD_SIZE 初始化单机 DDP；
- CUDA 使用 NCCL，CPU 测试使用 gloo/单进程；
- rank 偏移随机种子与可选确定性算法；
- fp32/fp16/bf16 autocast；
- fp16 GradScaler、梯度裁剪和 optimizer step；
- checkpoint 保存模型、optimizer、scheduler、scaler、epoch、global_step、配置和完整 RNG；
- 临时文件写完后 `os.replace` 原子提交；
- `latest.json` 解析和断点恢复。

已完成 CPU 单进程状态完整性回归测试。另以 `torchrun --nproc_per_node=2` 和 gloo 完成双进程 DDP 路径测试，验证两 rank 更新后的参数逐元素一致；该测试不占用 GPU。正式双 GPU/NCCL 压力测试仍等待全量 TTS 完成后执行，避免抢占 GPU 0。
