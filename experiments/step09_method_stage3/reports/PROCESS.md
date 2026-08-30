# 模块三 DDIM 集成过程报告

## 目标

把论文的潜空间三维奖励与 Constant-Noise 梯度修正接入 diffusers 0.27.2 的 `DDIMScheduler`。

## 关键实现约束

- 当前只支持正文对应的 epsilon prediction。
- DDPM 训练时间索引和论文奖励调度中的推理位置 t/T 分开传入，避免混淆 0..999 与 50 步采样序号。
- `eta_zero=0` 时，适配器输出必须与 diffusers 确定性 DDIM 的 `prev_sample` 数值一致。
- 预训练 latent encoder 冻结参数，但奖励对当前 latent 的梯度保持可用。
- 正式 `E_lat` 尚未部署，当前用轻量编码器验证 scheduler 集成性质。
- diffusers 0.27.2 没有 `previous_timestep()` 公共方法；适配器按该版本 `DDIMScheduler.step()` 的实现，用 `num_train_timesteps // num_inference_steps` 计算前一训练时间索引，同时保留对新版 API 的兼容分支。
