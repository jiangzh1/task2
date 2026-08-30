# 第二阶段偏好训练接口实现报告

## 已完成

依据论文训练策略第二阶段，实现了不依赖语音数据的偏好训练基础接口：

- `freeze_modules`：冻结第一阶段已经训练的网络参数，并设为评估模式；
- `add_noise_pair`：对正、负目标 sticker latent 在相同 diffusion timestep 分别添加标准高斯噪声并由噪声调度器缩放；
- `StageTwoPreferenceObjective`：对语义、情感、社交氛围三维评分执行论文公式中的逐维 margin 偏好损失；
- `in_batch_negative_loss`：按明确的数据规则构造 batch 内负样本矩阵：第 `i` 个样本自身（矩阵对角线）为正样本，同批其余 `j != i` 样本全部为负样本。
- 严格要求评分形状为 `[B, 3]`，维度顺序为 `sem/emo/atm`，且拒绝形状不一致的正负对。

## 验证

CPU 冒烟测试通过：正负对共享 timestep、零噪声时 noisy latent 与原 latent 相同、冻结网络不参与梯度计算、评分头获得有限梯度、三维 margin 损失有限。

测试结果：`artifacts/stage_two_smoke.json`。

## 尚未擅自决定的内容

正负样本规则已由项目方确认：正样本是该样本自身，负样本是同一 batch 内的其他样本。实现采用 `[B,B,3]` 评分矩阵；对角线为自身正样本，非对角线为 batch 内负样本。该规则不从七类情感或冲突标签额外推断偏好。

## 与语音生成的关系

本步骤不读取生成语音，也不占用 GPU，可与全量语音生成完全并行。实际第二阶段训练仍需：第一阶段正式 checkpoint、可部署的 Latent-CLIP 编码器，以及明确的正式偏好配对规则。
## 2026-08-29：批内配对训练器已接通

已新增 `InBatchPreferenceTrainer`。它严格使用已确认的规则：第 `i` 行条件对应第 `i` 个 sticker latent 是正样本；同一 batch 的所有 `j != i` 均是负样本。训练器对候选 latent 加高斯噪声，经预训练 `E_lat` 接口及三个二层 GELU 投影头生成 `[B,B,3]` 分数矩阵，并计算三维 margin 偏好损失。

服务器 CPU 冒烟测试通过：`score_matrix_shape=[3,3,3]`、损失有限且三个投影头均获得梯度。此测试使用占位 `E_lat` 仅验证连接与配对；正式训练前仍必须接入可验证的预训练 `E_lat` 权重。
