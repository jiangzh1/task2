# 正式实验前需要确认的超参数

以下数值不在现有论文/交接中固定，不能擅自作为正式实验设置。

| 参数 | 作用 | 建议的首轮候选 |
|---|---|---|
| `codebook_size` | 阶段一 VQ codebook 大小 | 7（与七类官方表情包标签一致，但仍需由消融确认） |
| `lambda_code` | 阶段一 VQ 损失权重 | 1.0 |
| `lambda_align` | InfoNCE 对齐损失权重 | 0.1 |
| Stage 1 学习率 | 新增模块 AdamW 学习率 | 1e-4 |
| Stage 1 batch size | SDXL 训练 batch | 每卡 2，梯度累积到全局 8 |
| Stage 1 epochs | 阶段一训练轮数 | 20，按 validation early stopping |
| Stage 2 学习率 | 三个双塔评分头 | 1e-4 |
| Stage 2 margin | 批内偏好 hinge margin | 0.2 |
| Stage 2 epochs | 偏好训练轮数 | 10，按 validation 选择 checkpoint |
| DDIM 推理步数 | 最终生成步数 | 50 |
| `eta_zero` | 奖励梯度修正强度 | 需小范围验证：0.01、0.03、0.05 |

上述建议只是首轮可复现设置，不是论文既定事实，也不能在 14 条冒烟通过前写入正式结论。
