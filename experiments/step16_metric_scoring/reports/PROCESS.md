# 自动指标逐样本计算实现报告

## 严格依据论文的指标定义

- **EGA**：由 Qwen3-VL-8B-Instruct 给出生成表情包的七类情感；预测类别等于目标官方 sticker 情感时记 100，否则记 0。
- **VQ-a**：由 Qwen3-VL-8B-Instruct 对视觉完整性、表情包风格适配性、视觉表意清晰度分别按 1–5 分评分，取三项均值。
- **CUA**：由 Qwen3-Omni-30B-A3B 根据当前语音与生成表情包，对当前表达意图一致性按 1–5 分评分。
- **DCA**：由 Qwen3-Omni-30B-A3B 根据完整上下文、当前语音与生成表情包，对对话场景一致性按 1–5 分评分。

## 已完成内容

`normalize_objective_annotations.py` 接收每样本的结构化大模型标注，严格校验七类情感标签和全部 1–5 分范围，并输出可直接供 `step13_evaluation_framework` 汇总的 `EGA`、`VQ-a`、`CUA`、`DCA` JSONL。

CPU 冒烟测试通过：EGA 的百分制换算、VQ-a 三项均值、CUA 与 DCA 原始评分均正确。

## 当前边界

本步骤完成了指标计算和输入校验，不虚构任何指标分数。实际调用 Qwen3-VL-8B-Instruct 与 Qwen3-Omni-30B-A3B 需要正式生成图片、当前音频和完整评测集到位后执行；其输出将进入本步骤，再进入已有的严格聚合与 bootstrap 置信区间框架。
