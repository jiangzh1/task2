# 真实音频特征链路更新（2026-08-29）

## 本次目标

继续验证论文 3.1.1 所需的真实特征输入链路。该步骤仅使用已完成的少量语音进行接口冒烟，不构成正式训练、消融或实验结果，也不修改正式数据集划分。

## 已修复的问题

1. `extract_real_features.py` 中 RoBERTa 与 WavLM 的加载均显式指定 `local_files_only=True`。此前 WavLM 权重已缓存完整，但 Transformers 在加载时仍尝试访问 Hugging Face 进行文件检查；网络连接重置导致单样本失败。现在提取阶段只使用已缓存的 `roberta-base`、`microsoft/wavlm-base-plus` 和 Whisper `base` 权重。
2. Whisper 的词级时间戳会产生预分词词表。RoBERTa fast tokenizer 已按其接口要求增加 `add_prefix_space=True`，从而允许 `is_split_into_words=True` 并保留 `word_ids` 到词级时间窗的映射。
3. 旧的 14 条小样本清单包含相对音频路径，执行目录变化时会解析错误。`prepare_smoke_subset.py` 现在将 `audio_root` 解析为绝对路径后写入清单。

## 重建的小样本清单

使用 `full_tts_pp_hashsafe_v2` 目前已经完整拼接的音频，按七类官方表情包情感各取文本最短的两条：

- Happiness 1,560 条可用；Sadness 177；Anger 86；Surprise 40；Disgust 2；Fear 16；Neutral 282。
- 共选出 14 条，清单为 `artifacts/smoke_subset_14.jsonl`，统计为 `artifacts/smoke_subset_summary.json`。

这只是当时可用于冒烟的已完成子集，绝不替代 12,930 条正式数据或正式 Train/Validation/Test 流程。

## 单样本真实验证结果

首条短语音 `f96bf7683422a45595e1` 已成功完成，耗时 163.33 秒，无失败项。输出文件为 `artifacts/features/f96bf7683422a45595e1.pt`；安全读取后张量形状如下：

| 字段 | 形状 |
| --- | --- |
| `text` | `[9, 768]` |
| `context` | `[479, 768]` |
| `acoustic`（冻结 WavLM 第 6 层） | `[242, 768]` |
| `prosody_lld`（eGeMAPSv02 LLD） | `[242, 25]` |
| `word_timestamps` | `[9, 2]` |

Whisper 返回 9 个词，RoBERTa 的词级表示也为 9 个，WavLM 与 LLD 的时间帧数均为 242，因此当前真实特征输出与模块一的数据契约一致。

## 后续

下一步可执行 14 条小样本特征提取，并据此实现/验证真实特征的批量 padding、秒级时间戳到 WavLM 帧区间的转换，以及模块一训练入口的非正式连通性测试。

该 CPU 任务不占用正在生成全量语音的 GPU，也不应作为正式训练或结果报告。
