# Step 01 数据清单与标签基线：完成报告

## 完成状态

步骤 01 已完成。已从 Cleaned 数据建立 12,930 条稳定、可追踪的实验样本清单，未修改任何原始数据。三份清单均能完整生成，行数与规范来源一致：

| Split | Manifest 样本数 |
|---|---:|
| Train | 10,785 |
| Validation | 1,000 |
| Test | 1,145 |
| Total | 12,930 |

每条记录均包含唯一 `sample_id`、当前目标轮、历史上下文、目标图片、描述、弱标签来源、Refined 状态、空置的人工验证字段、音频占位字段以及跨划分泄漏指纹。

## 标签完整性

### Refined 关联状态

| Split | Available | LLM Error | Missing | Duplicate |
|---|---:|---:|---:|---:|
| Train | 10,030 | 751 | 2 | 2 |
| Validation | 1,000 | 0 | 0 | 0 |
| Test | 1,145 | 0 | 0 | 0 |

训练集中的异常样本已保留在 manifest 中并明确标记，没有静默删除。`verified_conflict` 和 `verified_emotion` 均保持为空，避免把 VADER 或 LLM 候选误当成真值。

## 最终目标情感分布

| Emotion | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Happiness | 7,123 | 666 | 767 | 8,556 |
| Neutral | 1,534 | 142 | 171 | 1,847 |
| Sadness | 1,258 | 107 | 126 | 1,491 |
| Anger | 480 | 43 | 50 | 573 |
| Surprise | 224 | 30 | 17 | 271 |
| Fear | 146 | 10 | 8 | 164 |
| Disgust | 20 | 2 | 6 | 28 |

数据极度不平衡，尤其是 Disgust 和 Fear。后续不能只报告总体准确率，应至少报告宏平均指标、每类结果和置信区间；训练阶段需评估重采样或损失加权。

## 冲突候选差异

| Split | VADER 冲突 | LLM 冲突 | LLM 错误/不可用 |
|---|---:|---:|---:|
| Train | 1,346 | 586 | 755 |
| Validation | 108 | 54 | 0 |
| Test | 130 | 94 | 0 |

两种弱标注器差异较大。当前冲突标签不能作为最终监督或论文统计，下一阶段需输出分层人工审核清单，并定义一致、冲突、不确定三类标注协议。

## 上下文统计

- Train 历史轮数：3–11，均值 8.36
- Validation 历史轮数：5–11，均值 8.29
- Test 历史轮数：4–11，均值 8.22

数据确实适合研究多轮上下文，但后续需要进一步区分纯文本轮次与带历史表情包的轮次。

## 跨划分泄漏审计

完整对话、当前文本以及“当前文本+目标图片”在三个划分之间均无完全重复。目标图片路径方面：

- Train vs Validation：共享 533 个目标图片路径
- Train vs Test：0
- Validation vs Test：0

测试集目前没有发现目标图片路径泄漏，可继续保留用于最终评价。训练/验证之间的大量图片复用可能造成视觉记忆偏差，建议后续同时维护：

1. `official` 划分：保留原始划分，便于与基础数据集保持一致；
2. `strict` 划分：按目标图片分组，确保训练与验证图片不重叠，用于模型选择和鲁棒性检查。

## 数据门禁结果

- 样本集合完整：通过
- JSON 可解析：通过
- 目标图片存在：通过
- 唯一样本 ID：通过（使用 `<split>:<session_id>`）
- Refined 异常可追踪：通过
- 最终冲突真值：未通过，仍需人工/可靠标注协议
- 情感均衡：未通过，需要采样与评估策略
- Train/Validation 图片隔离：未通过，需要 strict 划分
- 音频可用性：未通过，尚未生成

## 下一步

允许进入 `step02_tts_pilot`，但只进行小规模技术验证，不将弱标签直接固化为最终语音情感真值。试点样本需要分层抽样、人工快速确认，并在生成后检查音频存在性、采样率、时长、静音比例和情感可辨识度。

## 可复核产物

- `scripts/build_manifest.py`
- `artifacts/manifest_train.jsonl`
- `artifacts/manifest_validation.jsonl`
- `artifacts/manifest_test.jsonl`
- `artifacts/manifest_summary.json`
- `reports/PROCESS.md`
- `reports/FINAL.md`

输入文件 SHA-256 已记录在 `manifest_summary.json` 中。
