# 全量官方数据清单：完成报告

## 结论

已直接依据 STICKERCONV 官方 Parquet 文件生成 12,930 条可用任务样本清单；旧版 `Cleaned` 与 `Refined` 数据及其 VADER/旧 LLM 标签没有进入本次清单或后续标签流程。

| 划分 | 官方行数 | 可用样本数 | 排除数 |
|---|---:|---:|---:|
| Train | 10,785 | 10,785 | 0 |
| Validation | 1,000 | 1,000 | 0 |
| Test | 1,146 | 1,145 | 1 |
| 合计 | 12,931 | 12,930 | 1 |

测试集被排除的唯一记录仅含首轮表情包，没有任何前序对话，因而不能构成“文本—表情包”配对样本；其余原始记录均被保留。

## 官方表情包标签分布

| 类别 | Train | Validation | Test | 合计 |
|---|---:|---:|---:|---:|
| Happiness | 7,123 | 666 | 767 | 8,556 |
| Neutral | 1,534 | 142 | 171 | 1,847 |
| Sadness | 1,258 | 107 | 126 | 1,491 |
| Anger | 480 | 43 | 50 | 573 |
| Surprise | 224 | 30 | 17 | 271 |
| Fear | 146 | 10 | 8 | 164 |
| Disgust | 20 | 2 | 6 | 28 |

此表仅描述官方七类表情包标签的自然分布，不是最终 Conflict/Consistent 的分布，故暂不能用于填写二分类实验表格。

## 任务配对与可追溯性

- 每个样本选择一段对话中最后一个携带表情包的轮次作为目标，使用该轮文本、其全部前序对话和该表情包建立配对。
- 目标角色以 assistant 为主：Train 10,638/10,785，Validation 988/1,000，Test 957/1,145；该不对称性来自官方数据构建，应在后续角色消融/案例分析中明确报告。
- 每个清单记录包含官方源文件名、源行索引、目标轮索引、稳定样本 ID、完整前序上下文及 `origin_anno`。
- 三个官方 Parquet 文件的 SHA-256 已写入 `official_manifest_summary.json`，以便后续复核版本。

## 已冻结的派生标签规则

- 原始 `origin_anno` 永远保留，视觉模型仅提供审计信息，不覆盖官方标签。
- 表情包极性：Happiness、Surprise → Positive；Sadness、Anger、Disgust、Fear → Negative；Neutral → Neutral。
- 后续先为文本生成极性；仅当文本与官方表情包极性为 Positive/Negative 或 Negative/Positive 时标为 Conflict，其余均为 Consistent。

## 下一步

将针对这 12,930 条记录运行已试验通过的文本极性判定流程，并按官方 Train/Validation/Test 划分分别保存结果。该阶段完成后，才能给出正式的 Conflict/Consistent 数据分布、数据集分析表和可进入模型训练的数据文件。
