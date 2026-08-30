# STICKERCONV 官方数据与本地派生数据对账报告

## 结论

服务器中的三份原始 Parquet 与 STICKERCONV 官方发布的样本规模和字段结构一致，没有证据表明数据下载错误。当前异常主要来自任务筛选与本地 Refined 标注流程；另发现官方论文表格中 Validation/Test 的 unique sticker 数量疑似对调。

本次核查只读执行，未修改原始数据、派生数据或论文正文。

## 官方原始数据核对

| Split | 官方论文样本数 | 本地原始 Parquet | 本地 Cleaned |
|---|---:|---:|---:|
| Train | 10,785 | 10,785 | 10,785 |
| Validation | 1,000 | 1,000 | 1,000 |
| Test | 1,146 | 1,146 | 1,145 |

原始 Parquet 字段为：`user_persona`、`user_status`、`conversations`、`emotion`。

Test 少一条并非下载损坏。原始 Test 第 578 行只有第 0 轮包含图片，其后 11 轮没有目标图片。`chuli.py` 使用 `last_img_idx < 1` 过滤“没有历史上下文可用于预测目标表情包”的样本，因此将其排除。该筛选对当前任务合理，但必须在数据构建说明中披露：官方 Test 1,146 条，任务可用 Test 1,145 条。

## 情感字段含义

- 顶层 `emotion` 是对话人物/场景情感，不是每个文本轮次的情感标签。
- 图片对象的 `origin_anno` 是 SER30K 的七类表情包情感：Happiness、Sadness、Anger、Surprise、Disgust、Fear、Neutral。
- STICKERCONV 没有提供逐文本轮次的七类情感真值。
- 图片对象还包含 LLM 生成的自然语言 `emotion` 描述；该描述与 `origin_anno` 有时并不一致，说明原始图片标签本身存在噪声。

## 任务目标分布

按“会话中最后一个带图片的轮次”为目标，过滤后得到：

| Emotion | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Happiness | 7,123 | 666 | 767 | 8,556 |
| Sadness | 1,258 | 107 | 126 | 1,491 |
| Anger | 480 | 43 | 50 | 573 |
| Surprise | 224 | 30 | 17 | 271 |
| Disgust | 20 | 2 | 6 | 28 |
| Fear | 146 | 10 | 8 | 164 |
| Neutral | 1,534 | 142 | 171 | 1,847 |
| Total | 10,785 | 1,000 | 1,145 | 12,930 |

训练目标中 10,638/10,785 来自 assistant 的最后表情包，仅 147 条来自 user。STICKERCONV 官方论文指出 user 更常使用负向表情包，而 system 更偏正向和中性；当前取最后目标轮的策略几乎总是取 system/assistant，因此进一步放大 Happiness 不平衡。这是官方数据生成机制与当前任务目标定义共同造成的，不是下载错误。

## 官方论文与发布数据的可疑差异

官方论文表 5 报告 Train/Validation/Test unique sticker 为 4,798/880/1,439。对本地官方 Parquet 按图片路径去重得到 4,798/1,439/880，Validation 与 Test 正好对调。

同时：

- Train 与 Validation 共享 1,328 个全部表情包路径、533 个目标表情包路径；
- Train 与 Test、Validation 与 Test 均无图片路径重叠；
- 这符合官方说明：训练和交叉验证使用向量库 1–80，测试使用 81–100。

因此本地 split 文件命名与官方划分机制一致，较可能是论文表格/Hugging Face 数据卡抄录时将 Validation/Test 的 unique sticker 数量对调。该问题不影响整体约 5.8K unique sticker 的描述，但若论文需要报告分划分 unique sticker，应以实际发布文件统计为准，并注明统计口径。

## Cleaned 与 Refined 问题归因

### Cleaned

Cleaned 样本集合基本正确。其新增的 `text_sentiment` 来自本地 VADER，不是 STICKERCONV 官方标注。VADER 结果存在明显正向偏斜，只能作为弱标签候选。

### Refined

Refined 是本项目通过本地 Ollama `llama3.2:1b` 对文本重新分类后生成的派生文件，不是官方数据版本。

Refined Train 的问题包括：

- 6,999 个文本轮次为 `ERROR_LLM`；
- 751 个目标样本为 LLM error；
- 2 个会话 ID 缺失、2 个 ID 重复；
- 并发 `as_completed` 写入导致顺序打乱；
- 以输出行数续传无法在乱序/失败情况下正确对应输入。

这些问题来自本地脚本和 Ollama 调用链，不是官方数据错误。稳定 manifest 已经做到“不丢样本并显式标错”，但 Refined 标签尚未修复，不能作为正式真值。

## 与论文草稿需要对齐的事项

以下事项涉及正文修改，当前仅提出，不直接改文档：

1. 官方划分不是严格 8:1:1，而是 10,785/1,000/1,146；任务过滤后为 10,785/1,000/1,145。建议改为“沿用官方划分，并过滤一条无可用历史目标的测试会话”。
2. “沿用基础数据集的七类共情情感”不准确。应写为“沿用 SER30K 表情包的七类情感标签”。顶层对话情感有 15 个实际取值，文本轮次没有七类真值。
3. `test_human_1200` 无法从当前 1,145 条任务测试集无放回抽取 1,200 条。需降低规模、从更大范围重建，或重新定义为 1,200 段录音而非 1,200 个唯一测试样本。
4. 冲突标签必须说明文本情感分类方法、正负中映射、Surprise 的处理方式、neutral-vs-polar 是否属于冲突，以及低置信度/分歧样本如何处理。
5. 真实人声、16 kHz 音频、512×512 图像、人工核验和 Fleiss' Kappa 尚未完成，正文应保持计划或占位语气，不能写成已完成事实。
