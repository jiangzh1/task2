# SpchConvSti 全量数据集构建：完成报告

## 最终结论

全量 SpchConvSti 二分类数据集已构建完成，可用于后续方法实现、训练与对比实验。数据集中只有 `Conflict` 与 `Consistent` 两类；无第三类或未确定类。

| 划分 | Conflict | Consistent | 总计 | Conflict 占比 |
|---|---:|---:|---:|---:|
| Train | 1,115 | 9,670 | 10,785 | 10.34% |
| Validation | 99 | 901 | 1,000 | 9.90% |
| Test | 129 | 1,016 | 1,145 | 11.27% |
| 合计 | 1,343 | 11,587 | 12,930 | 10.39% |

## 生成规则

1. 表情包七类情感始终采用官方 `origin_anno`，没有被视觉模型、文本模型或脚本覆盖。
2. 极性映射固定为：Happiness、Surprise → Positive；Sadness、Anger、Disgust、Fear → Negative；Neutral → Neutral。
3. 文本极性由 Gemma3 12B 与 Qwen3 8B 独立盲推理；11,519 条一致结果直接采用，1,411 条分歧由 Qwen3-VL 8B 在不读取图片、表情包标签或前两模型结论的条件下盲裁决。
4. 仅当文本极性与表情包极性为 Positive/Negative 或 Negative/Positive 时标为 `Conflict`；其他所有组合均标为 `Consistent`。

## 完整性核验

- 三个划分共 12,930 条记录，样本 ID 唯一。
- Train、Validation、Test 的记录数分别为 10,785、1,000、1,145。
- 每条记录均保存官方标签来源、文本极性、文本极性来源与二分类标签。
- 已验证所有记录满足固定的七类到极性映射和二分类冲突规则。

## 可直接使用的实验文件

- `final_dataset/spchconvsti_train.jsonl`
- `final_dataset/spchconvsti_validation.jsonl`
- `final_dataset/spchconvsti_test.jsonl`
- `final_dataset/final_dataset_summary.json`

## 解释与实验注意事项

- 二分类正负样本明显不平衡（Conflict 约 10.39%）。后续训练必须采用类别加权损失、重采样或其他明确的缓解策略；评估不能只报告 Accuracy，至少应报告 Macro-F1、Conflict 类 Precision/Recall/F1 以及混淆矩阵。
- 此处的 7 类官方表情包分布与最终二分类分布分别描述不同层面的标签，论文中应避免混用两者的比例。
- 当前数据集构建报告为可直接引用的数据集统计依据；正式 LaTeX 表格将在论文对应表格字段和指标名称核对后单独输出。
