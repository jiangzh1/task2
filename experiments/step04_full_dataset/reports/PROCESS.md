# 步骤 04：全量数据集构建过程记录

## 目标

直接从 STICKERCONV 官方 Parquet 文件重建全量任务清单，避免把旧版 Cleaned/Refined 脚本的弱标签带入当前论文方案。

## 已固定的标签策略

- 表情包七类标签始终保留官方 `origin_anno`，不由视觉模型改写。
- 极性映射：Happiness、Surprise 为 Positive；Sadness、Anger、Disgust、Fear 为 Negative；Neutral 为 Neutral。
- 文本极性将在后续阶段依据已验证的多模型裁决流程生成。
- 只有 Positive 与 Negative 交叉配对时标记为 Conflict；其他情形均为 Consistent。

## 当前动作

构建仅包含来源索引、目标文本、历史上下文、表情包原始字段与空置派生标签字段的全量清单。正式文本极性推理尚未启动。

## 已完成：官方清单构建

- 全量可用样本：12,930（Train 10,785；Validation 1,000；Test 1,145）。
- 测试集排除 1 条仅含首轮表情包、没有先前文本上下文的记录。
- `origin_anno` 的七类自然分布及源文件 SHA-256 已保存至 `artifacts/official_manifest_summary.json`。
- 清单由官方 Parquet 直接生成，未读取旧版 Cleaned/Refined 标签。

## 已完成：文本极性烟雾测试

- Gemma3 12B（GPU 0）与 Qwen3 8B（GPU 1）各完成 5 条盲推理，均成功写入结构化结果。
- 文本模型不读取图片、表情包标签或旧冲突标签；结果可断点续跑。
- 将启动两模型对全量 12,930 条清单的第一轮并行推理；正式二分类标签仍暂未生成。

## 已完成：第一轮全量文本极性推理

- Gemma3 与 Qwen3 均完成 12,930/12,930 条，解析成功率为 100%。
- 两模型一致 11,519 条（89.09%），分歧 1,411 条。
- 已进入第三模型的纯文本盲裁决阶段；尚未生成 Conflict/Consistent 正式标签。

## 已完成：最终二分类标签与数据集文件

- 第三模型完成全部 1,411 条分歧的纯文本盲裁决。
- 最终数据集包含 Conflict 1,343 条、Consistent 11,587 条；总计 12,930 条。
- 最终 JSONL 与汇总文件已输出至 `final_dataset/`，并将执行标签不变量与样本完整性核验。

## 已完成：可替换的冲突定义版本

- `policy_variants/strict_pn/`：仅正负交叉为 Conflict，共 1,343 条 Conflict。
- `policy_variants/neutral_mismatch/`：任意极性不一致为 Conflict，共 4,270 条 Conflict。
- 两版均包含完整的 Train/Validation/Test JSONL，且分别通过逐条规则核验；只需切换目录即可替换后续实验数据。

## 已完成：版本 B 的无空单元正式划分

- 由于用户选择版本 B，已在 `policy_variants/neutral_mismatch_stratified_811/` 创建正式训练、验证、测试文件。
- 新划分以目标表情包图片为分组单位，按官方情感 × 版本 B 二分类标签分层；总计 10,344 / 1,294 / 1,292，约为 8:1:1。
- 所有 14 个非空组合在 Train、Validation、Test 中均有样本，且 3,554 张目标图片跨划分重复数为 0。
