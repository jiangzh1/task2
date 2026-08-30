# 版本 B 最终数据划分与统计表报告

## 采用的版本

本文推荐使用 `policy_variants/neutral_mismatch_stratified_811/` 作为版本 B 的正式实验数据。Conflict 的定义为：**文本极性与表情包极性不同即为 Conflict**；因此包含正—负、正—中性与负—中性三种不一致情形。

## 为什么重建划分

先前的版本 B 沿用官方 Train/Validation/Test 划分，虽标签正确，但在 `Disgust-Consistent` 的 Validation 单元中出现了 0。这并非计算错误，而是该层只有 8 条样本且官方验证集中仅有 2 条 Disgust 所致。为避免表格中的可避免空单元，现对全部样本按“官方七类表情包情感 × Conflict/Consistent”分层，并按目标表情包图片分组重新划分。

## 最终划分

| 划分 | 样本数 | 占总样本比例 |
|---|---:|---:|
| Train | 10,344 | 79.999% |
| Validation | 1,294 | 10.008% |
| Test | 1,292 | 9.992% |
| 合计 | 12,930 | 100.000% |

比例等价于 8:1:1；因同一张目标表情包必须整体留在同一划分，Validation 与 Test 相差 2 条，这是保证图像不泄漏的必要离散误差。

## 核验结果

- 全部 12,930 条样本 ID 唯一。
- 3,554 张目标表情包在 Train、Validation、Test 之间无重复（图像跨划分泄漏为 0）。
- 14 个非空“情感 × 二分类标签”组合在三个划分中均至少出现 1 条样本。
- 每条 Conflict/Consistent 标签均满足版本 B 的固定规则；官方 `origin_anno` 未修改。

## 论文文字需同步之处

这是一项**实验设定变更**，但本文档没有修改论文源文件。请将论文中描述数据划分的句子改为：

> We partition the dataset into training, validation, and test sets at an approximately 8:1:1 ratio using group-aware stratified sampling over the seven official sticker-emotion categories and the binary conflict label; samples sharing the same target sticker are assigned to the same split.

若当前论文写的是“沿用 STICKERCONV 官方 Train/Validation/Test 划分”，该句应替换为上述表述。
