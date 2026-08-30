# 两种冲突定义的数据集版本

## 可互换版本

本目录保留相同的 12,930 条样本、相同的官方七类表情包标签、相同的文本极性与相同的 Train/Validation/Test 划分；两个版本仅改变 `conflict_label` 的判定规则。

| 版本目录 | 规则 | 适用解释 |
|---|---|---|
| `policy_variants/strict_pn/` | 仅 Positive/Negative（顺序不限）为 Conflict | 将“情感方向相反”严格定义为冲突。 |
| `policy_variants/neutral_mismatch/` | 任何文本—表情包极性不同均为 Conflict | 将“中性与非中性不匹配”也视为冲突。 |

两种版本中，`Positive/Positive`、`Negative/Negative` 与 `Neutral/Neutral` 都是 Consistent。第二个版本额外把 `Positive/Neutral`、`Neutral/Positive`、`Negative/Neutral`、`Neutral/Negative` 标为 Conflict。

## 两版最终数据分布

### 版本 A：正负冲突（`strict_pn`）

| 划分 | Conflict | Consistent | 总计 | Conflict 占比 |
|---|---:|---:|---:|---:|
| Train | 1,115 | 9,670 | 10,785 | 10.34% |
| Validation | 99 | 901 | 1,000 | 9.90% |
| Test | 129 | 1,016 | 1,145 | 11.27% |
| 合计 | 1,343 | 11,587 | 12,930 | 10.39% |

### 版本 B：极性不一致（含中性，`neutral_mismatch`）

| 划分 | Conflict | Consistent | 总计 | Conflict 占比 |
|---|---:|---:|---:|---:|
| Train | 3,588 | 7,197 | 10,785 | 33.27% |
| Validation | 316 | 684 | 1,000 | 31.60% |
| Test | 366 | 779 | 1,145 | 31.97% |
| 合计 | 4,270 | 8,660 | 12,930 | 33.02% |

两版均经样本 ID 唯一性、总数和冲突规则逐条核验：12,930 条记录全部通过，无规则错误。

## 使用限制

- 两个版本是可替换的实验设定，不能混合训练或把二者的统计结果放在同一列比较。
- 推荐将一版作为正文主设定，另一版作为标签定义敏感性实验；在论文中应明确写出具体规则。
- 原始官方 `origin_anno` 与文本极性均不因版本切换而变化。
