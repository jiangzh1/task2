# 论文表 \texttt{tab:statistics} 的可用 LaTeX 数据

本报告对应论文中“Statistics of the proposed \textit{SpchConvSti} dataset.”表。每一个数字按官方 `origin_anno` 七类表情包情感、二分类标签与官方 Train/Val/Test 划分逐条汇总而得。

## 版本 A：正负冲突

仅 Positive/Negative（顺序不限）记为 Conflict。该版本的原表已可完整填充；在论文正文将此版本作为主设定时，保留原来的 caption 与 `\label{tab:statistics}`。

| Emotion | Consistent Train | Val | Test | Conflict Train | Val | Test | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Happiness | 7023 | 655 | 761 | 100 | 11 | 6 | 8556 |
| Sadness | 662 | 56 | 44 | 596 | 51 | 82 | 1491 |
| Anger | 176 | 15 | 15 | 304 | 28 | 35 | 573 |
| Surprise | 219 | 30 | 17 | 5 | 0 | 0 | 271 |
| Disgust | 13 | 0 | 4 | 7 | 2 | 2 | 28 |
| Fear | 43 | 3 | 4 | 103 | 7 | 4 | 164 |
| Neutral | 1534 | 142 | 171 | 0 | 0 | 0 | 1847 |
| Total | 9670 | 901 | 1016 | 1115 | 99 | 129 | 12930 |

## 版本 B：极性不一致（含中性）

任意不同极性记为 Conflict。若采用此版作为主设定，应把 caption 改为 “Statistics of the proposed \textit{SpchConvSti} dataset under the neutral-inclusive conflict definition.”，并换用不同的 label，避免与版本 A 的引用冲突。

| Emotion | Consistent Train | Val | Test | Conflict Train | Val | Test | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Happiness | 6552 | 614 | 722 | 571 | 52 | 45 | 8556 |
| Sadness | 131 | 10 | 17 | 1127 | 97 | 109 | 1491 |
| Anger | 46 | 5 | 4 | 434 | 38 | 46 | 573 |
| Surprise | 163 | 22 | 13 | 61 | 8 | 4 | 271 |
| Disgust | 7 | 0 | 1 | 13 | 2 | 5 | 28 |
| Fear | 9 | 1 | 2 | 137 | 9 | 6 | 164 |
| Neutral | 289 | 32 | 20 | 1245 | 110 | 151 | 1847 |
| Total | 7197 | 684 | 779 | 3588 | 316 | 366 | 12930 |

## 使用提醒

两表只可二选一放入正文主表；另一版应作为敏感性实验或附录，不应以同一数据集的两个独立结果并列主张。
