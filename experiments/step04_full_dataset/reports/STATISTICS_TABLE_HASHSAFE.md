# 正式版本 B 数据统计表（SHA-256 安全划分）

以下表格基于独立复核通过的 `neutral_mismatch_hash_stratified_811`，数字完整，可直接用于正文。Emotion 行表示官方表情包七类情感；Consistent/Conflict 由文本三极性与表情包三极性是否相同确定，中性与正/负不匹配计为 Conflict，Surprise 在极性映射中归为 Positive。

```latex
\begin{table}[t]
\small
\centering
\caption{Statistics of the proposed \textit{SpchConvSti} dataset.}
\label{tab:statistics}
\begin{tabular}{c|ccc|ccc|c}
\toprule
\multirow{2}{*}{Emotion}
& \multicolumn{3}{c|}{Emotion-Consistent}
& \multicolumn{3}{c|}{Emotion-Conflict}
& \multirow{2}{*}{Total}\\
\cmidrule(lr){2-7}
& Train & Val & Test
& Train & Val & Test
&\\
\midrule
\textit{Happiness} & 6310 & 789 & 789 & 534 & 67 & 67 & 8556\\
\textit{Sadness}   & 126 & 16 & 16 & 1067 & 133 & 133 & 1491\\
\textit{Anger}     & 44 & 6 & 5 & 414 & 52 & 52 & 573\\
\textit{Surprise}  & 158 & 20 & 20 & 59 & 7 & 7 & 271\\
\textit{Disgust}   & 6 & 1 & 1 & 16 & 2 & 2 & 28\\
\textit{Fear}      & 10 & 1 & 1 & 122 & 15 & 15 & 164\\
\textit{Neutral}   & 273 & 34 & 34 & 1205 & 151 & 150 & 1847\\
\midrule
Total & 6927 & 867 & 866 & 3417 & 427 & 426 & 12930\\
\bottomrule
\end{tabular}
\end{table}
```
