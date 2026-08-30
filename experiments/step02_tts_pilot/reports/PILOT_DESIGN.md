# EmoVoice 语音试点设计

## 试点样本

从正式版本 B 数据集的 14 个“七类官方表情包情感 × Conflict/Consistent”组合中各选 2 条，共 28 条。试点只验证技术链路和音频质量，不用于论文最终统计。

## 语音情感依据

合成语音的目标情感采用官方表情包七类标签 `origin_anno`，因为论文中的语音代表需要生成的真实情感方向；文本极性只用于 Conflict/Consistent 标注。对于 Conflict 样本，故意让文本字面极性与合成语音/表情包极性不同，以构造论文方法需要识别的跨模态情感不一致。

| 官方标签 | EmoVoice 自然语言控制提示 |
|---|---|
| Happiness | clear happiness, warm enthusiasm, and an upbeat lively tone |
| Sadness | genuine sadness, subdued energy, and a soft sorrowful tone |
| Anger | controlled anger, firm emphasis, and tense forceful delivery |
| Surprise | positive surprise, bright excitement, and an animated rising intonation |
| Disgust | clear disgust, aversion, and a restrained repulsed tone |
| Fear | fear and unease, with cautious tension and slightly trembling delivery |
| Neutral | a calm neutral tone with natural pacing and no strong emotion |

Surprise 按用户已确认的 Positive 方向处理，因此使用“positive surprise”而不是惊恐式意外。

## 参考说话人

试点清单中的 `neutral_speaker_wav` 当前保持待填。RAVDESS/CREMA-D 下载并审核后，将选择无损坏、时长适中、低静音比例的中性语音，并保证参考说话人不跨正式数据划分造成身份泄漏。
