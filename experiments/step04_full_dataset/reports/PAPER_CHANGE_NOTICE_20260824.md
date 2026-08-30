# 本轮实验引出的正文修改通知

只读检查对象：`D:\硕士\U盘备份\正文存档.docx`。本轮未修改 DOCX。

## 1. Dataset Construction 中的 TTS 参考语音描述

当前英文段落包含：`we employ ... EmoVoice`，并写道 `RAVDESS and CREMA-D are used to build a reference speaker library`。

这与正式实现不一致：早期 RAVDESS/CREMA-D 域外参考方案效果平淡，已经停用；正式方案是 EmoVoice-PP 加官方域内中性参考语音，并按 Train/Validation/Test 隔离参考声线。

建议将相关两句改为：

> Specifically, we employ the emotion-controllable text-to-speech model EmoVoice-PP to synthesize speech, with the target emotion controlled by the official seven-class sticker emotion label. To reduce reference-domain mismatch and avoid split-level speaker leakage, we use in-domain neutral reference speech released with EmoVoice-PP and assign disjoint reference voices to the training, validation, and test sets.

引用文献键是否仍使用 `yang2025emovoice` 需按 EmoVoice-PP 官方论文/仓库最终核对。

## 2. Dataset Construction 中的图像预处理描述

当前句子：`All sticker images are uniformly adjusted to 512×512 resolution.`

正式 VAE 预处理已经具体化为：透明通道合成白色背景、转 RGB、保持长宽比缩放、白边补齐到 512×512，不裁掉 sticker 内容。

建议改为：

> All sticker images are converted to RGB by compositing transparent regions onto a white background, resized while preserving the aspect ratio, and padded to a resolution of 512×512.

## 3. 数据划分防泄漏描述

当前句子：`Samples sharing the same target sticker are assigned to the same split.`

该表述只覆盖“相同记录/路径”，不足以解释同图异路径。实际审计发现旧路径分组划分有 33 个字节相同图像组跨集合，现已修复。

建议替换为：

> We compute the SHA-256 hash of each target sticker file and assign byte-identical images to the same split, including duplicate images stored under different paths.

## 4. 方法段落中的独立拼写错误

模块二英文段落存在 `SSubsequently`，应改为 `Subsequently`。该错误与本轮实验无关，但属于可以直接更正的正文笔误。

## 5. 表格

hash-safe 版本的完整统计表见 `STATISTICS_TABLE_HASHSAFE.md`。总划分仍为 10,344/1,294/1,292，但部分情感单元格相对旧路径划分发生变化，正文应使用新表。
