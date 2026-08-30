# 步骤 02：情感语音生成过程记录

## 当前目标

为版本 B 的 12,930 条正式样本生成可追溯的情感语音，并完成 16 kHz 单声道处理与音频质量审计。本步骤完成前，不能把 SpchConvSti 数据集构建表述为全部完成。

## 2026-08-12 环境复检

- EmoVoice 官方代码已存在，固定提交为 `5285cb891611cf1ee2d9bd07b931cd3cf967cd64`。
- 现有 `.venv` 只是无 pip、无 torch 的空环境，不能运行推理。
- EmoVoice、Qwen2.5-0.5B 与 CosyVoice 权重尚未下载。
- 数据目录中未发现 RAVDESS 或 CREMA-D。
- 两张 RTX 3090 均空闲。
- 服务器当前可连接 PyPI 与 Hugging Face；旧报告中的“包源不可访问”结论已失效。

## 本轮执行

1. 修复独立 Python 3.10 环境并安装官方固定依赖。
2. 盘点并下载官方 EmoVoice 0.5B、Qwen 与 CosyVoice 推理资产。
3. 将官方占位推理脚本改为项目内可复现配置，固定使用 GPU 0。
4. 在参考说话人资产就绪后生成小规模音频，验证可懂度、情感控制、采样率、通道数、时长和静音比例。

## 已启动的后台任务

- `runtime/bootstrap_environment.log`：修复虚拟环境并安装 PyTorch 2.4.1 CUDA 12.1 与 EmoVoice 官方依赖。
- `runtime/download_tts_assets.log`：下载 EmoVoice 0.5B、Qwen2.5-0.5B 和 CosyVoice-300M-SFT 官方推理资产，支持断点续传。
- `runtime/download_reference_speech.log`：下载 RAVDESS 与 CREMA-D 参考语音数据并记录校验值。
- 三项任务均使用 `setsid` 在服务器后台运行，客户端断开不会中止。

## 已完成的试点输入准备

- 已从 `neutral_mismatch_stratified_811` 正式版本 B 构建 `artifacts/emovoice_pilot_28.jsonl`。
- 28 条样本完整覆盖 7 类情感 × 2 类 Conflict/Consistent，每个组合 2 条。
- 旧 `pilot_100.jsonl` 来自早期弱标签与旧划分，不再作为当前语音试点输入，但保留用于过程追溯。

## 重要限制

RAVDESS/CREMA-D 当前不在服务器，无法据此建立论文所述参考说话人库。环境与模型权重准备可先进行，但正式试点音频需要合法、可追溯的中性参考音频。本轮不会伪造 `test_human` 真人录音。

## 参考语音数据来源

- RAVDESS 从官方 Zenodo 记录 `1188976` 下载音频语音包（24 位演员、1,440 条语音；CC BY-NC-SA 4.0）。
- CREMA-D 使用完整的 7,442 条 16 kHz WAV 音频 Parquet 镜像下载，原始项目为 CheyneyComputerScience/CREMA-D（ODbL 1.0）。镜像仅用于传输与解析，报告中保留原项目来源。
- 下载完成后将只从中性语音建立参考说话人库，并做时长、静音、幅度、采样率与损坏文件检查。
