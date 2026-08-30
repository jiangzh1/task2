# Step 02 EmoVoice 环境预检报告

## 已完成

- 已生成固定种子 `20260810` 的 100 条技术试点清单。
- 七类情感均达到计划配额；其中 LLM 冲突候选 23 条、非冲突候选 77 条。
- 已定位并核对 EmoVoice 官方仓库。
- 已将官方代码固定到提交 `5285cb891611cf1ee2d9bd07b931cd3cf967cd64`。
- 官方代码 ZIP SHA-256：`BBDF889D2E13D600BF3541C4F2B7A37AFA3080EA43C213DCA5A830A237AAD982`。
- 官方代码已解压到 `code/EmoVoice/`。

## 官方许可证

- 代码：MIT License。
- 预训练模型：CC-BY-NC，原因是训练数据包含 Emilia。
- 本项目必须保持非商业研究用途，并在数据集说明和论文中记录该限制。

## 运行环境问题

1. 系统 Python 为 3.10.12，满足官方 Python 3.10 建议。
2. 当前系统 Python 没有 `torch`、`torchaudio`、`transformers` 或 `soundfile`。
3. 系统没有 Conda。
4. `python3 -m venv` 因缺少 `python3.10-venv` / `ensurepip` 创建失败。
5. 当前 pip 默认镜像无效；显式使用标准清华镜像仍无法获得包，说明容器 PyPI 出网受限。

## 官方推理脚本适配风险

官方 `inference_EmoVoice.sh` 不是开箱即用配置：

- Qwen、CosyVoice 和 EmoVoice 权重路径均为占位符。
- 默认 `CUDA_VISIBLE_DEVICES=2`，但当前容器只有 GPU 0 和 1 可见。
- `repetition_penalty` 与 `dataset_sample_seed` 在日志路径中被引用但未定义。
- 官方输出采样率为 22,050 Hz；论文数据要求 16,000 Hz，必须增加可审计的重采样步骤。
- 推理输入需要 `source_text`、`target_text`、`emotion_text_prompt`、`neutral_speaker_wav` 等字段，当前试点清单仍需转换适配器和参考说话人资产。

## 当前阻塞项

在实际生成第一条音频之前，需要：

1. 安装 `python3.10-venv`，或提供已有可用的 Python/Conda 环境。
2. 获得可访问的 Python 包源。
3. 下载三组模型资产：Qwen2.5-0.5B、CosyVoice-300M-SFT、EmoVoice 0.5B checkpoint。
4. 确定中性参考说话人音频及其许可证；RAVDESS/CREMA-D 可作为候选，但必须先建立说话人和情感映射协议。
5. 编写并测试 pilot JSONL 到 EmoVoice 推理 JSONL 的转换器。

## 状态

步骤 02 尚未完成。样本准备与代码预检完成，环境安装、权重准备、推理适配和音频质量验证仍待执行。
