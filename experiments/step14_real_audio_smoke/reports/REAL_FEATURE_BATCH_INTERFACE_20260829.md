# 真实特征批量接口实现记录（2026-08-29）

## 目的与边界

本实现把 step14 保存的真实音频特征缓存转换为模块一 `MultimodalFeatures` 批量输入，仅用于数据接口和小样本连通性验证，不构成正式训练、实验或结果。

## 新增实现

- `experiments/step07_method_stage1/src/spchconvsti/real_features.py`
  - 使用 `torch.load(..., weights_only=True)` 安全读取 step14 特征缓存；
  - 检查 RoBERTa 词特征、Whisper 词时间戳、WavLM 帧特征与 OpenSMILE LLD 的长度一致性；
  - 对文本、上下文、声学与韵律序列分别 zero-padding，并生成布尔 mask；
  - 以 WavLM 50 Hz 特征帧率将 Whisper 秒级词边界转为 `[start,end)` 帧索引；
  - 在返回前调用既有 `MultimodalFeatures.validate()`，拒绝任何越界或 padding 不一致的 batch。
- `experiments/step14_real_audio_smoke/scripts/validate_real_feature_batch.py`
  - 对实际 `.pt` 缓存执行批量契约验证，并报告输入形状和有效词帧区间。
- `extract_real_features.py` 的后续缓存将记录音频时长及 `wavlm_frame_rate=50.0`。正在运行的进程不受脚本更新影响，已有缓存默认按同样的 WavLM 50 Hz 帧率处理。

## 运行情况

已启动七类情感各两条（共 14 条）的 CPU 后台提取，首条缓存会自动跳过，后续已开始生成。批量契约验证已提交；它应在特征提取进程释放 CPU 后读取当前可用缓存完成。

## 后续

14 条缓存完成且批量契约验证通过后，才可把该 loader 接入模块一的小样本前向/反向连通测试。该测试只能证明真实输入可被模型消费，不能替代完整数据上的正式 Stage 1 训练。
