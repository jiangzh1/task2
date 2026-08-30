# Step 01 数据清单与标签基线：过程记录

## 目标

把现有 Cleaned 数据转换为稳定、可追踪、不会静默丢样本的数据清单，并将 VADER/LLM 输出明确降级为“弱标签候选”。本步骤不覆盖 `data/` 下的任何文件，不合成语音，也不把弱标签当作论文真值。

## 输入与原则

- Cleaned JSONL 是样本集合与顺序的规范来源。
- Refined JSONL 仅按 `session_id` 关联，用于保留弱标签候选和错误状态。
- 每条样本使用 `<split>:<session_id>` 作为唯一 `sample_id`，避免三个划分内部编号重复。
- 当前任务目标定义为会话中最后一个带表情包的轮次；此前轮次构成历史上下文。
- 正式标签字段 `verified_conflict` 和 `verified_emotion` 在验证前保持空值。
- 音频字段预先定义，但当前状态统一为 `not_created`。

## 计划检查

1. 为三套划分生成稳定排序的 manifest。
2. 校验目标图片存在性。
3. 标记 Refined 的 available、missing、duplicate 和 llm_error 状态。
4. 统计最终目标轮次的七类情感和冲突候选分布。
5. 检查跨划分的完整对话、当前文本、目标图片及文本—图片组合重叠。
6. 记录所有输入文件 SHA-256，保证后续结果可追溯。

## 预期产物

- `scripts/build_manifest.py`
- `artifacts/manifest_train.jsonl`
- `artifacts/manifest_validation.jsonl`
- `artifacts/manifest_test.jsonl`
- `artifacts/manifest_summary.json`
- `reports/FINAL.md`
