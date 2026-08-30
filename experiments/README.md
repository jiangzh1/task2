# Task2 实验执行路线

所有实验步骤均放在 `experiments/` 下。每个步骤至少包含：

- `reports/PROCESS.md`：执行目标、过程、假设、命令和中间发现
- `reports/FINAL.md`：完成状态、结果、风险、复现方法和下一步门禁
- `scripts/`：该步骤的可复现代码
- `artifacts/`：机器可读统计、配置和日志摘要

## 阶段划分

1. `step00_inventory`：资产与环境审计（已完成）
2. `step01_dataset_manifest`：规范数据清单、标签与泄漏审计
3. `step02_tts_pilot`：约 100 条情感语音合成、16 kHz 处理与质量验证
4. `step03_dataset_build`：全量 SpchConvSti 构建与版本冻结
5. `step04_dataset_analysis`：类别、冲突、语音、图像和对话统计
6. `step05_eval_harness`：统一生成、缓存、自动指标和人工评价工具
7. `step06_baselines`：文本、SER 引导和直接语音驱动基线
8. `step07_method_stage1`：双模态对齐、上下文补全与冲突推理
9. `step08_method_stage2`：双流条件注入、CA-FM 与风格码本
10. `step09_method_stage3`：潜空间奖励和轨迹校正
11. `step10_main_experiments`：主结果、分组结果、多随机种子
12. `step11_ablation`：模块与输入消融
13. `step12_human_case_analysis`：人工评价、成功/失败案例和论文图表

每个步骤只有在 `FINAL.md` 明确写明“通过门禁”后才进入下一步；若发现数据或方法问题，允许回退并记录版本变化。
