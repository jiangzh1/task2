# Step 00 实验资产盘点：过程记录

## 目标

在不改动现有数据、代码和历史产物的前提下，核对 `/data/jzh/2026/task2` 的数据完整性、已有处理流程、服务器环境与潜在复现风险，为后续数据集构建建立真实基线。

## 执行原则

- 原始 `data/` 目录只读，不覆盖 Cleaned/Refined 文件。
- 审计脚本仅读取元数据、JSONL 内容和文件存在性；不读取或修改图片像素。
- 所有新增内容写入 `experiments/step00_inventory/`。
- 论文草稿中的规模、硬件和实验结论均视为待验证假设。

## 已完成检查

1. SSH 已恢复，目标目录可访问。
2. 未发现项目级 `AGENTS.md`。
3. 项目包含 `.git`，但当前运行环境未安装 `git` 命令。
4. 已识别原始 Parquet、Cleaned/Refined JSONL、SER_Dataset 图像及现有情感处理脚本。
5. 当前容器可见 2 张 NVIDIA RTX 3090，而非论文草稿中的 4 张。
6. 已审阅 `chuli.py`、`text_emo.py` 和 `text_emo_fwq.py`，发现需要通过数据审计进一步验证的顺序、断点续传和调试残留风险。

## 本步骤产物

- `scripts/audit_inventory.py`：只读审计脚本。
- `artifacts/inventory.json`：机器可读审计结果。
- `reports/FINAL.md`：完成报告（审计运行结束后生成）。
