# 全量语音 v3：短分段与双 GPU 队列（2026-08-31）

## 变更目的

v2 的 20 词分段仍会出现大量 3,000 audio token 超限。单条失败尝试约耗时 220 秒，导致有效吞吐极低。v3 仅对未成功部分再次切分为最多 10 词，并使用双 GPU 并发。

## 数据安全与断点续跑

- v2 已成功的 4,815 个分段与 2,318 个最终样本通过硬链接复用；原 v2 目录不修改、不覆盖。
- v3 新队列共 107,150 个分段；成功分段状态由 SQLite 保存。
- 两个 worker 通过 SQLite `BEGIN IMMEDIATE` 原子领取任务，互不重复。每个 worker 的未完成租约在正常停止时会自动归还队列。
- worker PID 分别记录在 `artifacts/full_tts_pp_hashsafe_v3_multigpu/pids/worker_gpu0.pid` 和 `worker_gpu1.pid`。需要释放一张 GPU 时，对对应 PID 发送 `TERM`；另一 worker 和已完成音频不受影响。

## 当前运行

- `worker_gpu0` 绑定 GPU 0，`worker_gpu1` 绑定 GPU 1。
- 两者已分别领取 8 个不同分段并启动 EmoVoice 推理；各自日志为 `worker_gpu0.log` 与 `worker_gpu1.log`。
- 队列维护命令延迟导入 PyTorch 并固定为 CPU，避免管理进程抢占 GPU 运行时；只有实际 EmoVoice 推理进程初始化 GPU。
