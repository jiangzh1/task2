# 论文公式—代码映射

本映射依据本地只读副本 `D:\硕士\U盘备份\正文存档.docx` 的 Method 章节建立，未修改正文。

## 模块一：上下文协同意图推理与语义补全

| 正文定义 | 代码位置 | 实现状态 |
|---|---|---|
| WavLM 特征 A 与 OpenSMILE/BiLSTM 韵律 R 的残差适配公式 | `src/spchconvsti/stage1.py::ProsodyEnhancedSpeechAdapter` | 已实现；真实特征待音频完成 |
| 两层投影头 g_text、g_speech 与 z_y、z_s | `stage1.py::ProjectionHead` | 已实现 |
| 双向 InfoNCE，L_align | `stage1.py::bidirectional_info_nce` | 已实现并测试 |
| Whisper 秒级时间戳转 WavLM 帧区间 | `stage1.py::seconds_to_frame_spans` | 已实现 |
| 词级时间窗均值对齐 | `stage1.py::WordTimestampAligner` | 已实现精确均值测试 |
| e_y、e_s 与 Delta_global=e_y-e_s | `SpeechTextConflictReasoner.forward` | 已实现 |
| d_i、sim_i、omega_i、Delta_local | `SpeechTextConflictReasoner.forward` | 已实现，含 mask |
| Delta=[Delta_global;Delta_local] | `SpeechTextConflictReasoner.forward` | 已实现 |
| 当前文本查询上下文的 cross-attention | `context_attention` | 已实现 |
| alpha=MLP(c_bar)，Delta_tilde=alpha⊙Delta | `context_arbitration` | 已实现；使用 Sigmoid 将仲裁范围限制在 [0,1] |
| Transformer 深层交互与 h_joint | `fusion_transformer` | 已实现为融合种子加三个模态 token |

## 模块二：冲突感知条件特征注入

| 正文定义 | 代码位置 | 实现状态 |
|---|---|---|
| 内容/情感双流投影 | `src/spchconvsti/stage2.py::DualStreamConditionProjector` | 已实现 |
| 最近邻情感风格码本 | `EmotionStyleQuantizer` | 已实现 |
| L_code 与 stop-gradient | `EmotionStyleQuantizer.forward` | 已实现 |
| gamma_adapt、beta_adapt | `ConflictAwareFeatureModulation.forward` | 已实现 |
| F'=(1+gamma)⊙F+beta | `ConflictAwareFeatureModulation.forward` | 已实现 |
| SD1.5 cross-attention 与 U-Net 层级挂载 | 尚无 | 未实现；需要真实 diffusers/SD1.5 集成 |

## 模块三：一致性评估与轨迹自校正

| 正文定义 | 代码位置 | 实现状态 |
|---|---|---|
| Tweedie 的 z0 估计 | `src/spchconvsti/stage3.py::tweedie_endpoint_estimate` | 已实现 |
| sem/emo/atm 三个投影奖励头 | `MultiDimensionalLatentReward` | 已实现 |
| lambda_sem、lambda_emo、lambda_atm 动态调度 | `dynamic_reward_weights` | 已实现并验证和为 1 |
| Constant-Noise 梯度 | `constant_noise_trajectory_correction` | 已实现，噪声预测显式 detach |
| DDIM 临时状态及 eta_t 梯度修正 | `ddim_temporary_step`、`constant_noise_trajectory_correction` | 已实现 |
| 真实预训练 E_lat/Latent CLIP 接入 | 抽象 `latent_encoder` 参数 | 未实现；等待选择并部署论文指定模型 |

## 损失函数

`src/spchconvsti/losses.py` 已实现 L_content、第一阶段 L_total 和第二阶段 margin preference loss。
