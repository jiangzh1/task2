# 第一阶段端到端编排当前状态

模块一、双流/VQ、DDPM 前向加噪、真实 diffusers U-Net/CA-FM 与第一阶段总损失已经连成同一计算图。小型 U-Net 测试中，输出形状和损失均正常，模块一、conditioner、U-Net、CA-FM 全部获得梯度。

完整 SD1.5 U-Net 上的端到端测试也已通过：batch=2、FP16、`[2,4,64,64]` latent，峰值显存约 3137.21 MiB；总损失与预测均有限，模块一、conditioner、内容 token 适配器、CA-FM 均获得梯度，冻结的 U-Net 参数没有梯度。

该结果仍是合成特征下的工程连通性验证。全量真实语音特征和训练收敛尚未完成。
