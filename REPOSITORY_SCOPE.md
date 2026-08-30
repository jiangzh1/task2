# SpchConvSti 论文代码归档范围

本分支只归档本论文可复现所需的项目文件：

- 各实验步骤的自有源码、脚本、配置、README 和中文过程报告；
- 正式版本 B、按图像 SHA 分组的 `neutral_mismatch_hash_stratified_811` Train/Validation/Test JSONL 清单；
- `data/StickerConv_Refined_{Train,Test,Vail}.jsonl` 原始文本/标注清单。

不归档下列内容：

- 音频、表情图片、运行缓存、特征缓存、模型权重、checkpoint、虚拟环境；
- `data/SER_Dataset`、`data/处理过程`、parquet 文件等非本论文正式数据或无法在普通 GitHub 仓库安全管理的大型数据；
- 上游第三方代码的完整副本。

完整音频和大型资源仍保留在服务器指定目录，由项目脚本和报告说明其获取/生成方式。
