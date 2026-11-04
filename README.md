# 音频伪造检测实验项目

本仓库整理了 AASIST、SSL Anti-spoofing、Nes2Net 和 PellaV2 的代码，以及本机评测时保存的配置、预测表和报告。项目由本地文件重建；部分原始音频已丢失，因此保存的历史报告与能够重新运行的评测范围并不完全相同。

## 目录

| 目录 | 内容 |
| --- | --- |
| `aasist-main/` | AASIST 原始源码、较小的预训练权重、改进的下载代码和 AI 音频检测扩展 |
| `SSL_Anti-spoofing/` | SSL Anti-spoofing 源码 |
| `Nes2Net_ASVspoof_ITW/` | Nes2Net 源码 |
| `pellav2/` | PellaV2 推理与评测代码、模型配置及评测结果 |

## 当前评测状态

PellaV2 使用原权重、发布代码中的 0.5 阈值、中心 4 秒音频、固定随机种子 20261102，在每个数据集分别抽取 1000 条真实和 1000 条伪造音频。重测结果位于 `pellav2/retest/`：CVoiceFake_small 整体准确率 95.0%，MLAAD-tiny 98.4%。MLAAD Griffin-Lim 专项抽取 200 条真实与 200 条伪造，整体准确率 72.25%，伪造检出率 45.0%。这些都是抽样结果，不代表数据集全量准确率或实际业务准确率。

历史 In-The-Wild 和 ASVspoof 2019 LA 结果位于 `pellav2/`，其原始音频未恢复，目前不能重新运行对应评测。

## 本仓库未包含的文件

大型模型权重、CVoiceFake_small 与 MLAAD-tiny 原始音频、In-The-Wild 与 ASVspoof 2019 LA 原始音频均未提交。PellaV2 权重文件 `pellav2_detector.pt` 约 2.32 GB；下载来源为 [PellaV2 模型页](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector)。部分历史 CSV 记录了评测机器上的绝对路径，在其他电脑上重新评测前，需要准备数据并重新生成清单路径。

本仓库包含第三方项目的源码；请保留各子目录内原有许可证和声明。原始项目：[AASIST](https://github.com/clovaai/aasist)、[SSL Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)、[Nes2Net](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW)。
