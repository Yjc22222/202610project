# AASIST：AI 语音检测应用

本模块使用原有 AASIST.pth，不训练或修改权重。检测范围为 AI 合成、声音转换/克隆的语音迹象；“未检出 AI 伪造”不代表文件未被剪辑、拼接或篡改。

## 环境

使用已安装 PyTorch 的解释器。推理依赖 torch、torchaudio、numpy、soundfile；旧 main.py 的训练依赖继续保留。WAV/FLAC 可直接读取。MP3 等格式取决于 soundfile 解码支持；M4A/AAC 等解码失败时可使用 FFmpeg。将 ffmpeg 加入 PATH，或在 config/inference.json 的 audio.ffmpeg 中填写 ffmpeg.exe 的绝对路径。格式不支持时明确返回错误，不判为真实。

PowerShell 中，在项目目录执行下面每条命令时分别按回车。本文以已配置的解释器为例：

```powershell
$pythonExe = 'C:\Users\27714\Documents\Codex\2026-10-27\new-chat\outputs\pytorch\python\python.exe'
```

## 直接检测

项目中提供经过验证集校准的 artifacts/threshold.json 后，可以直接运行：

```powershell
& $pythonExe .\inference.py --input 'D:\音频\示例.wav' --output .\artifacts\detection.json
```

批量检测：

```powershell
& $pythonExe .\inference.py --input 'D:\音频' --recursive --output .\artifacts\batch.json
```

程序同时保存 JSON（含片段信息）和 CSV。prediction=spoof 表示疑似 AI 伪造，bonafide 表示未检出 AI 伪造。raw_score 越高越支持 bonafide；原始分数不是概率。status=error 时 prediction 为空，查看 error，不能将失败视为正常音频。存在文件错误时退出码为 1，启动/配置错误为 2。

尚无阈值时可加 --scores-only，只计算分数、不输出分类。可用 --device cpu 或 --device cuda 指定设备。auto 优先使用 CUDA，CPU 较慢。

## 处理流程

默认保持幅度，不降噪。soundfile 解码为 float32 后转单声道，并用 torchaudio 重采样至 16 kHz。默认最短 0.25 秒、最长 1800 秒、单文件最大 256 MB，限制均在配置中列明；超限明确报告失败。

输入窗口固定 64600 个采样点（4.0375 秒）；短音频重复补齐。长音频默认滑动窗口，步长 32300，额外覆盖尾部。默认对各片段 bonafide logit 取均值作为文件分数。可疑窗口分数不等于精确的单词级 AI 伪造定位。均值可能弱化极短伪造片段；若改用 q10 或其他汇总设置，必须重新校准并独立验证。

mode=first 仅用于对齐原项目基线，它只检查音频开头。正式默认为 sliding。

## 自己校准阈值

CSV 必需列为 path,label；label 只能为 bonafide 或 spoof。相对路径相对于 CSV 所在目录解析。推荐增加 group（原录音来源/说话人组）和 type（合成来源）。同源录音及其转码、裁剪版本必须放在同一组，不能跨验证集和测试集。

从本地 ASVspoof 2019 LA 的 dev 和 eval 分别平衡抽取每类 500 条：

```powershell
& $pythonExe .\prepare_manifests.py --dataset-root .\LA --split dev --per-class 500 --output .\artifacts\validation.csv
& $pythonExe .\prepare_manifests.py --dataset-root .\LA --split eval --per-class 500 --output .\artifacts\test.csv
```

两条命令分别执行。--per-class 0 使用整个数据分区。抽样结果只用于开发验证，完整验收需使用事先确定、覆盖实际业务场景的独立数据。

```powershell
& $pythonExe .\calibrate.py --manifest .\artifacts\validation.csv
& $pythonExe .\benchmark.py --manifest .\artifacts\test.csv --output .\artifacts\benchmark.json
```

校准默认选择验证集上平衡准确率最高的阈值，可用 --method eer 改用等错误率附近阈值。模型权重不更新。验证文件处理失败则拒绝生成新阈值，不跳过失败文件挑选样本。阈值保存在配置指定的位置，绑定权重、模型结构、音频处理、窗口、汇总方式和相关源代码版本。修改这些设置后旧阈值会被拒绝。

## 报表口径

spoof 为指标正类（原模型内部第 0 类为 spoof，第 1 类为 bonafide）。benchmark 的主指标 accuracy_on_all=正确分类数/全部输入数，失败计为不正确；accuracy_on_scored 单独报告成功分类样本的准确率。spoof_recall_on_all 的分母包含所有伪造输入，伪造文件处理失败也不能提高检出率。报表包含精确率、F1、真实语音误报率、覆盖率、混淆矩阵和失败数量。按 type 分组的结果也会保存。

独立测试发现校准集与测试集有相同文件内容、相同 group 或完全相同清单时会拒绝运行；检测不能代替人工确保同源录音及各种派生版本不跨组。EER 仅作为附加描述指标，其在测试集上的阈值不会用于实际分类。90% 的参考标志只表示本次数据上的结果，不保证未知合成器、真实业务录音或其他篡改类型的效果。

benchmark.errors.csv 保存误判和失败文件；读取失败样本不能从最终验收统计中悄悄删除。最终的类别比例、误报限制、格式范围和失败计分方式需要在业务验收前固定。
