# 中文长音频识别精度补测

日期：2026-10-02。Windows，Ryzen 7 7840H / Radeon 780M，与英文长音频测试同一台机器。

## 两种 Whisper 实现的区别

whisper.cpp 与 faster-whisper 都运行 Whisper 模型。相同模型的核心识别能力接近，
不同实现的分段、搜索、数值精度和上下文处理仍可能改变文字结果，不能把其中一个直接称为更准确的模型。

| 项目 | whisper.cpp | faster-whisper |
|---|---|---|
| 推理引擎 | C/C++，ggml | CTranslate2，Python API |
| 权重格式 | ggml 权重，可量化 | CTranslate2 转换权重，可量化 |
| GPU 路线 | 原生构建提供 Vulkan、HIP/ROCm、CUDA 等选项 | 标准发行版 GPU 文档采用 NVIDIA CUDA；本机 AMD 使用自建 CTranslate2 |
| 项目集成 | 当前适配通过 CLI 子进程，每次重新加载模型 | 当前 Python 工作流、字幕和对齐集成较完整，可复用已加载模型 |
| 适合本项目的用途 | AMD 通用 Vulkan 备选、独立进程部署 | 保留为默认文件转写后端，沿用现有功能 |

来源：[whisper.cpp v1.9.4 文档](https://github.com/ggml-org/whisper.cpp/blob/v1.9.4/README.md)、
[faster-whisper 官方文档](https://github.com/SYSTRAN/faster-whisper)。
集成状况描述的是当前项目适配器，不是上游软件能力的完整边界。

## 数据与计分方法

使用 [AISHELL-3 官方数据仓库](https://huggingface.co/datasets/AISHELL/AISHELL-3)
`test/content.txt` 中的人工文字标注和对应真实录音；固定 revision
`f20d5db4a31fe779ef07bb1af4ea92da5c786622`。
原始文字为汉字与拼音成对标注，提取汉字，不由待测模型生成参考答案。
这是朗读语音/TTS 数据的小规模抽样，不是 AISHELL-1 标准 ASR 测试集，也不保证与模型训练数据隔离。

| 说话人 | 独立语句数 | 拼接音频时长 |
|---|---:|---:|
| SSB0693 | 22 | 91.614 秒 |
| SSB0711 | 35 | 92.070 秒 |
| SSB0716 | 36 | 90.559 秒 |
| 合计 | 93 | 274.243 秒 |

各人按数据文件顺序选取语句直到超过 90 秒；不重复语句、不额外插入静音。
44100 Hz 单声道通过多相滤波重采样为 16000 Hz PCM16。
原始来源 URL、各句时间范围、文字和最终音频 SHA256 保存在
`.cache/fixtures/chinese/manifest.json`。

主要指标为 CER = (替换字数 S + 漏字数 D + 多字数 I) / 参考字数 N，越低越好。
采用 Unicode NFKC、英文小写、去除标点和空白，然后用 OpenCC t2s 统一简繁。
同时保留不做简繁转换的 CER；不把阿拉伯数字自动展开为读音，也不做同音字宽松匹配。
总 CER 按三个音频的错误总数除以参考字总数，不平均三组百分比。
CER 不等于句子准确率，不评价标点、字幕时间戳或说话人划分。

第二组包含以“幺”朗读的电话号码，Whisper 常输出阿拉伯数字；第三组包含温度和百分数。
另报“数字格式归一 CER”：11 位数字逐位展开，四位年份逐位展开，百分号转为“百分之”，
其余整数按中文数词展开，小数按“点”逐位展开，并统一“幺/一”。
这一辅助规则针对当前语料，不能冒充覆盖所有场景的通用中文文本规范化。
原始 CER 与辅助分数均保留，避免用格式处理隐藏实际错误。

Whisper 使用多语种 large-v3-turbo：whisper.cpp 为非量化 ggml 权重，
faster-whisper 为现有 CT2 权重，分别报告 float16/int8_float16/CPU int8。
基线为 4 个 CPU 线程，显式中文、beam_size=1、best_of=1、无 VAD；
后续另测 beam=5 和关闭前文上下文的组合，GUI 关键参数组合的 best_of=5，
仍固定 temperature=0。并未声明两个引擎所有解码细节相同。
Qwen3-ASR-0.6B 使用中文、float16、max_new_tokens=2048；其模型和算法与 Whisper 不同。
sherpa 使用中文 Zipformer 14M 模型及 int8 ONNX 权重，100ms 输入帧、greedy_search，无端点重置。
中文流式模型来源：[sherpa 官方模型文档](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)。

全部推理串行；每个配置在每段音频执行一次，三段覆盖不同说话人。
这些耗时是三段输入的平均值，不是同一输入重复三次的稳定中位数。
whisper.cpp 的处理时间按原生日志 total-load 近似计算；外部总时间和加载时间另外保存。
其他后端的处理时间不含模型加载，但包含音频读取及完整结果生成。
不同实现的计时边界不完全等价。历史内核缓存沿用；不称为全新机器冷启动测试。
流式表格采用尽快送帧的处理吞吐，不能把它当作麦克风识别延迟。
Qwen CPU 的英文性能已有记录，本轮中文精度使用 GPU，没有重测 Qwen CPU 中文速度。

## 结果

39 项实际识别全部完成：13 个配置 × 3 段长音频，参考文字共 956 字。
计分自检、音频时长、语句唯一性和 SHA256 校验通过；项目结构检查 9 项通过。

| 后端与配置 | 平均处理秒数 | 原始 CER | 数字归一 CER | 替换 S | 漏字 D | 多字 I |
|---|---:|---:|---:|---:|---:|---:|
| whisper.cpp CPU，beam=1 | 65.043 | 8.05% | 4.71% | 76 | 1 | 0 |
| whisper.cpp Vulkan，beam=1 | 10.094 | 8.05% | 4.71% | 76 | 1 | 0 |
| whisper.cpp ROCm，beam=1 | 9.285 | 8.16% | 4.71% | 77 | 1 | 0 |
| whisper.cpp ROCm，beam=5 | 11.308 | 7.85% | 4.39% | 74 | 1 | 0 |
| faster-whisper CPU int8，beam=1，保留上下文 | 30.679 | 19.46% | 16.00% | 78 | 1 | 107 |
| faster-whisper ROCm float16，beam=1，保留上下文 | 13.205 | 36.30% | 32.95% | 77 | 1 | 269 |
| faster-whisper ROCm int8_float16，beam=1，保留上下文 | 10.958 | 20.08% | 16.63% | 79 | 1 | 112 |
| faster-whisper ROCm float16，beam=5，保留上下文 | 15.534 | 27.20% | 23.74% | 73 | 1 | 186 |
| faster-whisper ROCm float16，beam=1，关闭上下文 | 10.380 | 8.05% | 4.60% | 76 | 1 | 0 |
| faster-whisper ROCm float16，beam=5，关闭上下文 | 12.247 | 8.37% | 4.81% | 79 | 1 | 0 |
| faster-whisper ROCm int8→int8_float16，beam=5，关闭上下文 | 13.244 | 8.68% | 5.13% | 82 | 1 | 0 |
| Qwen3-ASR-0.6B ROCm float16 | 43.472 | 3.35% | 3.35% | 31 | 0 | 1 |
| sherpa-onnx 中文 Zipformer 14M，CPU int8 | 2.485 | 23.22% | 23.01% | 108 | 106 | 8 |

S/D/I 为原始 CER 的计数；数字归一后的详细计数在 JSON 各组 `number_format_score` 中。
本组所有配置的简繁严格 CER 与原始 CER 相同，差异主要来自数词写法、同音字和重复文本。
whisper.cpp CPU/Vulkan/ROCm beam=1 的外部平均总时间分别为 66.383 / 11.703 / 12.030 秒；ROCm beam=5 为 14.739 秒。
表中使用其原生 total-load 时间，与 Python 后端计时边界仍存在差别。

### 参数对照与异常

开启前文上下文时，faster-whisper 的部分输出在音频末尾重复：
SSB0711 float16 beam=1 输出了多次“我现在出门了”，额外插入 108 字；
SSB0716 float16 beam=1 插入 161 字，CPU int8 插入 105 字。
把 float16 的 beam 从 1 改为 5、仍保留上下文，合计插入 186 字，未消除问题。

关闭前文上下文的 float16 beam=1、float16 beam=5 和 int8 beam=5 对照均没有这种长串重复，
三个配置总插入字数为 0。不得把这组有限结果视为彻底杜绝幻觉的保证。
当前 GUI 保存的设置本来就是 beam=5、best_of=5、condition_on_previous_text=False，
模型精度索引 0 对应 int8。实际测量确认 GPU 请求 int8 后的有效计算类型为 int8_float16。
该关键配置组合数字归一 CER 为 5.13%，平均处理 13.244 秒；不是异常基线的 20%～36%。
本轮线程数固定为 4、语言固定中文、temperature=0、无 VAD，未声称完整复制 GUI 全部参数。

### 选择建议

- 保留 faster-whisper 为现有功能默认后端，并保持当前关闭前文上下文的设置。
  float16 + beam=1 + 关闭上下文在此样本为 4.60% / 10.380 秒，
  可以作为速度优先候选；本组提高 beam 并未改善其精度，不据此推出普遍结论。
- 添加 whisper.cpp Vulkan 备选以改善 AMD 部署兼容性；若维护 ROCm 构建，
  本组 beam=5 为 4.39% / 11.308 秒。与调好参数的 faster-whisper 差距有限，
  没有依据仅因更换推理引擎就全面重写现有工作流。
- Qwen3-ASR-0.6B 在此小样本的错误最少：32/956 字，数字归一 CER 3.35%，
  平均处理 43.472 秒，约为当前 int8 GUI 参数组合的 3.28 倍。
  建议增加中文精度优先选项，字幕使用前需补齐对齐；当前适配返回纯文本。
  官方另有 [Qwen3-ForcedAligner](https://github.com/QwenLM/Qwen3-ASR)。
- 中文 Zipformer 14M 的流式吞吐最快，但漏字 106 字，数字归一 CER 23.01%，
  不建议作为精度优先默认。该模型很小，结果不能代表所有 sherpa-onnx 中文模型；
  应另选更强中文模型再评估生产适用性。

错误例子：SSB0693 的“修鞋奶奶婉拒捐款”，两种 Whisper 的部分配置识别为
“休闲奶奶玩具捐款”，Qwen 在本例识别正确；但 Qwen 仍把“工艺”写成“公益”。
SSB0711 的“天鹅绒”在 Qwen/Whisper 中均出现同音字错误，说明低总 CER 仍需要人工核对。

原始输出、各组 CER 与耗时保存在 `.cache/reports/chinese/`
及 `.cache/reports/chinese-comparison.json`。

## 复现

源代码位于 `src/project_tools/prepare_chinese_fixtures.py`、
`benchmark_chinese_suite.py`、`score_chinese.py`；使用既有隔离实验环境，不修改主项目依赖。
评分环境新增 `opencc-python-reimplemented==0.1.7`。

```powershell
.venv/Scripts/python.exe src/project_tools/prepare_chinese_fixtures.py
.venv/Scripts/python.exe src/project_tools/benchmark_chinese_suite.py
.venv/Scripts/python.exe src/project_tools/benchmark_chinese_suite.py --mitigations
.cache/experiments/streaming-env/Scripts/python.exe src/project_tools/score_chinese.py
```

套件跳过已存在的报告，重新测量前需明确移走对应报告。
权重路径当前采用这台机器已安装的 turbo 模型；迁移机器时需调整路径。

这组输入以普通话、干净单人朗读为主，尚不能代表会议、噪声、口音、中英混说，
也不足以推出统计显著的引擎准确率排名。进一步生产判断应采用用户真实音频及独立人工校对文字。
