# 后端评估与 GUI 决策

日期：2026-10-02。设备：Windows，AMD Radeon 780M Graphics。
本页保留初始短音频实验；后续 91.465 秒识别、60 秒音乐分离及 ROCm 对照见
[长音频复测](LONG_AUDIO_BENCHMARK.md)，后端选择以该次复测结论为准。
中文 93 条语句、39 项实际识别及 CER 对照见 [中文精度补测](CHINESE_ASR_EVALUATION.md)，
包括当前 GUI 关键解码配置、数字格式差异和前文上下文造成的末尾重复对照。
这份记录保留初始接口统一、community-1 升级和候选后端评估。**2026-10-03 已完成 GUI 集成**：
whisper.cpp / Qwen / sherpa 可选入口、原生流式修订、Qwen 对齐字幕、Roformer 分离页面及完整窗口验证，见
[GUI 使用说明与验收](GUI_BACKENDS.md)。下文关于“尚未接入 GUI”的描述属于 10 月 2 日历史状态。

## 决策

保留 PySide6 / QFluentWidgets，暂不重做 GUI。实验中需要变化的是后端能力、参数和结果展示：
Vulkan 可以返回带时间戳的片段，Qwen 默认仅返回文本，原生流式会修订尚未结束的文本。
这些差异已在独立接口表达，可以逐步接入现有窗口；本轮没有证据说明更换 GUI 框架会解决推理或时间戳问题。
完整窗口的 offscreen 构造仍有既有平台限制，本轮组件检查不等于完整桌面交互验收。

主工作流继续使用 faster-whisper，community-1 已作为说话人分离默认模型。
Vulkan、Qwen、Sherpa 和 Roformer 保留为实验候选，不据一个短样本替换现有默认后端。
后续长输入复测确认 Vulkan / ROCm 对 turbo 模型有效加速；在缓存内核后，现有 Demucs
的稳定处理快于本次 Roformer，community-1 的 GPU 也快于 CPU。冷启动数据与稳定数据详见长音频报告。
下一轮产品集成需要后端能力驱动的参数表、实时文本修订展示、Qwen 对齐，以及真实麦克风和多说话人验证。

## 已实现的边界

- `domain.segments` 定义应用自己的片段和词，复制原生词并保留 speaker，不依赖 Qt / torch。
- `RecognitionBackend` 返回片段、语言/时长元数据以及可选文本。未知置信度和 VAD 时长保留 `None`。
- 文件与现有实时分块 Worker 使用同一接口，原生 faster-whisper 模型自动包装。
- `WhisperCppBackend` 将 CLI JSON 时间转换为秒，支持 CPU / Vulkan，拒绝未映射的参数。
- `QwenBackend` 显式标记纯文本结果没有片段时间戳；字幕 Worker 会拒绝未经对齐的结果。
- `StreamingSession` 接收音频并输出包含修订号、最终状态的 `StreamUpdate`；Sherpa 适配使用原生有状态流。
- community-1 消费 `DiarizeOutput` 的排他时间轴，同时保存完整原始输出，兼容旧 Annotation 和空结果。

现有 GUI 模型加载和参数表仍使用 faster-whisper，现有录音仍按块识别。
实验后端不接受完整的 faster-whisper 参数集合，不能直接注入当前 GUI 参数表完成全部功能。

## 实测

数据均在 `.cache/reports/`，音频在 `.cache/fixtures/`，模型在 `.cache/models/`。
耗时是此设备、此输入与此配置下的测量，不是通用排行榜；不同模型、精度和测量边界不能直接比较。

| 项目 | 输入与配置 | 结果 | 限制 |
| --- | --- | --- | --- |
| community-1 | JFK 前 10 秒，CPU，新令牌加载 | 1.25 秒；4 个区间，1 位说话人；排他输出可用 | 单人短样本只验证加载与输出契约，没有评估多人准确率 |
| whisper.cpp CPU | tiny.en，JFK 11 秒，3 次独立 CLI | 中位 0.825 秒 | 包含进程启动和模型加载 |
| whisper.cpp Vulkan | 同模型同音频同测量，3 次 | 中位 0.939 秒；日志确认 `using Vulkan0 backend`；文字与 CPU 一致 | 本短样本没有加速优势；没有测更大模型 |
| sherpa-onnx 原生流式 | 英文 Zipformer 20M int8；100 ms 块，按真实速度回放 | 首个非空结果约 2.108 秒，18 次修订，最终结果可用 | 开头识别漏词；不是麦克风端到端延迟，没有测中文/噪声 |
| Qwen3-ASR-0.6B CPU | JFK 11 秒，float32 | 识别 52.74 秒，文字完整 | 首次加载含模型下载，不用于加载性能比较 |
| Qwen3-ASR-0.6B ROCm | JFK 11 秒，float16 | 识别 13.91 秒，文字与 CPU 一致 | 没有默认词/片段时间戳；GPU 有 MIOpen / SDPA 回退警告 |
| Qwen 统一适配器 | 官方中文 4.204 秒，ROCm float16 | 3.842 秒；「甚至出现交易几乎停滞的情况。」；语言规范化为 zh | 未做中文语料准确率统计；没有启用 ForcedAligner |
| MelBand Roformer | 官方 Demucs 教程混音 150–155 秒的 5 秒片段；GPU、segment_size=64、overlap=2、batch=1、autocast | 约 5.00 秒；两个有限、完整的 44.1 kHz 双声道输出；参考人声 SI-SDR 6.22 dB | 没有同配置旧分离模型基线，不能证明优于 Demucs |

## 实验环境

主环境保持 torch `2.15.0a0+rocm10.2.0a20260926`、transformers `5.17.0`、
CTranslate2 `4.8.2`；未覆盖自建 CT2 DLL。

- `.cache/experiments/streaming-env`：Python 3.12.13、sherpa-onnx 1.13.8。
- `.cache/experiments/qwen-env`：Python 3.14.5、qwen-asr 0.0.6、transformers 4.57.6、accelerate 1.12.0、tokenizers 0.22.2。通过 `.pth` 借用主环境 ROCm torch；实验包优先于主环境。
- `.cache/experiments/separation-env`：Python 3.14.5、audio-separator 0.47.0、beartype 0.22.9。借用 Qwen 环境音频依赖和主环境 torch。
  包装库钉住的旧 beartype 在 Python 3.14 下不可用；仅实验环境覆盖为已实测的稳定版。这是兼容性例外，未写入主依赖。
- 这三个环境是本机实验组合，不是可独立部署的完整依赖锁；环境包清单保存为报告目录中的 `*-environment.txt`。
- whisper.cpp 固定 v1.9.4（提交 `927cfce34f31707e17f2bff35c349632fb9e2c3a`），使用本机 GCC / Ninja 构建静态 Vulkan CLI。Vulkan-Headers 1.4.365、shaderc 2026.3、SPIRV-Headers / SPIRV-Tools 提供构建依赖。

Vulkan 构建设置：`GGML_VULKAN=ON`、`GGML_OPENMP=OFF`、`WHISPER_BUILD_TESTS=OFF`、
`BUILD_SHARED_LIBS=OFF`；Vulkan loader 为 `C:/Windows/System32/vulkan-1.dll`。
构建目录为 `.cache/experiments/whisper.cpp/build-vulkan`。
设置 `Vulkan_INCLUDE_DIR` 指向 Vulkan-Headers/include，`Vulkan_GLSLC_EXECUTABLE` 指向
shaderc/Library/bin/glslc.exe，`CMAKE_PREFIX_PATH` 与额外 C++ include 指向 spirv-prefix。
构建目标为 `whisper-cli`；仅 GPU 枚举日志不足以证明 Vulkan 推理，测试检查实际使用的 backend。

## 复测

以下命令使用现成实验环境和缓存，不执行主环境依赖同步。
令牌从被 Git 忽略的本地配置读取，不进入命令或报告。

```powershell
.\.venv\Scripts\python.exe -m project_tools.check_backends
.\.venv\Scripts\python.exe -m project_tools.check_structure
.\.venv\Scripts\python.exe src/project_tools/benchmark_diarization.py --audio .cache/fixtures/jfk.wav
.\.venv\Scripts\python.exe src/project_tools/benchmark_whisper_cpp.py --executable .cache/experiments/whisper.cpp/build-vulkan/bin/whisper-cli.exe --model .cache/models/ggml-tiny.en.bin --audio .cache/fixtures/jfk.wav --runs 3
.\.cache\experiments\streaming-env\Scripts\python.exe src/project_tools/benchmark_streaming.py --model-dir .cache/models/streaming/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 --audio .cache/fixtures/jfk.wav --pace --report .cache/reports/streaming-paced.json
.\.cache\experiments\qwen-env\Scripts\python.exe src/project_tools/benchmark_qwen.py --audio .cache/fixtures/qwen-zh.wav --device cuda --language zh --report .cache/reports/qwen-zh-gpu.json
.\.cache\experiments\separation-env\Scripts\python.exe src/project_tools/benchmark_separation.py --audio .cache/fixtures/hdemucs_mix_150_155.wav --reference .cache/fixtures/hdemucs_vocals_segment.wav
```

本轮后端契约检查 6 项、结构检查 9 项通过；此前既有修复检查 15 项通过。
覆盖原生片段到应用类型、时间平移、speaker、独立流状态、空说话人结果以及纯文本边界。
主窗口完整交互、录音设备、长音频与候选模型准确率不在这些检查覆盖内。

## 官方资料与样本

- [community-1 模型与排他输出](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [whisper.cpp 源码与构建](https://github.com/ggml-org/whisper.cpp/tree/v1.9.4)，[JFK 音频](https://github.com/ggml-org/whisper.cpp/blob/v1.9.4/samples/jfk.wav)
- [sherpa-onnx 原生在线识别示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py)
- [Qwen3-ASR 接口与 ForcedAligner](https://github.com/QwenLM/Qwen3-ASR)，[中文样本](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav)
- [audio-separator 支持的模型及调用](https://github.com/nomadkaraoke/python-audio-separator)
- [PyTorch 官方 Demucs 教程与参考片段](https://github.com/pytorch/audio/blob/release/2.8/examples/tutorials/hybrid_demucs_tutorial.py)
