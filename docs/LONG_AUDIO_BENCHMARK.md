# 长音频后端性能复测

日期：2026-10-02。本机 Windows，Ryzen 7 7840H / Radeon 780M。

中文长音频与 CER 精度补测见 [中文识别评估](CHINESE_ASR_EVALUATION.md)，
包含三位说话人、Whisper CPU/Vulkan/ROCm、Qwen3-ASR 及中文原生流式模型。

## 输入与测量方法

识别输入为 91.465 秒、16 kHz 单声道 PCM16 WAV。由
[Hugging Face 的 LibriSpeech 验证样本](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
中 9 条不同语句按顺序拼接，没有循环短音频或添加静音。
来源 ID 和文字保存在 `.cache/fixtures/librispeech-long.json`。
SHA256：`63a3b163d73ce443480d7bb7e825c1c7340e54978af2e3fd93b4cc2ba814ae43`。
这些语句来自同一朗读上下文，不能代替连续会议、中文、多人或噪声测试。

音乐分离使用上轮官方 Demucs 教程混音的前 60 秒，44.1 kHz 双声道 FLOAT WAV。
它与识别测试的输入和任务不同，不参与识别后端速度排名。

全部推理串行执行，正式测量时没有并行编译或其他测试任务。
Whisper 系列固定 4 个 CPU 线程、beam_size=1、best_of=1，不启用 VAD。
测试 tiny.en 和 large-v3-turbo 两组，避免仅凭小模型决定实际使用方案。
faster-whisper 的 turbo 使用用户当前配置中的本地模型。
CPU tiny 使用 float32、CPU turbo 使用 int8、ROCm faster-whisper 使用 float16；
whisper.cpp 使用原始 ggml 浮点权重。相同系列不意味着所有实现与精度完全相同。

Whisper 每项连续运行 3 次，记录全部耗时和中位数，首轮保留。
faster-whisper 加载单独计时，推理必须消费完惰性片段。
whisper.cpp 的外部时间包含启动进程、加载、识别和 JSON 输出；同时记录 CLI 原生 load/total 时间。
原生 total-load 是该实现加载后到识别结束的近似口径，仍不与另一实现严格等价。
Qwen CPU 仅测一次，不能视为多轮稳定中位数。
community-1 在首次 GPU 比 CPU 慢后，另测同一已加载流水线的三次处理，报告中位数；
Qwen GPU、两种分离模型也补测三次，以检查首次内核初始化成本。
Qwen 增加 max_new_tokens 到 2048，避免短样本的 256 上限截断长输入。

RTF=处理耗时/音频时长，越低越快；实时倍率=音频时长/处理耗时。
按实际速度回放的流式总耗时包含等待音频到达，不代表推理吞吐。
英文 WER 仅对本输入计算：小写、去掉撇号，再按 ASCII 字母数字分词，比较最后一轮输出。
它是单样本检查，不是准确率排行榜，也不评价字幕时间戳质量。

## ROCm 构建

使用与 Vulkan 相同的 whisper.cpp v1.9.4 源码提交
`927cfce34f31707e17f2bff35c349632fb9e2c3a`，不是不同版本的第三方预编译程序。
AMD HIP SDK 7.1 的 clang-cl 21 与 MSVC 14.51 的 cmath 存在声明冲突。
选用本机已有的 MSVC 14.44 后构建通过，没有修改系统头文件或主项目依赖。

构建配置：`GGML_HIP=ON`、`GGML_HIP_RCCL=OFF`、`GGML_HIP_NO_VMM=ON`、
`GPU_TARGETS=gfx1103`、`GGML_OPENMP=OFF`、`BUILD_SHARED_LIBS=ON`。
工具集环境由 `VsDevCmd.bat -arch=x64 -host_arch=x64 -vcvars_ver=14.44` 提供。
独立构建目录为 `.cache/experiments/whisper.cpp/build-rocm`。
测试子进程 PATH 包含 HIP SDK bin，沿用本项目的
`ROCBLAS_USE_HIPBLASLT=1`、`HSA_OVERRIDE_GFX_VERSION=11.0.0` 兼容设置。
检查日志中的实际 `using ROCm0 backend`，而不是只检查设备枚举。
官方构建参考：[whisper.cpp v1.9.4](https://github.com/ggml-org/whisper.cpp/blob/v1.9.4/README.md)。

## 复测与证据

维护脚本位于 `src/project_tools/`：

- `prepare_long_fixture.py`：从缓存 parquet 生成长语音和来源清单（在 streaming-env 执行，需要 pyarrow）。
- `benchmark_whisper_cpp.py`：用 `--devices cpu vulkan` 或 `--devices rocm` 选择后端；`--model` 选择 tiny.en 或 turbo。
- `benchmark_faster_whisper.py`：用 `--device cpu/cuda`、`--compute-type`、`--model` 指定配置。这里 cuda 指当前自建 ROCm CTranslate2 的兼容接口。
- `benchmark_long_suite.py`：串行执行 Qwen GPU/CPU、流式吞吐/实际速度回放、community-1、60 秒 Roformer。
- `summarize_long_benchmarks.py`：汇总原始报告和单输入 WER。

原始报告、标准输出和编译日志保存在 `.cache/reports/`。主环境和自建 CT2 DLL 保持原样。
实验环境依赖与部署限制仍见 [BACKEND_EVALUATION.md](BACKEND_EVALUATION.md)。

## large-v3-turbo 实测结果

处理同一段 91.465 秒语音。加载后耗时是三次中位数；whisper.cpp 使用每次原生 total-load，
faster-whisper 使用消费完全部片段的外部计时，因此是接近但并非严格相同的计时边界。
加载栏：whisper.cpp 为三次中位数；faster-whisper 为本进程唯一一次加载。

| 后端 | 配置 | 加载秒 | 加载后处理秒 | 加载后实时倍率 | CLI 完整调用秒 | 单样本 WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| whisper.cpp | CPU，浮点权重 | 0.453 | 61.680 | 1.48× | 62.503 | 6.16% |
| whisper.cpp | Vulkan | 0.835 | 8.914 | 10.26× | 10.106 | 6.16% |
| whisper.cpp | ROCm | 2.434 | 7.349 | 12.45× | 10.071 | 5.69% |
| faster-whisper | CPU int8 | 3.998 | 18.591 | 4.92× | — | 5.69% |
| faster-whisper | ROCm float16 | 2.644 | 7.439 | 12.29× | — | 6.16% |
| faster-whisper | ROCm int8_float16 | 4.751 | 6.365 | 14.37× | — | 6.16% |

这组结果说明大模型上 Vulkan 已有明确 GPU 加速收益。ROCm 的加载后处理更快，
但本次 whisper.cpp ROCm 加载开销也更大，完整调用与 Vulkan 接近。
faster-whisper ROCm 和 whisper.cpp ROCm 的加载后处理速度接近，不能把前者 7.44 秒
与后者完整调用 10.07 秒直接解释成实现间的纯推理差距。
CPU int8 与原始浮点权重的精度不同，不能用这两项证明后端在同精度下的效率差异。

## tiny.en 实测结果

相同 91.465 秒输入，三次中位数。whisper.cpp 列为完整调用耗时，faster-whisper 列为加载后处理。

| 后端 | 配置 | 耗时秒 | 单样本 WER |
| --- | --- | ---: | ---: |
| whisper.cpp | CPU | 3.372 | 9.00% |
| whisper.cpp | Vulkan | 3.034 | 9.00% |
| whisper.cpp | ROCm | 1.944 | 9.00% |
| faster-whisper | CPU float32 | 2.355 | 10.43% |
| faster-whisper | ROCm float16 | 1.978 | 9.95% |

tiny.en 比 turbo 更快，同时在本输入的英文 WER 更高；不能将不同模型的速度差全部归因于后端。

## 其他方案的长输入

| 方案 | 音频秒 | 加载秒 | 处理秒 | 处理实时倍率 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-ASR-0.6B CPU float32 | 91.465 | 7.099 | 470.323 | 0.19× | 单次；完整文字；英文 WER 5.69% |
| Qwen3-ASR-0.6B ROCm float16 | 91.465 | 10.325 | 28.594 | 3.20× | 三次中位数；完整文字；英文 WER 5.69% |
| sherpa-onnx 英文 Zipformer 20M int8 | 91.465 | 0.915 | 2.455 | 37.26× | 不限速提交；WER 9.48% |
| community-1 CPU | 91.465 | 2.074 | 36.199 | 2.53× | 三次中位数；18 个区间、1 位说话人；排他时间轴 |
| community-1 ROCm | 91.465 | 2.150 | 6.414 | 14.26× | 三次 12.900 / 6.414 / 6.003 秒，18 个区间、1 位说话人 |
| MelBand Roformer ROCm | 60 | 2.921 | 37.413 | 1.60× | 三次中位数；两条音轨；autocast、segment_size=64、overlap=2 |
| HDEMUCS_HIGH_MUSDB_PLUS ROCm | 60 | 0.946 | 4.457 | 13.46× | 三次中位数；四条音轨；float32、7.8 秒块、overlap=0.1；使用缓存权重 |

流式按真实速度回放用时 91.478 秒，首个非空结果出现在输入 2.1 秒处，墙钟 2.107 秒，
214 次修订。最终文本与不限速提交一致。这不是麦克风设备端到端延迟测量。

分离测试验证所有输出都是完整 60 秒，样本有限，没有对这段混音的参考音轨评分。
现有 Demucs 工作流本次稳定处理明显快于 Roformer，但两/四音轨、精度、重叠设置均不同，
不能用速度证明分离质量更好，也不能替代可控的同精度模型基线。
Demucs 有 MIOpen workspace 警告但成功输出四个完整音轨，使用现有 Worker 的分块算法。

## 更新后的选择

保留当前 GUI 和 faster-whisper 主工作流。当前设备上，large-v3-turbo 的 ROCm
int8_float16 是本轮较快的应用配置候选；没有自动修改用户的精度设置。
whisper.cpp ROCm 的浮点处理速度与 faster-whisper ROCm float16 接近，Vulkan
也有明确大模型加速收益，值得作为可选后端，不再以 11 秒 tiny.en 样本否定它。
本轮旧短音频与新长音频的解码配置并不完全相同，不能将变化全部归因于音频长度。
Qwen 在这段英文上没有显示出替换 Whisper 的速度或准确率优势，中文与对齐仍需独立评估。
Sherpa 的原生流式路径值得接入实时文本显示；其最终文字质量在本样本接近 tiny.en，
低于 turbo / Qwen，尚未测中文与现场噪声。
community-1 单人长输入只能验证性能与返回契约，不能证明多人区分能力。
community-1 首次冷 GPU 推理 69.822 秒，比 CPU 慢；随后缓存内核的三次结果中位为 6.414 秒，
稳定使用时 GPU 优势明显。初次运行结果保存在 `community-long-rocm-first-pass.json`，没有从记录中删除。
Demucs 首次加载 30.016 秒包含 319 MB 权重下载，首次 GPU 处理 56.442 秒也包含内核初始化成本；
缓存内核后同进程三次为 11.049 / 4.448 / 4.457 秒，中位 4.457 秒。
Qwen GPU 初次长输入 50.531 秒，复测三次为 30.804 / 28.594 / 28.379 秒。
Roformer 三次为 37.413 / 37.236 / 39.551 秒，未显示出类似幅度的初始化差异。
这些初次记录分别保存在 `demucs-long-first-pass.json`、`qwen-long-gpu-first-pass.json`、
`separation-long-first-pass.json`。判断持续运行性能应看复测结果；偶尔冷启动的实际等待还需加上初始化和加载。

原始结果汇总为 `.cache/reports/long-comparison.json`，包含每项 RTF、最后一轮 WER 与输入清单。
后端契约检查 6 项、项目结构检查 9 项通过。实验适配尚未接入 GUI 后端选择，未修改当前应用精度设置。
