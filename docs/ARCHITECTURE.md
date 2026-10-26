# 项目结构与维护指南

本项目是 Python / PySide6 桌面应用。应用源码位于 `src/`，内部开发工具也位于 `src/`，启动和安装脚本位于根目录，
依赖由 `pyproject.toml` 与 `uv.lock` 管理。

## 目录结构

```text
src/faster_whisper_GUI/
  app.py                        启动、日志、启动画面和事件循环
  __main__.py                   python -m faster_whisper_GUI
  __init__.py                   轻量包入口，不隐式加载推理库
  config.py                     参数选项、默认值和格式常量
  version.py                    应用版本信息
  logging_setup.py              日志与诊断包；路径和共享状态保持稳定
  domain/
    parameters.py               Whisper / VAD 参数类型
    segments.py                 统一片段表示与外部结果转换
  backends/
    base.py                     后端接口、统一识别结果与元信息
    faster_whisper.py           已加载 faster-whisper 模型的适配
    catalog.py                  后端选择、默认设置、各后端参数映射
    whisper_cpp.py              CPU / Vulkan / ROCm CLI 适配与取消
    remote.py                   Qwen / sherpa 独立进程适配
    qwen.py / aligned.py        Qwen 识别、测量时间戳与字幕分段
    sherpa.py / streaming.py    原生在线会话与文本修订协议
    audio.py / token_segments.py 音视频解码、状态重采样与流式字幕
  runtime/
    rocm.py                     ROCm/HIP 探测、DLL 目录和进程状态
    inference.py                幂等推理初始化顺序
    paths.py                    源码与运行时数据目录
    preferences.py              YAML 配置的安全读取、校验与原子保存
    timecode.py                 时间码转换
    diagnostics.py              任务阶段日志
    model_process.py            持久模型子进程、JSON 通信与资源释放
  transcription/
    parameters.py               文件与实时转录共用的参数映射
    file.py                     文件转录、并发、临时字幕和取消
    native_streaming.py         GUI 原生在线转录、状态重采样与修订
    streaming.py                旧分块实现，保留兼容与回归检查
  tasks/
    base.py                     GuardedWorker 异常护栏
    capture.py                  麦克风采集与 WAV 保存
    export.py                   字幕导出后台任务
    model.py                    Whisper 模型加载
    backend_load.py             可选识别后端加载
    conversion.py               Hugging Face -> CTranslate2 模型转换
    alignment.py                对齐和说话人分离后台任务
    separation.py               Demucs 人声分离后台任务
    roformer.py                 Roformer 独立进程分离与取消
    audio_split.py              按说话人切分音频
  subtitles/
    readers.py                  SRT / JSON 读取
    writers.py                  SRT / TXT / VTT / LRC / SMI / JSON / ASS 写出
  ui/
    pages/                      模型、转录、VAD、执行、输出、主页、分离、设置页
    widgets/                    文件列表、参数控件、结果表格等可复用控件
    models/segments.py          字幕片段的 Qt 表格模型
    icons.py                    图标枚举
    styles.py                   QSS 资源选择
    translation.py              Qt 翻译对象
    window/
      view.py                   创建布局、页面和控件，读取并应用配置
      main.py                   窗口初始化、信号连接、持有任务状态
      parameters.py             从控件读取和转换任务参数
      model.py                  模型操作的界面编排
      transcription.py          文件/实时转录的界面编排
      results.py                结果表格生命周期、简繁转换
      postprocessing.py         对齐与说话人分离的界面编排
      files.py                  导入、导出及音频切分的界面编排
      separation.py             人声分离的界面编排
      settings.py               配置备份、恢复、保存和退出
      feedback.py               日志投递、进度与消息提示
      signals.py                任务状态信号
      main_constants.py         旧分块兼容参数
src/whisperx/                       本地维护的 WhisperX 副本，保留现有适配
src/resource/                       Qt 图标、QSS、翻译资源
src/project_tools/              离线回归与维护脚本（不打包进应用）
setup.ps1                   环境搭建
启动GUI.bat                 Windows 启动入口
docs/                           架构、运行时与参考资料
```

GUI 多后端入口、独立环境与原生流式/字幕处理细节见 [GUI 工作流](GUI_BACKENDS.md)。
`src/project_tools/model_server.py` 是独立模型进程入口；GUI 通过 `ModelProcess` 通信，不在主环境导入 Qwen / sherpa / audio-separator。

## 模块职责和依赖方向

界面模块读取控件并提交后台任务；后台任务调用转录、对齐、分离和字幕模块。
结果表格及字幕模块消费统一片段表示。运行时模块负责 DLL 与推理依赖的初始化。

主窗口职责分组是内部 mixin，共用 `MainWindows` 的控件和任务状态。
它们不创建额外 QObject，不另外持有一套窗口状态。Qt 信号仍定义在
`MainWindows` / Worker 上，界面对象继续由主线程持有。

`view.py` 创建控件，`main.py` 连接信号；耗时算法放在后台任务中。
职责分组不是独立的业务接口，不应作为其它模块的基类。
目前保留共享窗口状态以控制迁移范围，后续可按实际需求进一步缩小状态依赖。

新代码使用规范导入，例如：

```python
from faster_whisper_GUI.transcription.file import TranscribeWorker
from faster_whisper_GUI.tasks.export import OutputWorker
from faster_whisper_GUI.subtitles.writers import writeSubtitles
from faster_whisper_GUI.ui.window.main import MainWindows
```

## 启动与 AMD 运行时

1. 启动器调用 `app.main()`，切换到项目运行目录，再通过 `runtime.rocm.setupROCm()` 注册 DLL 目录。
2. 初始化 `logging_setup`，再建立 QApplication 和启动画面。
3. 推理模块通过 `runtime.inference.prepare_inference_runtime()` 预加载 torch。
4. 随后才导入 faster-whisper / CTranslate2 / WhisperX。
5. 建立主窗口并进入 Qt 事件循环。

`prepare_inference_runtime()` 成功返回后缓存结果；初始化失败会继续抛给调用方，
不会缓存异常。`rocm.py` 保留既有 AMD 环境兼容设置，未改变加速策略。

单独导入包、启动模块、参数类型、时间码或字幕写出模块，不会加载 Qt、torch 或 CTranslate2。
片段与 Word 是应用类型，结果转换和字幕读取不初始化 Qt、torch 或 faster-whisper。

## 转录结果契约

文件转录与实时转录统一返回 `(segments, audio_path, info)` 结果列表。
片段使用 `domain.segments.segment_Transcribe`，包含 `start`、`end`、`text`、
`words` 和 `speaker`。保留这个既有类名以兼容调用方。

实时结果先规范化为应用片段，再平移片段和词的时间戳。禁止将缺少 `speaker`
的原生 faster-whisper Segment 直接交给表格或导出模块。

转录参数映射集中在 `transcription/parameters.py`。新增参数时，文件和实时
转录共用同一映射，避免两份实现漂移。实时块长继续为 30 秒。

字幕写出只依赖轻量常量、时间码和文本库；片段类型只在类型检查时导入，
因此离线导出不需要建立 Qt 应用或初始化 GPU。

## 安装与启动

`pyproject.toml` 使用 setuptools 从 `src` 发现三个应用包，包含 WhisperX 的
`assets/mel_filters.npz` 和 Qt 资源文件。uv 将本项目以可编辑方式安装。
`src/project_tools/` 是维护脚本，`docs/` 是文档，运行时配置保存在 `config/config.yml`，日志保存在 `logs/`，项目内部缓存统一放入 `.cache/`。

```powershell
uv run python -m faster_whisper_GUI
```

`启动GUI.bat` 调用模块入口并配置统一字节码缓存。仅更新应用代码时可使用
`uv pip install --no-deps --editable .`，避免覆盖现有自建 CTranslate2 DLL。
一般完整环境搭建仍按 `setup.ps1` 执行，在同步依赖后重新放置 ROCm DLL。

## 配置持久化

`runtime/preferences.py` 是配置文件的唯一读写入口，路径由 `PROJECT_ROOT` 定位，不依赖工作目录。
`config/config.yml` 包含 `ui`、`model`、`transcription`、`vad`、`separation`、`settings`、`output` 七个配置节。
主题、主题色和字体只保存在 `ui`；模型名、精度和转写语言保存文本值，不保存重复的下拉框索引。

QFluentWidgets 仅维护内存中的界面状态。主题、颜色和字体更新均使用 `save=False`；
应用退出时依据自动保存开关统一保存 YAML，备份与恢复使用同一格式和读写入口。
首次启动缺少配置时使用默认值；读取损坏配置时保留原文件，关闭自动保存，避免覆盖。
加载备份前先检查配置结构和控件字段类型，任务运行期间禁止加载。

配置使用安全 YAML 类型，拒绝重复键和 Python 对象标签；手写的 `yes`、`on` 和日期保持字符串。
保存先完整写入同目录临时文件，再原子替换目标。配置可能含访问令牌，必须被 Git 忽略并排除于诊断包。
PyYAML 保存会重排格式且不保留手写注释。

## 验证

在仓库根目录、现有项目环境中执行：

```powershell
.\.venv\Scripts\python.exe -m project_tools.check_structure
.\.venv\Scripts\python.exe -m project_tools.check_fixes
.\.venv\Scripts\python.exe -m project_tools.check_workers
.\.venv\Scripts\python.exe -m project_tools.check_logging
.\.venv\Scripts\python.exe -m unittest project_tools.check_preferences project_tools.check_gui_interactions project_tools.check_startup
```

- `check_structure.py`：可编辑安装与模块入口、轻量导入、实现模块导入方向、漏导入、运行时路径、
  Qt 信号、文件转录到临时 SRT/JSON 再读取，以及配置保存。
- `check_fixes.py`：已有崩溃路径、第三方数据契约、跨线程投递和复用回归。
- `check_workers.py`：异常护栏、失败信号与所有后台任务的覆盖。
- `check_logging.py`：控制台/界面/文件日志、诊断包、保留策略和共享日志状态。

这些检查不下载模型，不代表 GPU 推理性能或全部真实音频场景已验证。
配置和界面检查使用真实 Qt 控件与隔离测试数据，验证备份恢复、主题修改和配置往返。
修改模型加载、录音或 GPU 适配后，还应执行对应的人工功能验证。

## 识别后端

文件和实时分块 Worker 通过 `RecognitionBackend.recognize(audio, options)` 调用后端，
返回 `Recognition(segments, info, text, segment_timestamps)`。片段保持惰性迭代，以支持片段间取消；时间戳单位为秒。
现有模型自动包装为 `FasterWhisperBackend`，也可向 Worker 注入实现该接口的后端。
GUI 支持 faster-whisper、whisper.cpp、Qwen 和 sherpa；参数按后端映射。
Qwen 通过 ForcedAligner 提供实测词级时间戳，再按字数、时长和停顿生成字幕。
原生流式使用独立的 `StreamingBackend` / `StreamingSession`，通过 `StreamUpdate`
表达可修订文本、修订号与最终状态；`SherpaStreamingBackend` 已用按实际速度回放的音频验证。
GUI 录音使用 sherpa 原生在线会话；文件识别与录音的使用入口见 [GUI 工作流](GUI_BACKENDS.md)。
实验结果、环境隔离与 GUI 决策见 [BACKEND_EVALUATION.md](BACKEND_EVALUATION.md)。

说话人分离支持 community-1 的原生 `DiarizeOutput`，字幕默认消费排他时间轴，
原始输出保存在 `last_output`；旧 Annotation 返回值及空输出同样可用。
离线契约检查：`.venv/Scripts/python.exe -m project_tools.check_backends`。
