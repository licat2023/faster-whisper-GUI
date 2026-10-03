# GUI 多后端与字幕工作流

2026-10-03：以下入口已写入应用并通过本机真实模型验证。

## 使用入口

1. **模型参数**页顶部选择 faster-whisper、whisper.cpp、Qwen3-ASR + 时间戳对齐或 sherpa-onnx，填写该后端的模型参数，然后点击**加载模型**。
2. **转写参数**页随所选后端显示对应参数。切换后端后需要重新加载；正在加载或识别时锁定后端选择。
3. **执行转写**页添加音视频文件，点击**开始**；结果进入原有字幕编辑表格。
4. **声乐分离**页顶部可选择 Demucs 或 MelBand Roformer。选择输入文件与输出目录后点击**提取**；Roformer 输出人声、伴奏两个 WAV 音轨。

| 后端 | 模型参数 | 转写参数 |
| --- | --- | --- |
| faster-whisper | 保留原来的模型、设备、精度、线程、缓存设置 | 保留原来的 Whisper / VAD 参数 |
| whisper.cpp | ggml 文件、CLI 路径、CPU / Vulkan / ROCm、线程数 | 语言、beam size、best of、初始提示词 |
| Qwen | ASR 模型、ForcedAligner 模型、独立 Python、GPU / CPU、生成上限 | 语言、字幕分段字数、目标最长秒数 |
| sherpa | Zipformer 模型目录、独立 Python、线程数 | 模型语言、字幕分段字数、目标最长秒数 |
| Roformer | 模型文件名、独立 Python、窗口帧数、重叠次数 | 分离输入列表与输出目录 |

whisper.cpp 切换设备时会切换本机默认 CLI 路径；用户手填的 CLI 路径保持原值。实际运行会检查 GPU 后端日志，错误的编译版本不会默默计作 Vulkan / ROCm。Qwen 的 `cuda` 是 PyTorch 的 GPU 接口名，本机实际使用 ROCm。

## 原生流式录音

选择并加载 **sherpa-onnx**，在执行页选择**原生流式录音**。采集的 PCM 持续送入同一个 Zipformer 在线会话，不再每 30 秒重新转录。重采样使用有状态的 PyAV 滤波器，并在停止时处理剩余样本。界面替换整个当前识别假设，显示修订序号与累计音频时长，避免把修订文字重复追加。

点击**停止录音**后，先关闭采集并写完 WAV，再排空队列、结束在线会话，最终结果进入字幕表格。流式字幕使用模型输出的 token 起始时间；每组结束取下一组起始时间，末组取录音时长。因此末尾静音可能包含在末段中，不能把这些时间当作词尾强制对齐。词级时间戳留空。

faster-whisper、whisper.cpp 和当前 Qwen 适配提供文件识别，录音入口只对原生在线 sherpa 后端开放。保留旧分块模块用于兼容和回归检查，GUI 已不再调用它。

## Qwen 字幕与编辑

Qwen 模型加载时同时加载 ForcedAligner；识别后自动取其测量的词级起止时间，按字数、时长和停顿分组并保留原文标点。一个不可再分的长词可能超过分段目标。对齐缺失、倒序或文字不匹配会明确失败，不以平均分配时间代替。

结果进入现有表格，可编辑并导出 SRT、TXT、VTT、ASS、LRC、SMI、JSON。手动修改整段文字后，该段旧词级对齐会清空，保留段级起止时间，确保所有导出格式使用校订后的文字。可编辑词级列，或通过现有对齐流程重新取得对齐。

## 交互回归修复（2026-10-03）

新转写和字幕导入会保留其它文件的结果。同一文件存在未保存的手动编辑时，替换前会提示确认；取消替换会保留旧编辑。字幕删除和合并会同步更新导出数据及表格状态。保存字幕使用点击保存时的快照，保存过程中继续编辑的内容仍保持未保存状态。

移动标签后，关闭和刷新操作仍对应正确的文件。音频分段成功、失败或取消后均恢复输出页按钮；退出时先停止并等待分段线程完成。任务运行期间不能加载配置，损坏配置会明确提示失败。

交互回归使用真实 Qt 控件、合成字幕和模拟 ffmpeg 返回值，不保存用户配置，不下载模型或使用麦克风：

```powershell
.\.venv\Scripts\python.exe -m project_tools.check_gui_interactions
```

## 环境与目录

GUI 主环境没有同步或替换现有 ROCm torch / CTranslate2。faster-whisper、Qwen、sherpa、Roformer 通过子进程运行，JSON 消息返回统一片段；模型输出和错误写到 `.cache/logs/model-*.log`。faster-whisper 使用启动 GUI 的同一个 Python 环境，保留模型、精度、Whisper / VAD 和词级时间戳参数。切换后端或卸载时结束模型进程，避免主线程释放后台创建的 ROCm/CTranslate2 模型导致原生崩溃。取消文件识别会终止对应模型进程，下一次任务需要重新加载。Roformer 的子进程随分离任务释放。

每个模型进程串行接收文件识别请求；GUI 的多个文件会排队处理。ROCm DLL 目录在每个子进程中单独注册，父进程初始化标记不会让子进程跳过初始化。

## 启动与界面修复

窗口启动不再导入 torch、CTranslate2、Transformers 或 WhisperX。模型加载在后台子进程执行；WhisperX 后处理和 Demucs 的依赖在使用相应功能时加载。停用的模型转换控件保留为有父控件的隐藏控件，不再弹出独立窗口。浅色主题为普通标签显式设置深色文字，深色主题仍使用浅色文字。

本机诊断时主窗口可见约 15–19 秒；修复后 Windows 窗口检查约 2.5 秒，含启动画面与配置读取的完整入口检查约 5.4 秒。这些是在当前机器和已有缓存下的测量，模型加载时间另计。

新增 `src/project_tools/check_startup.py` 的 4 项检查覆盖启动依赖、独立窗口、两种主题和子进程 ROCm 初始化。真实 ROCm large-v3-turbo 模型验证了 faster-whisper → sherpa → faster-whisper、中文识别、VAD、词级时间戳、卸载、取消及模型子进程异常后的 GUI 恢复；原始崩溃复现脚本也已通过回归。原有 43 项回归检查和新增 4 项检查全部通过。

```powershell
.venv/Scripts/python.exe src/project_tools/check_startup.py
$env:QT_QPA_PLATFORM = 'windows'
.venv/Scripts/python.exe src/project_tools/check_startup.py --entrypoint
.venv/Scripts/python.exe src/project_tools/check_startup.py --model '本地 CT2 模型目录' --device cuda
```

默认使用本机已有环境：

- `.cache/experiments/qwen-env/Scripts/python.exe`
- `.cache/experiments/streaming-env/Scripts/python.exe`
- `.cache/experiments/separation-env/Scripts/python.exe`

它们仍是本机实验环境，尚不是可移植的独立依赖锁。新机器需要自行准备相应环境并在模型参数中指定 Python 路径；缺失路径会明确报错。Qwen 的两个模型已缓存于 `.cache/models/qwen`；Roformer 模型在 `.cache/models/separation`。模型名称加载可能访问下载服务，可改填本地模型目录。

## 验证结果与复测

- 结构检查 9 项、后端契约 6 项、既有修复 15 项、新 GUI / 流式 / 资源回归 13 项通过；worker 异常护栏检查通过。
- 四个可选模型均用真实权重跑通：Qwen 中文样本 4.204 秒，得到 13 个词级时间戳、1 段字幕（0.40–3.68 秒）；whisper.cpp 与 sherpa 识别同一个样本；Roformer 分离 5 秒音乐片段并验证两个音轨有效。
- 完整 Qt 窗口验证通过加载、参数读取、后台任务、结果表格；Qwen 还验证了表格校订后经 GUI 导出 SRT / VTT / ASS / LRC / JSON。
- 原生录音链路使用固定音频回放，执行真实 PCM 转换、48kHz→16kHz 重采样、WAV 保存和修订回调；实体麦克风尚未人工验收。
- 在 Windows Qt 平台上检查了页面截图。此轮是功能集成验收，性能和中文准确率对比仍见原有长音频报告。

```powershell
.\.venv\Scripts\python.exe src/project_tools/check_integration.py
.\.venv\Scripts\python.exe src/project_tools/check_live_backends.py qwen
.\.venv\Scripts\python.exe src/project_tools/check_live_backends.py sherpa
.\.venv\Scripts\python.exe src/project_tools/check_live_backends.py whisper.cpp
.\.venv\Scripts\python.exe src/project_tools/check_live_backends.py roformer
.\.venv\Scripts\python.exe src/project_tools/check_gui_flow.py qwen
.\.venv\Scripts\python.exe src/project_tools/check_gui_flow.py sherpa --separation
.\.venv\Scripts\python.exe src/project_tools/check_gui_flow.py whisper.cpp
```

输出和截图在 `.cache/checks/`，GUI 测试不会保存用户配置，也不会使用麦克风。回放 WAV 和临时字幕由现有应用流程写入 `temp/`。
