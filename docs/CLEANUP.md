# 工作区清理与缓存

应用、内部维护工具及 Qt 资源源码统一位于 `src/`。
根目录保留启动/安装脚本、项目元数据、文档与运行时数据。旧模块兼容导出和根目录 Python
启动器已移除，使用 `uv run python -m faster_whisper_GUI` 或 `启动GUI.bat`。

## 字节码

`.pyc` 是可重建的 Python 字节码，可删除，下一次导入时自动生成。
启动批处理设置 `PYTHONPYCACHEPREFIX`；`setup.ps1` 调用以下工具，在项目
虚拟环境安装启动钩子，直接调用该环境的 Python 时也集中缓存：

```powershell
.venv/Scripts/python.exe -m project_tools.cache_policy
```

缓存集中到 `.cache/pycache/`，Python 按原始路径分层存放，不会把不同目录的
同名模块覆盖。显式设置的 `PYTHONPYCACHEPREFIX` 优先。移动仓库或重建虚拟环境后
重新运行上述命令。系统 Python 和其他虚拟环境不受影响。

清理源码旁的字节码、构建目录及 egg-info：

```powershell
& src/project_tools/clean_generated.ps1
# 也删除集中缓存（下次运行重新生成）：
& src/project_tools/clean_generated.ps1 -ClearBytecodeCache
```

依据：[Python 缓存前缀说明](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPYCACHEPREFIX)。

## 2026-10-01 清理记录

- 移除 30 个旧模块兼容文件；内部导入及回归检查统一使用分组后的模块。
- 移除无引用的 PyAV/Nuitka 辅助和 WhisperX 旧字幕处理模块。
- 清理分散字节码、构建副本及包元数据，保留可重复执行的清理工具。
- 参数说明和提示词参考移至 `docs/reference/`；ROCm 构建指南保留在 `docs/runtime/`。
- 完整翻译源合并到 `src/resource/_rc/Translater/en.ts`；旧翻译源归档。
- `.probe`、`.hip-verify`、`.crw`、一次性日志迁移脚本、未使用的代理配置及
  旧配置备份归档到工作区外：
  `C:\Users\licat\AppData\Local\faster-whisper-GUI\workspace-archives\20261001-133950`。

模型缓存已从 `cache/` 迁移至 `.cache/models/`，文件内容通过 SHA256 校验。录音与临时字幕 `temp/`、日志 `logs/`、当前配置及虚拟环境保留。
这里的清理不涉及第三方依赖升级或自建 ROCm DLL 替换。

## 统一缓存目录

项目内部缓存统一在 `.cache/`：`models/` 存模型权重，`pycache/` 存 Python 字节码，
`build-dist/` 存打包验证产物。对齐与人声分离使用 `runtime/paths.py` 定义的绝对模型缓存路径，
不依赖调用时的工作目录。清理工具默认保留模型；`-ClearBytecodeCache` 也只删除字节码缓存。
用户手动指定的模型目录及用户主目录的 Hugging Face 共享缓存保持原配置。
