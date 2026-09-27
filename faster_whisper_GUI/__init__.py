import sys as _sys

import whisperx

# logging_setup 可能在包被导入之前就已按文件路径加载（见 FasterWhisperGUI.py 的引导：
# 日志必须早于 ctranslate2 建立，而 ctranslate2 正是被下面这行 whisperx 间接拉起来的）。
# 那种情况下 sys.modules 里已经有了同一个实例，但包属性还没挂上，
# `import faster_whisper_GUI.logging_setup as x` 会拿不到。这里补挂一次。
_logging_setup = _sys.modules.get("faster_whisper_GUI.logging_setup")
if _logging_setup is not None and "logging_setup" not in globals():
    logging_setup = _logging_setup
