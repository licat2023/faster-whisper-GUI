r"""项目重构回归：轻量导入、Qt 信号和转录到字幕的完整数据流。

    .venv\Scripts\python.exe src/project_tools/check_structure.py

不下载模型、不录音；使用原生 faster-whisper Segment 和可控推理替身。
"""

from __future__ import annotations

import ast
import builtins
from contextlib import chdir
import importlib
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import symtable
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class StructureChecks(unittest.TestCase):
    def test_central_bytecode_cache(self):
        code = """
import sys
from pathlib import Path
import faster_whisper_GUI
expected = Path(sys.argv[1]).resolve()
assert Path(sys.pycache_prefix).resolve() == expected
assert Path(faster_whisper_GUI.__cached__).resolve().is_relative_to(expected)
"""
        environment = os.environ.copy()
        environment.pop("PYTHONPYCACHEPREFIX", None)
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / ".cache" / "pycache")],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            environment["PYTHONPYCACHEPREFIX"] = temporary
            result = subprocess.run(
                [sys.executable, "-c", code, temporary], cwd=ROOT,
                env=environment, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_editable_install_and_entrypoint(self):
        """在仓库之外也能导入应用，导入启动模块不会创建 Qt 应用。"""
        code = """
import sys
from pathlib import Path
import faster_whisper_GUI
from faster_whisper_GUI.app import main
from faster_whisper_GUI.runtime.paths import PROJECT_ROOT
assert Path(faster_whisper_GUI.__file__).resolve() == Path(sys.argv[1])
assert PROJECT_ROOT == Path(sys.argv[2])
assert callable(main)
assert 'PySide6' not in sys.modules and 'torch' not in sys.modules
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-c", code,
                 str(ROOT / "src" / "faster_whisper_GUI" / "__init__.py"), str(ROOT)],
                cwd=temporary, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lightweight_imports(self):
        """字幕序列化和参数工具可以在没有 Qt/GPU 初始化的进程里使用。"""
        code = """
import sys
sys.path.insert(0, "src")
import faster_whisper_GUI
from faster_whisper_GUI.domain.parameters import WhisperParameters
from faster_whisper_GUI.runtime.timecode import secondsToHMS
from faster_whisper_GUI.subtitles.writers import writeSRT
assert secondsToHMS(1.25) == '00:00:01,250'
assert not any(name in sys.modules for name in
               ['torch', 'ctranslate2', 'faster_whisper', 'whisperx', 'PySide6'])
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_module_globals_and_import_direction(self):
        """捕获拆分后漏导入的辅助函数，并禁止新实现依赖旧兼容层。"""
        package = ROOT / "src" / "faster_whisper_GUI"
        canonical = {"ui", "tasks", "transcription", "subtitles", "domain", "runtime", "backends"}
        stable = {"config", "logging_setup", "version"}
        for directory in sorted(canonical):
            for path in sorted((package / directory).rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                table = symtable.symtable(source, str(path), "exec")
                known = set(dir(builtins)) | {
                    "__name__", "__file__", "__package__", "__class__",
                    "__annotations__", "__conditional_annotations__",
                }
                known.update(
                    s.get_name() for s in table.get_symbols()
                    if s.is_assigned() or s.is_imported() or s.is_namespace()
                )

                def visit(scope):
                    for symbol in scope.get_symbols():
                        if symbol.is_referenced() and symbol.is_global():
                            self.assertIn(symbol.get_name(), known,
                                          f"{path}:{scope.get_name()}")
                    for child in scope.get_children():
                        visit(child)

                visit(table)
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        parts = node.module.split(".")
                        if parts[0] == "faster_whisper_GUI" and len(parts) > 1:
                            self.assertIn(parts[1], canonical | stable, str(path))

    def test_canonical_modules_import(self):
        import faster_whisper_GUI

        prefixes = ("ui", "tasks", "transcription", "subtitles", "domain", "runtime", "backends")
        for item in pkgutil.walk_packages(
            faster_whisper_GUI.__path__, "faster_whisper_GUI.",
        ):
            if item.name.split(".")[1] in prefixes:
                importlib.import_module(item.name)

    def test_runtime_paths(self):
        from faster_whisper_GUI.runtime import rocm
        from faster_whisper_GUI import logging_setup
        from faster_whisper_GUI.runtime.paths import PROJECT_ROOT

        self.assertEqual(PROJECT_ROOT, ROOT)
        self.assertEqual(logging_setup.BASE_DIR, ROOT)
        self.assertEqual(Path(rocm.__file__).resolve().parents[3], ROOT)

    def test_window_contract(self):
        from faster_whisper_GUI.ui.window.main import MainWindows

        self.assertGreaterEqual(
            MainWindows.staticMetaObject.indexOfSignal("signal_guiLog(QString)"), 0,
        )
        self.assertTrue(callable(MainWindows.getParamTranscribe))
        self.assertTrue(callable(MainWindows.saveConfig))
        self.assertTrue(callable(MainWindows.transcribeProcess))
        self.assertTrue(callable(MainWindows.outputSubtitleFile))
        self.assertEqual(MainWindows.__name__, "MainWindows")

    def test_file_transcription_and_export(self):
        """任务编排 -> 原生 Segment 归一化 -> 临时 SRT -> JSON 导出再读取。"""
        from faster_whisper_GUI.transcription.file import TranscribeWorker
        from faster_whisper_GUI.tasks.export import OutputWorker
        from faster_whisper_GUI.domain.parameters import WhisperParameters
        from faster_whisper_GUI.domain.segments import segment_Transcribe
        from faster_whisper_GUI.subtitles.readers import readJSONFileToSegments
        from faster_whisper.transcribe import Segment, Word

        source = Segment(
            id=0, seek=0, start=0.25, end=1.0, text=" hello",
            tokens=[1], avg_logprob=-0.1, compression_ratio=1.0,
            no_speech_prob=0.0, words=[Word(0.25, 1.0, " hello", 0.9)],
            temperature=0.0,
        )
        info = SimpleNamespace(language="en", language_probability=1.0,
                               duration=1.0, duration_after_vad=1.0)
        calls = []

        class Model:
            def transcribe(self, **kwargs):
                calls.append(kwargs)
                return iter([source]), info

        parameters = {name: getattr(WhisperParameters, name)
                      for name in WhisperParameters.__annotations__}
        parameters.update(language="en", task=0)
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
                parameters["audio"] = ["sample.wav"]
                worker = TranscribeWorker(model=Model(), parameters=parameters,
                                          vad_parameters={})
                completed = []
                worker.signal_process_over.connect(completed.append)
                worker.runTranscribe()
                self.assertEqual(len(completed), 1)
                self.assertEqual(calls[0]["audio"], "sample.wav")
                self.assertEqual(calls[0]["task"], "transcribe")
                segments = completed[0][0][0]
                self.assertIsInstance(segments[0], segment_Transcribe)
                self.assertIsNone(segments[0].speaker)
                srt = Path("temp/sample.srt").read_text(encoding="utf-8")
                self.assertIn("00:00:00,250 --> 00:00:01,000", srt)
                self.assertIn("hello", srt)

                exporter = OutputWorker(completed[0], "exports", "JSON")
                export_finished = []
                exporter.signal_write_over.connect(lambda: export_finished.append(True))
                exporter.run()
                self.assertIsNone(exporter.lastError)
                self.assertEqual(export_finished, [True])
                self.assertFalse(exporter.is_running)
                restored = readJSONFileToSegments("exports/sample.json")
                self.assertEqual(restored[0].text, source.text)
                self.assertEqual(restored[0].start, source.start)
                self.assertEqual(restored[0].words[0].probability, 0.9)
        finally:
            os.chdir(previous)

    def test_configuration_persistence(self):
        """分组后的窗口方法仍保存相同配置键和控件数据。"""
        from faster_whisper_GUI.ui.window.main import MainWindows

        def page(value):
            return SimpleNamespace(getParam=lambda: value)

        window = SimpleNamespace(
            page_model=page({"device": "cpu"}),
            page_setting=page({"saveConfig": True}),
            page_demucs=page({"segment": 10}),
            page_transcribes=page({"language": "en"}),
            page_output=page({"format": "SRT"}),
            page_VAD=page({"threshold": 0.5}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "config.json"
            MainWindows.saveConfig(window, str(target))
            result = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(result["model_param"], {"device": "cpu"})
        self.assertEqual(result["Transcription_param"], {"language": "en"})
        self.assertEqual(set(result), {"theme", "demucs", "model_param", "vad_param",
                                      "setting", "Transcription_param", "output_whisperX"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
