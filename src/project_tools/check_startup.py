"""GUI startup regressions; optional real ROCm faster-whisper -> sherpa switching."""

if not __debug__:
    raise RuntimeError("Run verification without -O or PYTHONOPTIMIZE; assertions are required")

import argparse
import os
from pathlib import Path
import sys
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
START = time.perf_counter()
from faster_whisper_GUI import logging_setup
DEST = ROOT / '.cache/checks/startup-fixed'
DEST.mkdir(parents=True, exist_ok=True)
logging_setup.resolveLogDir = lambda: DEST
logging_setup.pruneOldRuns = lambda *args, **kwargs: []
from faster_whisper_GUI.ui.window.main import MainWindows
from faster_whisper_GUI.ui.window.view import UIMainWin
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QLabel
from qfluentwidgets import setTheme, Theme


class Window(MainWindows):
    def readConfigJson(self, *args):
        UIMainWin.readConfigJson(self, '')

    def onModelLoadClicked(self):
        if not getattr(self, '_booting', True):
            super().onModelLoadClicked()

    def closeEvent(self, event):
        event.accept()  # Verification must not save the user's settings.


class StartupChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.window = Window.__new__(Window)
        cls.window._booting = True
        Window.__init__(cls.window)
        cls.window._booting = False
        cls.window.page_setting.combox_autoGoToOutputPage.setCurrentIndex(1)
        cls.window.show()
        cls.app.processEvents()
        print(f'Window visible in {time.perf_counter() - START:.3f}s', file=sys.__stdout__, flush=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls.window.FasterWhisperModel, 'close'):
            cls.window.FasterWhisperModel.close()
        cls.window.close()

    def test_startup_does_not_import_inference_libraries(self):
        self.assertEqual([name for name in ['torch', 'ctranslate2', 'faster_whisper',
                                           'transformers', 'whisperx'] if name in sys.modules], [])

    def test_inherited_rocm_flag_does_not_skip_child_dll_registration(self):
        code = '''
import os
from unittest.mock import patch
from faster_whisper_GUI.runtime import rocm
os.environ[rocm.ROCM_ENV_FLAG] = '1'
os.environ['FASTER_WHISPER_GUI_ROCM_PID'] = '-1'
with patch.object(rocm, 'findROCmRoot', return_value='C:/fake-rocm'), \\
     patch.object(rocm, 'findDependentDllDirectories', return_value=[]), \\
     patch.object(os, 'add_dll_directory', create=True) as register:
    assert rocm.setupROCm()
    assert register.call_count == 1, 'Inherited flag skipped DLL registration'
    assert os.environ['FASTER_WHISPER_GUI_ROCM_PID'] == str(os.getpid())
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_main_window_is_visible_across_backend_selection(self):
        for index in range(self.window.page_model.backend_combox.count()):
            self.window.page_model.backend_combox.setCurrentIndex(index)
            self.app.processEvents()
            visible = [widget for widget in self.app.topLevelWidgets() if widget.isVisible()]
            self.assertEqual(visible, [self.window])
            for name in ['button_convert_model', 'button_set_model_out_dir', 'LineEdit_model_out_dir']:
                widget = getattr(self.window.page_model, name)
                self.assertIsNotNone(widget.parent())
                self.assertTrue(widget.isHidden())

    def test_separation_label_colors_in_both_themes(self):
        self.window.stackedWidget.setCurrentWidget(self.window.page_demucs)
        for theme in [Theme.LIGHT, Theme.DARK, Theme.LIGHT]:
            setTheme(theme, save=False, lazy=False)
            for backend in [0, 1]:
                self.window.page_demucs.backend_combox.setCurrentIndex(backend)
                self.app.processEvents()
                labels = [label for label in self.window.page_demucs.findChildren(QLabel)
                          if label.isVisible() and label.text()]
                self.assertTrue(labels)
                for label in labels:
                    color = label.palette().color(QPalette.ColorRole.WindowText)
                    self.assertLess(color.lightness(), 150) if theme == Theme.LIGHT else self.assertGreater(color.lightness(), 150)
                self.window.grab().save(str(DEST / f'separation-{theme.value.lower()}-{backend}.png'))


def check_switch(model, device):
    """Exercise the real GUI loader, recognition, switch, unload and child-failure paths."""
    StartupChecks.setUpClass()
    w, app = StartupChecks.window, StartupChecks.app

    def wait(worker):
        loop, timer = QEventLoop(), QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        worker.finished.connect(loop.quit)
        timer.start(240000)
        if worker.isRunning():
            loop.exec()
        assert not worker.isRunning(), 'Worker timeout'
        app.processEvents()
        assert worker.lastError is None, worker.lastError

    def load(identity):
        w.page_model.backend_combox.setCurrentIndex(w.page_model.backend_combox.findData(identity))
        w.onModelLoadClicked()
        wait(w.loadModelWorker)
        assert w.FasterWhisperModel is not None and w.loaded_backend_id == identity
        return w.FasterWhisperModel

    def recognize():
        w.page_process.fileNameListView.FileNameModle.setStringList([str(ROOT / '.cache/fixtures/qwen-zh.wav')])
        w.transcribeProcess()
        wait(w.transcribe_thread)
        assert w.current_result and w.current_result[0][0]
        print('Recognized: ' + ''.join(s.text for s in w.current_result[0][0]), file=sys.__stdout__, flush=True)

    try:
        w.page_model.lineEdit_model_path.setText(str(Path(model).resolve()))
        w.page_model.model_local_RadioButton.setChecked(True)
        w.page_model.setDevice(device)
        w.page_model.preciese_combox.setCurrentText('int8')
        w.page_model.switchButton_use_v3.setChecked('v3' in str(model))
        w.page_model.switchButton_local_files_only.setChecked(True)
        # Cover VAD and word timestamp serialization, including Chinese punctuation.
        w.page_transcribes.combox_language.setCurrentIndex(
            next(i for i in range(w.page_transcribes.combox_language.count())
                 if w.page_transcribes.combox_language.itemText(i).lower().startswith('zh')))
        w.page_transcribes.switchButton_word_level_timestampels.setChecked(True)
        first = load('faster-whisper')
        assert first.transport.process.pid != os.getpid()
        recognize()
        assert any(s.words for s in w.current_result[0][0])
        load('sherpa')
        assert first.transport.process.poll() is not None
        recognize()
        second = load('faster-whisper')
        recognize()
        w.unloadWhisperModel()
        assert w.FasterWhisperModel is None and second.transport.process.poll() is not None
        third = load('faster-whisper')
        third.transport.process.kill()
        third.transport.process.wait(timeout=5)
        w.transcribeProcess()
        worker = w.transcribe_thread
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(30000, loop.quit)
        if worker.isRunning():
            loop.exec()
        app.processEvents()
        assert not worker.isRunning() and worker.lastError is not None
        assert w.FasterWhisperModel is None and w.page_process.button_process.isEnabled()
        load('sherpa')
        recognize()
        fourth = load('faster-whisper')
        w.transcribeProcess()
        cancelled_worker = w.transcribe_thread
        QTimer.singleShot(100, w.cancelTrancribe)
        loop = QEventLoop()
        cancelled_worker.finished.connect(loop.quit)
        QTimer.singleShot(30000, loop.quit)
        if cancelled_worker.isRunning():
            loop.exec()
        app.processEvents()
        assert not cancelled_worker.isRunning()
        assert w.FasterWhisperModel is None and w.page_process.button_process.isEnabled()
        assert fourth.transport.process.poll() is not None
        load('sherpa')
        assert not any(name in sys.modules for name in ['torch', 'ctranslate2', 'faster_whisper'])
        print('PASS: real GUI FW -> sherpa -> FW, recognition, unload, native child failure, cancellation and recovery', file=sys.__stdout__)
    finally:
        StartupChecks.tearDownClass()


def check_entrypoint():
    """Run the actual launcher entry, splash, saved configuration and Windows GUI."""
    from unittest.mock import patch
    from faster_whisper_GUI.app import main
    original_exec = QApplication.exec
    failures = []
    inspected = []

    def enter_loop(app):
        def inspect():
            inspected.append(True)
            try:
                visible = [widget for widget in app.topLevelWidgets() if widget.isVisible()]
                assert len(visible) == 1, [type(widget).__name__ for widget in visible]
                assert not any(name in sys.modules for name in ['torch', 'ctranslate2', 'whisperx', 'transformers'])
                visible[0].grab().save(str(DEST / 'entrypoint.png'))
                print(f'PASS actual entrypoint: one window, no inference imports, {time.perf_counter() - START:.3f}s',
                      file=sys.__stdout__, flush=True)
            except Exception as error:
                failures.append(error)
            finally:
                app.quit()
        QTimer.singleShot(250, inspect)
        return original_exec()

    # Loading models and saving settings are tested separately, preserving user config here.
    with patch.object(MainWindows, 'onModelLoadClicked', lambda self: None), \
         patch.object(MainWindows, 'closeEvent', lambda self, event: event.accept()), \
         patch.object(QApplication, 'exec', enter_loop):
        try:
            main()
        except SystemExit as error:
            assert error.code == 0
    assert inspected, "entrypoint callback did not run"
    assert not failures, failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Opt in to real backend checks with a local CT2 model directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--entrypoint', action='store_true')
    args = parser.parse_args()
    if args.entrypoint:
        check_entrypoint()
    elif args.model:
        check_switch(args.model, args.device)
    else:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
