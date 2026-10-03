"""Opt-in real Qt window/worker checks, using replayed audio instead of a microphone."""

if not __debug__:
    raise RuntimeError("Run verification without -O or PYTHONOPTIMIZE; assertions are required")

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def main(identity, separation=False, ui_only=False):
    from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
    prepare_inference_runtime()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QEventLoop, QTimer, Qt
    from faster_whisper_GUI.ui.window.main import MainWindows
    from faster_whisper_GUI.ui.window.view import UIMainWin
    from faster_whisper_GUI.backends.catalog import default_settings
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    app = QApplication.instance() or QApplication([])
    destination = ROOT / '.cache/checks/gui-flow' / identity
    destination.mkdir(parents=True, exist_ok=True)

    class Window(MainWindows):
        def readConfigJson(self, config_file_path=''):
            UIMainWin.readConfigJson(self, '')
        def onModelLoadClicked(self):
            if not getattr(self, '_booting', False):
                super().onModelLoadClicked()
        def closeEvent(self, event):
            event.accept()  # Do not write the user's configuration during verification.

    window = Window.__new__(Window)
    window._booting = True
    Window.__init__(window)
    window._booting = False
    window.page_setting.combox_autoGoToOutputPage.setCurrentIndex(1)
    window.show()

    def wait(worker, seconds=240):
        if worker.isRunning():
            loop = QEventLoop()
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            worker.finished.connect(loop.quit)
            timer.start(seconds * 1000)
            loop.exec()
            assert not worker.isRunning(), 'Worker timeout'
        app.processEvents()
        assert worker.lastError is None, worker.lastError

    try:
        window.page_model.backend_combox.setCurrentIndex(window.page_model.backend_combox.findData(identity))
        assert window.page_transcribes.backend_options.identity == identity
        window.stackedWidget.setCurrentWidget(window.page_model)
        app.processEvents()
        window.grab().save(str(destination / 'model.png'))
        if ui_only:
            window.stackedWidget.setCurrentWidget(window.page_transcribes)
            app.processEvents()
            window.grab().save(str(destination / 'parameters.png'))
            window.page_demucs.backend_combox.setCurrentIndex(1)
            window.stackedWidget.setCurrentWidget(window.page_demucs)
            app.processEvents()
            window.grab().save(str(destination / 'separation.png'))
            print('PASS: GUI panels rendered')
            return
        window.onModelLoadClicked()
        wait(window.loadModelWorker)
        assert window.FasterWhisperModel is not None and window.loaded_backend_id == identity
        assert window.page_model.button_model_lodar.isEnabled()
        source = ROOT / '.cache/fixtures/qwen-zh.wav'
        window.page_process.fileNameListView.FileNameModle.setStringList([str(source)])
        window.transcribeProcess()
        worker = window.transcribe_thread
        assert worker is not None
        wait(worker)
        assert window.current_result and window.tableModel_list
        assert window.transcribe_thread is None and window.page_process.button_process.isEnabled()
        if identity == 'qwen':
            assert all(s.words for s in window.current_result[0][0])
        window.stackedWidget.setCurrentWidget(window.page_output)
        app.processEvents()
        window.grab().save(str(destination / 'subtitles.png'))
        if identity == 'qwen':
            table = next(iter(window.tableModel_list.values()))
            corrected = window.current_result[0][0][0].text + '（校订）'
            assert table.setData(table.index(0, 3), corrected, Qt.ItemDataRole.EditRole)
            assert not window.current_result[0][0][0].words
            window.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(destination / 'edited'))
            for format in ['SRT', 'VTT', 'ASS', 'LRC', 'JSON']:
                window.page_output.combox_output_format.setCurrentText(format)
                window.outputSubtitleFile()
                worker = window.outputWorker
                wait(worker)
                output = destination / 'edited' / ('qwen-zh.' + format.lower())
                assert '校订' in output.read_text(encoding='utf-8-sig')

        if identity == 'sherpa':
            audio, rate = sf.read(source, dtype='float32')
            # CaptureAudioWorker still executes its real PCM conversion and WAV-writing path.
            mono = resample_poly(audio, 48000, rate)
            pcm = np.repeat((mono * 32767).astype('<i2')[:, None], 2, axis=1).tobytes()
            class Input:
                def __init__(self, **kwargs): self.offset = 0
                def start(self): pass
                def stop(self): pass
                def close(self): pass
                def read(self, count):
                    end = min(len(pcm), self.offset + count * 4)
                    chunk = pcm[self.offset:end]
                    self.offset = end
                    if not chunk:
                        window.audio_capture_thread.stop()
                        chunk = b'\0' * (count * 4)
                    return chunk, False
            window.page_process.combox_capture.setCurrentIndex(1)  # 48kHz stereo 16-bit
            window.page_process.audio_capture_RadioButton.setChecked(True)
            window.stackedWidget.setCurrentWidget(window.page_process)
            with patch('faster_whisper_GUI.tasks.capture.sd.RawInputStream', Input):
                window.audioCaptureProcess()
                worker = window.audio_stream_worker
                wait(worker)
            assert window.current_result and window.current_result[0][0]
            assert window.page_process.processResultText.toPlainText()
            app.processEvents()
            window.grab().save(str(destination / 'stream.png'))

        if separation:
            if hasattr(window.FasterWhisperModel, 'close'):
                window.FasterWhisperModel.close()
            window.FasterWhisperModel = None
            window.page_demucs.backend_combox.setCurrentIndex(1)
            window.page_demucs.outputGroupWidget.LineEdit_output_dir.setText(str(destination / 'separation'))
            window.page_demucs.fileListView.setFileNameListToDataModel([
                str(ROOT / '.cache/fixtures/hdemucs_mix_150_155.wav')])
            window.demucsProcess()
            worker = window.demucsWorker
            assert worker is not None
            wait(worker)
            assert window.demucsWorker is None and window.page_demucs.process_button.isEnabled()
        print(f'PASS: real GUI {identity}, model load, file recognition, subtitle table' +
              (', native capture replay' if identity == 'sherpa' else '') + (', Roformer separation' if separation else ''))
    finally:
        if hasattr(window.FasterWhisperModel, 'close'):
            window.FasterWhisperModel.close()
        window.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['qwen', 'sherpa', 'whisper.cpp'])
    parser.add_argument('--separation', action='store_true')
    parser.add_argument('--ui-only', action='store_true')
    args = parser.parse_args()
    main(args.backend, args.separation, args.ui_only)
