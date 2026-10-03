"""Roformer separation through a dedicated environment and cancellable process."""
from pathlib import Path
from PySide6.QtCore import Signal
from faster_whisper_GUI.tasks.base import GuardedWorker
from faster_whisper_GUI.runtime.model_process import ModelProcess


class RoformerWorker(GuardedWorker):
    signal_vr_over = Signal(bool)
    file_process_status = Signal(dict)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.transport = None
        self.is_running = False
        self._stop_requested = False

    def status(self, task, file='', done=False):
        self.file_process_status.emit({'task': task, 'file': file, 'status': done})

    def run(self):
        if self._stop_requested:
            self.signal_vr_over.emit(False)
            return
        if not self.settings.get("audio"):
            raise ValueError("没有选择有效的音频文件")
        self.is_running = True
        settings = self.settings
        destination = Path(settings['output_path']).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self.transport = ModelProcess(settings['interpreter'])
        try:
            self.status('load model')
            self.transport.request('load', settings={**settings, 'backend': 'roformer', 'output_path': str(destination)})
            for audio in settings['audio']:
                if not self.is_running:
                    break
                self.status('separate sources', audio)
                files = self.transport.request('separate', audio=str(Path(audio).resolve()))
                if not files or not all(Path(file).is_file() and Path(file).stat().st_size > 0 for file in files):
                    raise ValueError('分离模型没有生成完整输出文件')
                self.status('file over', audio, True)
            self.signal_vr_over.emit(self.is_running)
        finally:
            self.is_running = False
            self.transport.close()

    def stop(self):
        self._stop_requested = True
        self.is_running = False
        if self.transport:
            self.transport.terminate()

    def onError(self, exc):
        self.is_running = False
        self.signal_vr_over.emit(False)
