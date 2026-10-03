"""Load optional backend in a background task, publishing only a ready model."""
from PySide6.QtCore import Signal
from faster_whisper_GUI.tasks.base import GuardedWorker
from faster_whisper_GUI.backends.catalog import load_backend


class BackendLoadWorker(GuardedWorker):
    setStatusSignal = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.model = None
        self.is_running = False
        self._stop_requested = False

    def run(self):
        if self._stop_requested:
            self.setStatusSignal.emit(False)
            return
        self.is_running = True
        try:
            self.model = load_backend(self.settings)
            if self._stop_requested:
                if hasattr(self.model, "close"):
                    self.model.close()
                self.model = None
                self.setStatusSignal.emit(False)
            else:
                self.setStatusSignal.emit(True)
        finally:
            self.is_running = False

    def onError(self, exc):
        self.is_running = False
        self.setStatusSignal.emit(False)

    def stop(self):
        self._stop_requested = True
        self.is_running = False
