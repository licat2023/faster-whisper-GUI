"""Queue-driven native streaming with stateful resampling and revisable text."""
import queue
import numpy as np
from PySide6.QtCore import Signal
from faster_whisper_GUI.tasks.base import GuardedWorker


class NativeStreamWorker(GuardedWorker):
    signal_update = Signal(object)
    Signal_process_over = Signal(list)

    def __init__(self, backend, audio_queue, source_rate, wav_path, parameters, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.audio_queue = audio_queue
        self.source_rate = source_rate
        self.wav_path = wav_path
        self.parameters = parameters
        self.is_running = False
        self._stop_requested = False

    def run(self):
        if self._stop_requested:
            self.Signal_process_over.emit([])
            return
        self.is_running = True
        session = self.backend.start()
        resampler = None
        if self.source_rate != self.backend.sample_rate:
            from faster_whisper_GUI.backends.audio import StreamingResampler
            resampler = StreamingResampler(self.source_rate, self.backend.sample_rate)
        previous = -1
        try:
            while self.is_running:
                try:
                    frame = self.audio_queue.get(timeout=.2)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                if resampler:
                    frame = resampler.process(frame)
                if len(frame):
                    update = session.accept(frame)
                    if update.revision != previous:
                        self.signal_update.emit(update)
                        previous = update.revision
            if not self.is_running:
                self.Signal_process_over.emit([])
                return
            if resampler:
                tail = resampler.finish()
                if len(tail):
                    session.accept(tail)
            final = session.finish()
            self.signal_update.emit(final)
            result = self.backend.stream_result(self.parameters)
            segments = list(result.segments)
            self.Signal_process_over.emit([(segments, self.wav_path, result.info)] if segments else [])
        finally:
            self.is_running = False

    def stop(self):
        self._stop_requested = True
        self.is_running = False

    def onError(self, exc):
        self.is_running = False
        self.Signal_process_over.emit([])
