"""麦克风采集与录音文件保存。"""

import logging
from contextlib import ExitStack
import os
import numpy as np
from PySide6.QtCore import Signal, QDateTime
from faster_whisper_GUI.tasks.base import GuardedWorker
import sounddevice as sd
import wave

log = logging.getLogger(__name__)

class CaptureAudioWorker(GuardedWorker):
    Signal_process_over = Signal(np.ndarray)

    def __init__(self
                , parent=None
                , rate = 48000
                , channels = 2
                , dType = 16
                , audio_queue : "queue.Queue" = None
                , wav_path : str = ""
            ) -> None:
        
        super().__init__(parent)
        self.rate = rate
        self.channels = channels
        self.dType = dType
        self.is_running = False
        self._stop_requested = False
        # sounddevice 的采样格式名，对应原先的 paInt16 / paInt24
        self.format_capture = {16: "int16", 24: "int24"}
        # 每样本字节数。实测两种库都是紧凑排布（int24 为 3 字节，不是 4 字节填充）
        self.sample_width = {16: 2, 24: 3}
        self.buffer_size = 2048
        # 实时转写时把音频同时推到这个队列（由 AudioStreamTranscribeWorker 消费）；
        # 为 None 则只录音，保持原来的行为
        self.audio_queue = audio_queue
        # 录出的文件路径。由调用方生成并同时交给实时转写线程，
        # 这样停止后能按普通文件走既有的结果展示/导出流程
        self.wav_path = wav_path

    def _toMonoFloat32(self, raw: bytes) -> np.ndarray:
        """把采集到的原始 PCM 转成单声道 float32（范围 -1..1）。"""
        if self.dType == 16:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            # 24-bit 是紧凑的 3 字节小端，numpy 没有原生类型，手动拼
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
            data = v.astype(np.float32) / 8388608.0
        if self.channels > 1:
            data = data.reshape(-1, self.channels).mean(axis=1)
        return data
    
    def run(self):
        try:
            if self._stop_requested:
                return
            self.is_running = True
            with ExitStack() as resources:
                stream = sd.RawInputStream(samplerate=self.rate, channels=self.channels,
                    dtype=self.format_capture[self.dType], blocksize=self.buffer_size)
                resources.callback(stream.close)
                stream.start()
                resources.callback(stream.stop)
                os.makedirs('./temp', exist_ok=True)
                if not self.wav_path:
                    stamp = QDateTime.currentDateTime().toString('yyyy-MM-dd-hh-mm-ss-zzz')
                    self.wav_path = os.path.abspath(os.path.join('./temp', stamp + '.wav'))
                wf = resources.enter_context(wave.open(self.wav_path, 'wb'))
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width[self.dType])
                wf.setframerate(self.rate)
                while self.is_running:
                    data, overflowed = stream.read(self.buffer_size)
                    if overflowed:
                        log.warning('[录音] 输入缓冲溢出，音频可能有丢失')
                    raw = bytes(data)
                    wf.writeframes(raw)
                    if self.audio_queue is not None:
                        self.audio_queue.put(self._toMonoFloat32(raw))
        finally:
            self.is_running = False
            if self.audio_queue is not None:
                self.audio_queue.put(None)
    def onError(self, exc):
        self.is_running = False

    def stop(self):
        self._stop_requested = True
        self.is_running = False
