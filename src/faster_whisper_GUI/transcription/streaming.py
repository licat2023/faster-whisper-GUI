"""实时分块转录与时间戳归一化。"""

from faster_whisper_GUI.runtime.inference import prepare_inference_runtime

prepare_inference_runtime()


import logging
import queue
from dataclasses import replace as _dcReplace
from typing import List
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from PySide6.QtCore import Signal
from faster_whisper_GUI.tasks.base import GuardedWorker
from faster_whisper_GUI.backends.faster_whisper import FasterWhisperBackend
from faster_whisper_GUI.domain.segments import segment_Transcribe

from faster_whisper_GUI.transcription.parameters import buildTranscribeKwargs

log = logging.getLogger(__name__)

class AudioStreamTranscribeWorker(GuardedWorker):
    """
    实时转写：消费 CaptureAudioWorker 推来的音频块，滚动分块送去 faster-whisper。

    faster-whisper 没有真正的流式 API —— 它一次吃完一段音频。实时字幕的通行做法
    是把音频切成若干秒一块逐块转写，再把每块的时间戳平移到全局时间轴。

    为什么必须重采样：Whisper 前端硬编码 16 kHz（mel 滤波器组只覆盖 40-7960 Hz），
    而采集按用户选的音质档（44.1/48 kHz）进行。实测重采样开销约 0.1% 实时，
    且 48k→16k 转写的输出与直接用原生 16k 逐字一致 —— 模型本来就看不到 8 kHz
    以上的内容。不重采样反而会让时间轴压缩 3 倍、输出变成胡言乱语且慢 6.5 倍。

    块长（chunk_seconds）默认 30 秒，这是实测出来的最优值，不是随手定的。
    同一段 58.6 秒真实语音，与整段文件转写对比：

        块长    段数   耗时(占实时)   词数      字符相似度
        8s      10    14.3s (24%)   96/105       83%
        15s      9    17.7s (30%)   83/105       70%
        30s      9     7.3s (12%)  105/105      100%

    原因是 Whisper 的输入窗口本来就是 30 秒，而且**每段输入都会被填充到 30 秒**：
    一个 8 秒的块要花约 1.15 秒，一个 30 秒的块才 2.57 秒 —— 切得越碎，
    固定开销占比越高，同时边界切在词/数字串中间造成的伪影也越多。
    30 秒块还能给出与文件转写 100% 一致的输出。

    代价是首段结果要等约 30 秒。调小 chunk_seconds 可以换更快的反馈，
    但要接受上面那张表里的质量下降。每调用一次 model.transcribe() 约有 1 秒
    固定开销，所以分块数量直接决定总代价。
    """

    # 结束时发 [(segments, wav_path, info)]，与 TranscribeWorker 的格式一致，
    # 以便直接复用现有的结果展示与导出流程
    Signal_process_over = Signal(list)
    # 实时增量：当前累计的全部 segments，供界面边录边刷新
    signal_segments = Signal(list)

    def __init__(self
                , parent = None
                , model : WhisperModel = None
                , parameters : dict = None
                , vad_filter : bool = False
                , vad_parameters : dict = None
                , num_workers : int = 1
                , output_format : str = "srt"
                , output_dir : str = ""
                , audio_queue : "queue.Queue" = None
                , source_rate : int = 48000
                , chunk_seconds : float = 30.0
                , wav_path : str = ""
            ) -> None:
        super().__init__(parent)

        self.is_running = True
        self.model = model
        self.backend = model if hasattr(model, "recognize") else FasterWhisperBackend(model)
        self.parameters = parameters
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters
        self.num_workers = num_workers
        self.output_format = output_format
        self.output_dir = output_dir
        self.audio_queue = audio_queue
        self.source_rate = source_rate
        self.chunk_seconds = chunk_seconds
        self.wav_path = wav_path

        self.segments = []
        self.signal_segments_count = 0
        self._last_info = None

    # ---------------------------------------------------------------- 内部
    def _to16k(self, audio: np.ndarray) -> np.ndarray:
        """把采集采样率下的单声道 float32 重采样到 Whisper 要求的 16 kHz。"""
        target = self.backend.sample_rate
        if self.source_rate == target:
            return audio
        return torchaudio.functional.resample(
            torch.from_numpy(audio), self.source_rate, target
        ).numpy()

    def _shiftSegments(self, segments, offset: float) -> list:
        """
        把一块音频的时间戳平移到全局时间轴。

        先规范化任意后端片段，再平移应用片段和词的时间戳。
        words 里每个 Word 也有自己的时间戳，必须一起平移 —— 否则
        VTT/LRC/SMI 的逐字歌词时间轴会全部错位。
        """
        out = []
        for native in segments:
            seg = segment_Transcribe(native)
            seg.start += offset
            seg.end += offset
            seg.words = [_dcReplace(word, start=word.start + offset, end=word.end + offset)
                         for word in seg.words]
            out.append(seg)
        return out

    def _transcribeChunk(self, audio: np.ndarray, offset: float) -> list:
        kwargs = buildTranscribeKwargs(self.parameters, self.vad_filter, self.vad_parameters)
        # 实时字幕必须要有时间戳，否则一块只能落成一条、无法导出成正常字幕
        kwargs["without_timestamps"] = False
        recognition = self.backend.recognize(self._to16k(audio), kwargs)
        if not recognition.segment_timestamps:
            raise ValueError("分块字幕识别需要时间戳；纯文字请使用原生流式会话")
        self._last_info = recognition.info
        return self._shiftSegments(recognition.segments, offset)

    # ------------------------------------------------------------------ 线程
    def run(self):
        buffered: List[np.ndarray] = []
        buffered_samples = 0
        offset = 0.0                     # 已转写的音频秒数
        chunk_samples = max(1, int(self.chunk_seconds * self.source_rate))
        audio_seconds = 0.0
        log.info("%s", f"[实时] 开始：{self.source_rate} Hz -> 16 kHz，每 {self.chunk_seconds:.0f}s 转写一块")

        try:
            while True:
                try:
                    block = self.audio_queue.get(timeout=0.2)
                    if block is None:
                        self.is_running = False
                except queue.Empty:
                    block = None

                if block is not None and block.size:
                    buffered.append(block)
                    buffered_samples += block.size

                stopping = not self.is_running
                queue_drained = self.audio_queue.empty()

                # 停止后必须等队列排空再收尾。若写成 (stopping and buffered_samples > 0)，
                # 队列里每剩一个块都会立刻满足条件，等于每个 2048 帧的块都调用一次
                # model.transcribe() —— 每次约 1 秒固定开销，积压上千块就是十几分钟。
                if (buffered_samples >= chunk_samples
                        or (stopping and queue_drained and buffered_samples > 0)):
                    audio = np.concatenate(buffered) if buffered else np.zeros(0, np.float32)
                    buffered.clear()
                    buffered_samples = 0
                    new = self._transcribeChunk(audio, offset)
                    audio_seconds += audio.size / self.source_rate
                    offset += audio.size / self.source_rate
                    if new:
                        self.segments.extend(new)
                        self.signal_segments.emit(list(self.segments))

                if stopping and buffered_samples == 0 and self.audio_queue.empty():
                    break
        finally:
            self.is_running = False
            log.info("%s", f"[实时] 结束：音频 {audio_seconds:.1f}s，共 {len(self.segments)} 段")

        if self.segments:
            self.Signal_process_over.emit([(self.segments, self.wav_path, self._last_info)])
        else:
            # 没有识别到任何内容（例如全程静音）——发空列表，让界面走"无结果"分支
            log.info("%s", "[实时] 未识别到语音内容")
            self.Signal_process_over.emit([])

    def onError(self, exc):
        self.is_running = False
        self.Signal_process_over.emit([])

    def stop(self):
        self.is_running = False
