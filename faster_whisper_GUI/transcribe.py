# coding:utf-8

# from threading import Thread
from concurrent import futures
import logging
import os
import queue
from dataclasses import replace as _dcReplace
from typing import List
import codecs
import torch
import torchaudio
import numpy as np 
import av
import json
import hashlib

from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionInfo

import webvtt
from PySide6.QtCore import (QThread, Signal, QDateTime)

from faster_whisper_GUI.workers import GuardedWorker
# 麦克风录音用 sounddevice（PortAudio 绑定）。
# 原先用 PyAudio：它只发 cp3X 专用 wheel，每出一个新 Python 版本都要等维护者
# 重新构建 —— Python 3.14 就是因此被卡住的（无 cp314 wheel，退回源码构建又缺
# PortAudio 头文件，报 fatal error C1083: portaudio.h）。
# sounddevice 发的是纯 Python wheel 并把编好的 PortAudio DLL 捆在包里，
# 与 Python 版本无关；设备覆盖与 PyAudio 一致（同一套 PortAudio）。
import sounddevice as sd
import wave

from .config import (
                    Language_dict
                    , SUBTITLE_FORMAT
                    , Language_without_space
                )

from .seg_ment import segment_Transcribe
from .util import (
                    secondsToHMS, 
                    secondsToMS, 
                    WhisperParameters
                )

from .config import ENCODING_DICT, Task_list

log = logging.getLogger(__name__)


def buildTranscribeKwargs(parameters: dict, vad_filter: bool, vad_parameters: dict) -> dict:
    """
    把 GUI 的参数字典翻译成 WhisperModel.transcribe() 的关键字参数。

    文件转写与实时流式转写共用这一份：原先这 30 多个参数只写在 TranscribeWorker
    里，实时转写再抄一份的话两边会逐渐漂移。
    """
    return dict(
        language = parameters["language"],
        task = Task_list[int(parameters["task"])],
        log_progress = False,
        beam_size = parameters["beam_size"],
        best_of = parameters["best_of"],
        patience = parameters["patience"],
        length_penalty = parameters["length_penalty"],
        temperature = parameters["temperature"],
        compression_ratio_threshold = parameters["compression_ratio_threshold"],
        log_prob_threshold = parameters["log_prob_threshold"],
        no_speech_threshold = parameters["no_speech_threshold"],
        condition_on_previous_text = parameters["condition_on_previous_text"],
        initial_prompt = parameters["initial_prompt"],
        prefix = parameters["prefix"],
        repetition_penalty = parameters["repetition_penalty"],
        no_repeat_ngram_size = parameters["no_repeat_ngram_size"],
        prompt_reset_on_temperature = parameters["prompt_reset_on_temperature"],
        suppress_blank = parameters["suppress_blank"],
        suppress_tokens = parameters["suppress_tokens"],
        without_timestamps = parameters["without_timestamps"],
        max_initial_timestamp = parameters["max_initial_timestamp"],
        word_timestamps = parameters["word_timestamps"],
        prepend_punctuations = parameters["prepend_punctuations"],
        append_punctuations = parameters["append_punctuations"],
        multilingual = parameters["multilingual"],
        max_new_tokens = parameters["max_new_tokens"],
        chunk_length = parameters["chunk_length"],
        clip_timestamps = parameters["clip_timestamps"],
        hallucination_silence_threshold = parameters["hallucination_silence_threshold"],
        hotwords = parameters["hotwords"],
        language_detection_threshold = parameters["language_detection_threshold"],
        language_detection_segments = parameters["language_detection_segments"],
        vad_filter = vad_filter,
        vad_parameters = vad_parameters,
    )


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

        self.is_running = False
        self.model = model
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
        target = self.model.feature_extractor.sampling_rate
        if self.source_rate == target:
            return audio
        return torchaudio.functional.resample(
            torch.from_numpy(audio), self.source_rate, target
        ).numpy()

    def _shiftSegments(self, segments, offset: float) -> list:
        """
        把一块音频的时间戳平移到全局时间轴。

        Segment 是 dataclass（不是 namedtuple），用 dataclasses.replace。
        words 里每个 Word 也有自己的时间戳，必须一起平移 —— 否则
        VTT/LRC/SMI 的逐字歌词时间轴会全部错位。
        """
        out = []
        for seg in segments:
            words = None
            if seg.words:
                words = [_dcReplace(w, start=w.start + offset, end=w.end + offset)
                         for w in seg.words]
            out.append(_dcReplace(seg, start=seg.start + offset, end=seg.end + offset,
                                  words=words))
        return out

    def _transcribeChunk(self, audio: np.ndarray, offset: float) -> list:
        kwargs = buildTranscribeKwargs(self.parameters, self.vad_filter, self.vad_parameters)
        # 实时字幕必须要有时间戳，否则一块只能落成一条、无法导出成正常字幕
        kwargs["without_timestamps"] = False
        segments, info = self.model.transcribe(audio=self._to16k(audio), **kwargs)
        self._last_info = info
        return self._shiftSegments(segments, offset)

    # ------------------------------------------------------------------ 线程
    def run(self):
        self.is_running = True
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
            log.info("%s", f"[实时] 结束：音频 {audio_seconds:.1f}s，共 {len(self.segments)} 段")

        if self.segments:
            self.Signal_process_over.emit([(self.segments, self.wav_path, self._last_info)])
        else:
            # 没有识别到任何内容（例如全程静音）——发空列表，让界面走"无结果"分支
            log.info("%s", "[实时] 未识别到语音内容")
            self.Signal_process_over.emit([])

    def stop(self):
        self.is_running = False

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
        self.is_running = True
        # 不传 device 即使用系统默认输入设备，与迁移前行为一致
        stream = sd.RawInputStream(samplerate=self.rate
                            , channels=self.channels
                            , dtype=self.format_capture[self.dType]
                            , blocksize=self.buffer_size
                        )
        stream.start()

        temp_path = r"./temp"
        if not os.path.exists(os.path.abspath(temp_path)):
            os.mkdir(os.path.abspath(temp_path))

        # 路径一般由调用方给（实时转写需要和转写线程共用同一个）；
        # 没给就按时间戳自己生成，保持单独使用时也能工作
        if not self.wav_path:
            currentDateTime = QDateTime.currentDateTime().toString("yyyy-MM-dd-hh-mm-ss")
            self.wav_path = os.path.join(os.path.abspath(temp_path)
                                    ,f"{currentDateTime}.wav"
                                ).replace("\\", "/")
        wav_path = self.wav_path
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.sample_width[self.dType])
        wf.setframerate(self.rate)
        try:
            # 边采边写：原实现把全部音频堆在内存里最后一次性写盘，
            # 拿掉 time.sleep(5) 后按真实速率累积（48kHz 立体声 16bit 约 691 MB/小时）。
            while self.is_running:
                data, overflowed = stream.read(self.buffer_size)
                if overflowed:
                    log.warning("%s", "[录音] 输入缓冲溢出，音频可能有丢失")
                raw = bytes(data)
                wf.writeframes(raw)
                if self.audio_queue is not None:
                    # 实时转写走这条：转成单声道 float32，重采样交给下游按块做
                    self.audio_queue.put(self._toMonoFloat32(raw))
        finally:
            # 异常或取消时也要把已录到的部分落盘并释放设备
            wf.close()
            stream.stop()
            stream.close()

    def stop(self):
        self.is_running = False

class OutputWorker(GuardedWorker):
    signal_write_over = Signal()

    def __init__(self, 
                    segments_path_info:list, 
                    output_dir:str, 
                    format:str, 
                    output_code = "UTF-8",
                    aggregate_contents = False,
                    parent=None
                ) -> None:
        
        super().__init__(parent)
        self.is_running = False
        self.segments_path_info = segments_path_info
        self.format = format
        self.output_dir = output_dir
        self.output_code = output_code
        self.aggregate_contents = aggregate_contents

    def stop(self):
        self.is_running = False
        # self.signal_process_over.emit()

    def run(self):
        self.is_running = True
        output_format = self.format
        output_dir = self.output_dir

        # 检查输出目录
        if output_dir != "" and not os.path.exists(output_dir):
            log.info("%s", f"\nCreate output dir : {output_dir}")
            # 给定的输出目录不存在时 创建输出目录
            os.makedirs(output_dir)
        
        if self.segments_path_info is None:
            return
        
        # 后续处理
        for segments, path, info in self.segments_path_info:

            if self.output_dir == "":
                output_dir,_ = os.path.split(path)

            log.info("%s", "Output...")
            # 输出到字幕文件
            if output_format.lower() == "all":
                output_format_ = SUBTITLE_FORMAT
            else:
                output_format_ = [output_format]
            for format in output_format_:
                file_out = getSaveFileName( path
                                            , format=format
                                            , rootDir=output_dir
                                        )
                writeSubtitles(outputFileName=file_out
                            , segments=segments
                            , format=format
                            , language=info.language
                            , fileName=path
                            , file_code = self.output_code
                            , aggregate_contents = self.aggregate_contents
                        )

        log.info("%s", "\n【Over】")
        self.signal_write_over.emit()
        self.stop()

class TranscribeWorker(GuardedWorker):
    signal_process_over = Signal(list)

    def __init__(self
                ,parent=None
                ,model : WhisperModel = None
                ,parameters : WhisperParameters = None
                ,vad_filter : bool = False
                ,vad_parameters : dict = None
                ,num_workers : int = 1
            ) -> None:
        
        super().__init__(parent)

        self.is_running = False
        self.model = model
        self.parameters = parameters
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters
        self.num_workers = num_workers

        self.segments_path_info = []
        
        
    def transcribe_file(self, file) -> (TranscriptionInfo, List): # type: ignore
        # try:
        #     is_av_file = self.try_decode_avFile(file)
        #     if not is_av_file:
        #         return (None,None)
        # except Exception as e: # 捕获异常
        #     print(f'    {file.split("/")[-1]} 不是一个有效的音视频文件\n')
        #     print(f"    error:{str(e)}")
        #     print(f"    ignore File : {file} \n")
        #     return (None, None)

        log.info("%s", "开始处理音频...")
        segments, info = self.model.transcribe(
                                                audio=file,
                                                **buildTranscribeKwargs(self.parameters,
                                                                        self.vad_filter,
                                                                        self.vad_parameters)
                                            )
        
        try:
            self.detect_Audio_info(info)
        except Exception as e:
            log.error("%s", f"{file} 处理失败!")
            log.error("%s", str(e))
            return (None, None)

        # segments = list(segments)
        segmentsTranscribe : List[segment_Transcribe] = []
        # 遍历生成器，并获取转写内容
        log.info("%s", f"Transcription for {file.split('/')[-1]}")

        for segment in segments:
            # 退出进程标识
            # print(self.is_running)
            if not self.is_running:
                # self.signal_process_over.emit(self.segments_path_info)
                return info, None

            log.info("%s", f'  [{str(round(segment.start, 5))}s --> {str(round(segment.end, 5))}s] {segment.text.lstrip()}')
            segmentsTranscribe.append(segment_Transcribe(segment))#.start, segment.end, segment.text))

        # if not self.is_running:
        #     self.signal_process_over.emit()
        return info, segmentsTranscribe

    def detect_Audio_info(self, info):
        if info.language != "zh":
            language = Language_dict[info.language]
            if language:
                language = language.capitalize()
        else:
            language = "Chinese" 
        language_probability = info.language_probability
        duration = info.duration
        duration = secondsToHMS(duration).replace(",", ".")
        duration_after_vad = info.duration_after_vad
        duration_after_vad = secondsToHMS(duration_after_vad).replace(",", ".")
        log.info("%s", f"  Detected language [{language}] with probability [{language_probability*100:.2f}%]")
        log.info("%s", f"  Audio duration     —— [{duration}] ")
        log.info("%s", f"  after VAD duration —— [{duration_after_vad}]")


    def try_decode_avFile(self, file) -> bool:
        '''
        尝试解析输入文件，并获取文件类型
        '''

        flag = False

        log.info("%s", "\n")
        log.info("%s", f"current task: {file}")
        log.info("%s", "  尝试解析文件")
        container = av.open(file, metadata_errors="ignore") # 尝试打开文件      
        av_cont = container.streams
        for stream in av_cont:
            if stream.codec_context.type == "audio":
                flag = True
                break

        if not flag:
            log.error("%s", "  解析失败！目标文件不是有效的音视频文件")
            
        container.close()
        return flag
    
    def run(self) -> None:
        """
        QThread 入口。

        异常由 GuardedWorker 统一接住（critical + 完整 traceback + 落盘），
        这里不再自行 try/except，避免两套处理互相遮蔽。
        """
        self.runTranscribe()

    def onError(self, exc) -> None:
        """
        转写线程崩溃时的界面恢复。

        traceback 由 GuardedWorker 统一记录（critical 级别、完整堆栈、直接落盘），
        这里只负责把界面从「正在处理中」放出来，避免永久卡住。
        """
        try:
            # signal_process_over 声明为 Signal(list)，不能发 None
            self.signal_process_over.emit([])
        except Exception:                              # noqa: BLE001
            log.debug("崩溃后发送 signal_process_over 失败", exc_info=True)

    def runTranscribe(self) -> None:
        self.is_running = True

        # 检查临时目录
        if not(os.path.exists(r"./temp")):
            os.mkdir(r"./temp")

        # model = self.model 
        parameters = self.parameters
        # vad_filter = self.vad_filter 
        # vad_parameters = self.vad_parameters 
        num_workers = self.num_workers

        files = parameters["audio"]

        # 忽略掉输入文件中可能存在的所有的字幕文件
        files = [file for file in files if file.split(".")[-1].upper() not in SUBTITLE_FORMAT]
        if ingnore_files := [
            file
            for file in files
            if file.split(".")[-1].upper() in SUBTITLE_FORMAT
        ]:
            new_line = "\n              "
            log.info("%s", f"ignore files: {new_line.join(ingnore_files)}")

        self.segments_path_info = []
        # 在线程池中并发执行相关任务，默认状况下使用单 GPU 该并发线程数为 1 ，
        # 提高线程数并不能明显增大吞吐量， 且可能因为线程调度的原因反而造成转写时间变长
        # 多 GPU 或多核心 CPU 可通过输入设备号列表并增大并发线程数的方式增加吞吐量，实现多任务并发处理
        # 但会造成内存或显存占用增多
        with futures.ThreadPoolExecutor(num_workers) as executor:
            results = executor.map(self.transcribe_file, files)
            new_line = "\n"

            for path, results in zip(files, results):
                # print(self.is_running)
                if not self.is_running:
                    self.signal_process_over.emit(self.segments_path_info)
                    return
                
                (info, segments) = results

                if segments is None:
                    continue

                self.segments_path_info.append((segments, path, info))
                        # print(
                        #         f"\nTranscription for {path.split('/')[-1]}:\n{new_line.join('[' + str(segment.start) + 's --> ' + str(segment.end) + 's] ' + segment.text for segment in segments)}"
                        #     )
                
                # 保存临时文件
                temp_output_save_file = getSaveFileName(audioFile=path, format="SRT", rootDir=r"./temp")
                writeSubtitles(temp_output_save_file, segments=segments, format="SRT",language=info.language, fileName=path)
                log.info("%s", f"save temp file: {os.path.abspath(temp_output_save_file)}")
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log.info("%s", "\n【Over】")
        self.signal_process_over.emit(self.segments_path_info)

        return
        
    def stop(self):
        self.is_running = False
        # self.signal_process_over.emit()

# ---------------------------------------------------------------------------------------------------------------------------

def writeSubtitles(outputFileName:str, 
                    segments:List[segment_Transcribe], 
                    format:str, 
                    language:str="",
                    fileName = "",
                    file_code = "UTF-8",
                    aggregate_contents = False
                ):
    
    file_code = ENCODING_DICT[file_code]

    if format == "SRT":
        writeSRT(outputFileName, segments, file_code = file_code)
    elif format == "TXT":
        writeTXT(outputFileName, segments, file_code=file_code, aggregate_contents = aggregate_contents)
    elif format == "VTT":
        writeVTT(outputFileName, segments,language=language, file_code=file_code)
    elif format == "LRC":
        wirteLRC(outputFileName, segments,language=language, file_code=file_code)
    elif format == "SMI":
        writeSMI(outputFileName, segments, language=language, avFile=fileName, file_code=file_code)
    elif format == "JSON":
        writeJson(outputFileName, segments, language=language, avFile=fileName, file_code=file_code)
    elif format == "ASS":
        writeASS(outputFileName, segments, file_code=file_code)

    log.info("%s", f"write over | {os.path.abspath(outputFileName)}")
    
def writeJson(fileName:str, segments:List[segment_Transcribe], language:str,avFile="", file_code="utf8"):

    _id = getMd5HashId(avFile, file_code=file_code)
    
    result = {
                "id": _id,
                "title": os.path.split(avFile)[-1],
                "format": "SubRip",
                "templates": {
                                "default": "__CONTENT__",
                                # 这里原本写作 "<i>__CONTENT__<\/i>"。
                                # \/ 在 Python 里是无效转义序列（反斜杠被保留），
                                # json.dump 又把它转义一次，于是写出的字幕文件里是
                                # "<\\/i>"，解析回来是 <\/i> —— 不是合法 HTML。
                                # 已去掉反斜杠，输出 <i>__CONTENT__</i>。
                                "italic": "<i>__CONTENT__</i>"
                            },
                "styles": {
                            "default": "font-style: 10px; line-height: 1; color: #FFF;"
                            }
                }
    
    result["data"] = []
    
    for segment in segments:
        start_time_HNS = secondsToHMS(segment.start)
        end_time_HMS = secondsToHMS(segment.end)

        result["data"].append(
                                {
                                "trigger": segment.start * 1000,
                                "lang": language,
                                "styles": [
                                    "default"
                                ],
                                "templates": [
                                    "default"
                                ],
                                "start": {
                                    "time": segment.start * 1000,
                                    "hour": int(start_time_HNS.split(":")[0]),
                                    "mins": int(start_time_HNS.split(":")[1]),
                                    "secs": int(start_time_HNS.split(":")[2].split(",")[0]),
                                    "ms": int(start_time_HNS.split(":")[2].split(",")[0])
                                },
                                "end": {
                                    "time": segment.end * 1000,
                                    "hour": int(end_time_HMS.split(":")[0]),
                                    "mins": int(end_time_HMS.split(":")[1]),
                                    "secs": int(end_time_HMS.split(":")[2].split(",")[0]),
                                    "ms": int(end_time_HMS.split(":")[2].split(",")[0])
                                },
                                "duration": {
                                    "secs": round(segment.end - segment.start, 3),
                                    "ms": round(segment.end - segment.start, 3) * 1000
                                },
                                "content": segment.text,
                                "meta": {
                                    "original": {
                                        "start": start_time_HNS,
                                        "end": end_time_HMS
                                    }
                                },
                                "words":[{"start":word.start,"end":word.end,"word":word.word,"probability":word.probability }for word in segment.words],
                                "speaker":segment.speaker or "",
                            }
                        )
    with open(os.path.abspath(fileName),'w',encoding=file_code) as fp:
    
        json.dump(
                    result,
                    fp,
                    ensure_ascii=False,
                    indent=4
                )
    

def writeSMI(fileName:str, segments:List[segment_Transcribe], language:str, avFile = "",file_code="utf8"):

    subtitle_color_list = ["white", "red", "blue", "green", "yellow", "cyan", "magenta"]

    # 获取音频或视频的名称
    _, fileName_ = os.path.split(fileName)
    baseName = fileName_.split('.')[0]

    if avFile:
        # 带有扩展名的文件名
        _, fileName_ = os.path.split(avFile)
    else:
        fileName_ = ""

    # 创建字幕的样式类
    language_type_CC = f"{language.upper()}CC"

    # 创建一个空的 smi 字幕字符串
    smi = ""
    # 添加 smi 字幕的头部标签
    smi += "<SAMI>\n"
    # 添加 smi 字幕的元数据和样式信息
    smi += "<HEAD>\n"
    # 标题
    smi += f"<TITLE>{baseName}</TITLE>\n"
    # 参数
    smi += "<SAMIParam>\n"
    smi += f"  Media {'{'}{fileName_}{'}'}\n"
    smi += "  Metrics {time:ms;}\n"
    smi += "  Spec {MSFT:1.0;}\n"
    smi += "</SAMIParam>\n"
    # 样式
    smi += "<STYLE TYPE=\"text/css\">\n"
    smi += "<!--\n"
    smi += "  P { font-family: Arial; font-weight: normal; color: white; background-color: black; text-align: center; }\n"
    speakers = ["SUB"]
    for segment in segments:
        try:
            speaker = segment.speaker
        except:
            speaker = "SUB"
        if not(speaker in speakers):
            speakers.append(speaker)
    if len(speakers) > 1:
        i = 0
        for speaker in speakers:
            smi += f"  #{speaker} {'{'} color: {subtitle_color_list[i]}; {'}'}\n"
            i += 1
    else:
        smi += "  #SUB{color: white; background-color: black; font-family: Arial; font-size: 12pt; font-weight: normal; text-align: left;}"
    if language != "zh":
        smi += f"  .{language_type_CC} {'{'} name: {Language_dict[language].capitalize()}; lang: {language}; SAMIType: CC; {'}'}\n"
    else:
        smi += f"  .{language_type_CC} {'{'} name: {'Chinese'}; lang: {language}; SAMIType: CC; {'}'}\n"
    smi += "-->\n"
    smi += "</STYLE>\n"
    smi += "</HEAD>\n"
    # 添加 smi 字幕的内容和时间信息
    smi += "<BODY>\n"
    # 遍历字幕列表，每个字幕是一个字典，包含 start, end, text, words 四个键
    for segment in segments:
        try:
            speaker = segment.speaker
            if not(speaker is None):
                speaker = segment.speaker
            else:
                speaker = "SUB"
        except:
            speaker = "SUB"
        
        # 添加字幕段的开始时间标签，格式为 <SYNC Start=毫秒数>
        smi += f"<SYNC Start={segment.start * 1000}>\n"
        # 添加字幕段的文本内容标签，格式为 <P Class=样式类名>文本内容
        # 如果有单词级时间戳，则在每个单词后面添加 <SPAN Class=样式类名>标签和时间戳
        if segment.words:
            if speaker != "SUB" and not(speaker is None):
                smi += f"  <P Class={language_type_CC}>{speaker}: "
            else:
                smi += f"  <P Class={language_type_CC}>"
            for word in segment.words:
                if not(language in Language_without_space):
                    word_text = word.word + " "
                else:
                    word_text = word.word

                # word_text = word.word
                try:
                    if word.end >= segment.start and word.end <= segment.end:
                        smi += f"<{secondsToHMS(word.start).replace(',','.')}><SPAN Class={language_type_CC}>{word_text}</SPAN>"
                    else:
                        smi += f"<SPAN Class={language_type_CC}>{word_text}</SPAN>"    
                    # smi += f"<SPAN Class={language_type_CC}>{word.word}</SPAN><{secondsToHMS(word.start).replace(',','.')}>"
                    # smi += f"{word.word}<SPAN Class={language_type_CC}>{secondsToHMS(word.start).replace(',','.')}</SPAN>"
                except:
                    smi += f"<SPAN Class={language_type_CC}>{word_text}</SPAN>"
            smi += "</P>\n"
        else:
            if speaker != "SUB" and not(speaker is None):
                smi += f"<P Class={language_type_CC}>{speaker}: {segment.text}</P>\n"
            else:
                smi += f"<P Class={language_type_CC}>{segment.text}</P>\n"

        # 添加字幕段的结束时间标签，格式为 <SYNC Start=毫秒数>
        smi += f"</SYNC>\n"
    # 添加 smi 字幕的尾部标签
    smi += "</BODY>\n"
    smi += "</SAMI>\n"

    # 使用 utf-8 重新编码字幕字符串
    smi:str = smi.encode("utf8").decode("utf8")

    # 将SMI字幕写入文件
    # f = codecs.open(fileName, "w",encoding=file_code)
    with codecs.open(fileName, "w", encoding=file_code) as f:
        f.write(smi)
    
    # return smi

def wirteLRC(fileName:str, segments:List[segment_Transcribe],language:str,file_code="utf8"):
    _, baseName = os.path.split(fileName)
    baseName = baseName.split(".")[0]
    with codecs.open(fileName, "w", encoding=file_code) as f:
        f.write(f"[ti:{baseName}]\n")
        f.write(f"[re:FasterWhisperGUI]\n")
        f.write(f"[offset:0]\n\n")

        for segment in segments:
            
            start:str = secondsToMS(segment.start)
            try:
                speaker = segment.speaker + ": "
            except:
                speaker = ""

            if segment.words:
                
                text = f"[{start[:8]}]" + speaker
                length = len(segment.words)
                for i in range(length):
                    word = segment.words[i]
                    if not(language in Language_without_space):
                        word_text = word.word + " "
                    else:
                        word_text = word.word
                    try:
                        if word.start >= segment.start and word.start <= segment.end:
                            text += f"<{secondsToMS(word.start)[:8]}>{word_text}"
                        else:
                            text += f"{word_text}"
                    except:
                        text += f"{word_text}"
                text += f"<{secondsToMS(segment.end)[:8]}>"
            else:
                text:str = f"[{start[:8]}]{speaker}{segment.text}"

            # 重编码为 utf-8 
            text:str = text.encode("utf8").decode("utf8")

            f.write(f"{text} \n")

def writeVTT(fileName:str, segments:List[segment_Transcribe],language:str,file_code="utf8"):
    # 创建一个空的 VTT 字幕对象
    _, baseName = os.path.split(fileName)
    vtt = webvtt.WebVTT()
    # 遍历字幕列表，每个字幕是一个字典，包含 start, end, text, words 四个键
    for segment in segments:
        # 创建一个空的字幕段对象
        cue = webvtt.Caption()
        # 设置字幕段的开始时间和结束时间，格式为 HH:MM:SS.mmm
        cue.start = secondsToHMS(segment.start).replace(",",".")
        cue.end = secondsToHMS(segment.end).replace(",", ".")
        # 设置字幕段的文本内容，如果有单词级时间戳，则输出时间戳和单词
        text = ""
        try:
            speaker = segment.speaker + ": "
        except:
            speaker = ""

        if segment.words:
            text = speaker + text
            for i in range(len(segment.words)):
                word = segment.words[i]
                if not(language in Language_without_space):
                    word_text = word.word + " "
                else:
                    word_text = word.word

                # if i == 0:
                if i == len(segment.words) - 1:
                    text += f"{word_text}"
                else:
                    try:
                        if word.end >= segment.start and word.end <= segment.end:
                            text += f"{word_text}<{secondsToHMS(word.end).replace(',','.')}>"
                        else:
                            text += f"{word_text}"
                        # text += f"<{secondsToHMS(word.start).replace(',','.')}>{word.word}"
                    except:
                        text += f"{word_text}"
        else:
            text = speaker + segment.text
        text:str = text.encode("utf8").decode("utf8")
        cue.text = text
        # 将字幕段添加到 VTT 字幕对象中
        vtt.captions.append(cue)
        
    vtt.save(fileName, file_code)

def writeTXT(fileName:str, segments, file_code="utf8", aggregate_contents=False):
    with codecs.open(fileName, "w", encoding=file_code) as f:
        speaker_temp = ""

        for segment in segments:
            
            text:str = segment.text
            try:
                speaker = segment.speaker + ": "
            except:
                speaker = ""
            
            if speaker_temp != speaker and aggregate_contents:
                f.write(f"\n{speaker.encode('utf8').decode('utf8')} \n")
                speaker_temp = speaker
            elif not aggregate_contents:
                text = speaker + text

            # 重编码为 utf-8 
            text:str = text.encode("utf8").decode("utf8")
            f.write(f"{text} \n")

            # text = speaker + text

            

def writeSRT(fileName:str, segments, file_code="UTF-8"):
    index = 1
    # encoding = ENCODING_DICT[file_code]
    with codecs.open(fileName, "w", encoding=file_code) as f:
        for segment in segments:
            start_time:float = segment.start
            end_time:float = segment.end
            text:str = segment.text

            try:
                speaker = segment.speaker + ": "
            except:
                speaker = ""

            text = speaker + text

            # 重编码为 utf-8 
            text:str = text.encode("utf8").decode("utf8")

            start_time:str = secondsToHMS(start_time)
            end_time:str = secondsToHMS(end_time)
            f.write(f"{index}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n")
            
            index += 1
    
def writeASS(fileName:str, segments, file_code="UTF-8"):
    with codecs.open(fileName, "w", encoding=file_code) as f:
        f.write("[Script Info]\n")
        f.write("; This file was generated by FasterWhisperGUI\n")
        f.write("; https://github.com/CheshireCC/faster-whisper-GUI\n")
        f.write("Original Script: FasterWhisperGUI\n")
        f.write("ScriptType: v4.00+\n")
        f.write("Collisions: Normal\n")
        f.write("PlayDepth: 0\n")
        f.write("Timer: 100.0000\n")
        f.write("WrapStyle: 0\n")

        f.write("\n[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: fwgDefault, Microsoft YaHei, 16, &H00FFFFFF, &H00FFFFFF, &H00000000, &H00000000, 0, 0, 0, 0, 100, 100, 0.00, 0.00, 1, 1, 0, 2, 20, 20, 20, 1\n")
        
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n")
        for segment in segments:
            # segment_Transcribe 始终带 speaker 字段，但 writeSubtitles 也会处理从
            # 外部读入（SRT/JSON）或由 whisperx 转换来的结果；用 getattr 兜底，
            # 避免因为缺一个字段在整个导出环节抛 AttributeError。
            speaker = getattr(segment, "speaker", None) or ""
            f.write(f'Dialogue: 0,{secondsToHMS(segment.start).replace(",",".")[:-1]},{secondsToHMS(segment.end).replace(",",".")[:-1]},fwgDefault,{speaker},0000,0000,0000,,{segment.text}\n')


def getSaveFileName(audioFile: str, format:str = "srt", rootDir:str = ""):
    path, fileName = os.path.split(audioFile)
    fileName = fileName.split(".")

    fileName[-1] = format.lower()

    fileName = ".".join(fileName)

    if rootDir != "":
        path = rootDir

    saveFileName = os.path.join(path, fileName).replace("\\", "/")
    return saveFileName

def getMd5HashId(fileName:str, file_code) -> str:
    
    md5 = hashlib.md5()
    md5.update(fileName.encode(file_code))
    id = md5.hexdigest()
    return id

