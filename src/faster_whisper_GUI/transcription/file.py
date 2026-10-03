"""文件转录、并发调度与取消。"""



import logging
from concurrent import futures
import os
from typing import List
import av
from faster_whisper_GUI.backends.base import TranscriptionInfo
from PySide6.QtCore import Signal
from faster_whisper_GUI.tasks.base import GuardedWorker
from faster_whisper_GUI.backends.faster_whisper import FasterWhisperBackend
from faster_whisper_GUI.config import Language_dict, SUBTITLE_FORMAT
from faster_whisper_GUI.domain.segments import segment_Transcribe
from faster_whisper_GUI.runtime.timecode import secondsToHMS
from faster_whisper_GUI.domain.parameters import WhisperParameters

from faster_whisper_GUI.transcription.parameters import buildTranscribeKwargs
from faster_whisper_GUI.backends.catalog import recognition_options
from faster_whisper_GUI.subtitles.writers import getSaveFileName, writeSubtitles

log = logging.getLogger(__name__)

class TranscribeWorker(GuardedWorker):
    signal_process_over = Signal(list)

    def __init__(self
                ,parent=None
                ,model = None
                ,parameters : WhisperParameters = None
                ,vad_filter : bool = False
                ,vad_parameters : dict = None
                ,num_workers : int = 1
            ) -> None:
        
        super().__init__(parent)

        self.is_running = False
        self.model = model
        self.backend = model if hasattr(model, "recognize") else FasterWhisperBackend(model)
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

        if not self.is_running:
            return None, None
        log.info("%s", "开始处理音频...")
        recognition = self.backend.recognize(
            file, recognition_options(self.backend, self.parameters, self.vad_filter, self.vad_parameters),
        )
        if not recognition.segment_timestamps:
            raise ValueError("当前后端只提供全文；字幕任务需要先运行时间戳对齐")
        segments, info = recognition.segments, recognition.info
        
        try:
            self.detect_Audio_info(info)
        except Exception as e:
            log.error("%s", f"{file} 处理失败!")
            log.error("%s", str(e))
            log.warning("音频元数据记录失败，继续识别", exc_info=True)

        # segments = list(segments)
        segmentsTranscribe : List[segment_Transcribe] = []
        # 遍历生成器，并获取转写内容
        log.info("%s", f"Transcription for {file.split('/')[-1]}")

        try:
            for segment in segments:
                # 退出进程标识
                # print(self.is_running)
                if not self.is_running:
                    # self.signal_process_over.emit(self.segments_path_info)
                    return info, None
    
                log.info("%s", f'  [{str(round(segment.start, 5))}s --> {str(round(segment.end, 5))}s] {segment.text.lstrip()}')
                segmentsTranscribe.append(segment_Transcribe(segment))#.start, segment.end, segment.text))
        finally:
            close = getattr(segments, 'close', None)
            if close:
                close()

        # if not self.is_running:
        #     self.signal_process_over.emit()
        return info, segmentsTranscribe

    def detect_Audio_info(self, info):
        if info.language != "zh":
            language = Language_dict.get(info.language, info.language)
            if language:
                language = language.capitalize()
        else:
            language = "Chinese" 
        language_probability = info.language_probability
        duration = info.duration
        duration = secondsToHMS(duration).replace(",", ".") if duration is not None else "unknown"
        duration_after_vad = info.duration_after_vad
        duration_after_vad = (secondsToHMS(duration_after_vad).replace(",", ".")
                              if duration_after_vad is not None else "unknown")
        probability = f"{language_probability*100:.2f}%" if language_probability is not None else "unknown"
        log.info("%s", f"  Detected language [{language}] with probability [{probability}]")
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
        try:
            self.runTranscribe()
        finally:
            self.is_running = False

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
        os.makedirs("./temp", exist_ok=True)

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
        # Submit only the active batch: cancellation never starts the remaining files.
        with futures.ThreadPoolExecutor(num_workers) as executor:
            for offset in range(0, len(files), num_workers):
                if not self.is_running:
                    break
                batch = files[offset:offset + num_workers]
                pending = [executor.submit(self.transcribe_file, path) for path in batch]
                for path, future in zip(batch, pending):
                    info, segments = future.result()
                    if not self.is_running:
                        for queued in pending:
                            queued.cancel()
                        break
                    if segments is None:
                        continue
                    self.segments_path_info.append((segments, path, info))
                    temp_output_save_file = getSaveFileName(audioFile=path, format='SRT', rootDir='./temp')
                    writeSubtitles(temp_output_save_file, segments=segments, format='SRT', language=info.language, fileName=path)
        log.info("%s", "\n【Over】")
        self.signal_process_over.emit(self.segments_path_info)

        return
        
    def stop(self):
        self.is_running = False
        if hasattr(self.backend, 'cancel'):
            self.backend.cancel()
