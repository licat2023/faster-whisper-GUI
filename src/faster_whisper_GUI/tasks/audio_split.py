# coding:utf-8

import logging
import os
import subprocess
import re
import csv
import hashlib
import threading

from PySide6.QtCore import Signal

from faster_whisper_GUI.runtime.timecode import secondsToHMS
from faster_whisper_GUI.tasks.base import GuardedWorker

log = logging.getLogger(__name__)

class SplitAudioFileWithSpeakersWorker(GuardedWorker):
    # 定义一个信号，用于在处理完成后发送结果
    result_signal = Signal(str)
    current_task_signal = Signal(str)

    def __init__(self, segments_path_info_list:list, output_path, language="", parent=None):
        super().__init__(parent)
        self.segments_path_info_list = segments_path_info_list
        self.output_path = output_path
        self.language = language
        self.is_running = False
        self.cancelled = False
        self._stop_requested = threading.Event()

        # 检查输出目录
        if output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    def creatCommandLine(self, start_time, end_time, fileName, output_path, speaker):

        output_fileName = self.getOutPutFileName(output_path, start_time, end_time, speaker)
        return [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            fileName,
            "-ss",
            start_time,
            "-to",
            end_time,
            output_fileName,
        ]
    
    def getOutPutFileName(self, output_path:str, start_time:str, end_time:str, speaker:str):
        speaker = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(speaker or "UnKnownSpeaker")).strip(". ") or "UnKnownSpeaker"
        fileName = ""
        if not(speaker is None) and speaker != "":
            fileName = os.path.join(output_path, f"{speaker}_{start_time.replace(':','_')}_{end_time.replace(':','_')}.wav")
        else:
            fileName = os.path.join(output_path, f"UnKnownSpeaker{start_time.replace(':','_')}_{end_time.replace(':','_')}.wav")
        return fileName.replace('\\','/')
    

    def run(self):
        if self._stop_requested.is_set():
            return
        self.is_running = True

        for result in self.segments_path_info_list:
            if not self.is_running or self._stop_requested.is_set():
                break
            segments, path, _info = result
            base_path, file = os.path.split(path)
            log.info("%s", f"    current task: {file}")

            self.current_task_signal.emit(file)

            if not self.output_path:
                output_path = base_path
            else:
                output_path = self.output_path
            basename = os.path.splitext(file)[0]
            if self.output_path:
                suffix = hashlib.sha256(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()[:8]
                basename += '-' + suffix
            output_path = os.path.join(output_path, basename)
            output_path = output_path.replace("\\","/")

            # print(output_path)
            # 检查输出路径
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # 每个输入文件有自己的标注文件；即使列表为空也能安全关闭。
            list_path = os.path.join(output_path, "00_list.csv")
            with open(list_path, "w", encoding="utf8", newline="") as list_file:
                # 格式：vocal_path,speaker_name,language,text
                writer = csv.writer(list_file)
                writer.writerow(["vocal_path", "speaker_name", "language", "text"])

                for segment in segments:
                    if not self.is_running or self._stop_requested.is_set():
                        break
                    start_time = secondsToHMS(segment.start).replace(',', '.')
                    end_time = secondsToHMS(segment.end).replace(',', '.')
                    speaker = segment.speaker

                    if speaker is None or speaker == "":
                        speaker = "UnKnownSpeaker"

                    commandLine = self.creatCommandLine(
                        start_time, end_time, path, output_path, speaker
                    )
                    completed = subprocess.run(
                        commandLine,
                        shell=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        encoding="utf-8",
                        errors="replace",
                        text=True,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        check=False,
                    )
                    if completed.returncode != 0:
                        details = (completed.stderr or "").strip()
                        log.error(
                            "ffmpeg 分割失败（退出码 %s）：%s",
                            completed.returncode,
                            details or "未返回错误信息",
                        )
                        raise subprocess.CalledProcessError(
                            completed.returncode,
                            commandLine,
                            stderr=completed.stderr,
                        )

                    # 获取并整理文件名
                    output_fileName = self.getOutPutFileName(
                        output_path, start_time, end_time, speaker
                    )
                    output_fileName = output_fileName.replace('\\', '/')

                    # 输出标注信息
                    def safe_cell(value):
                        value = str(value)
                        return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value
                    writer.writerow([output_fileName, safe_cell(speaker), safe_cell(self.language), safe_cell(segment.text.strip())])

        # 完成后发送结果信号
        result = "over"
        if not self.cancelled:
            self.result_signal.emit(result)
        self.is_running = False

    def onError(self, exc):
        self.stop()

    def stop(self):
        self.cancelled = True
        self._stop_requested.set()
        self.is_running = False

