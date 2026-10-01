# coding:utf-8

import logging
import os
import subprocess

from PySide6.QtCore import Signal

from .transcribe import secondsToHMS
from .workers import GuardedWorker

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
        fileName = ""
        if not(speaker is None) and speaker != "":
            fileName = os.path.join(output_path, f"{speaker}_{start_time.replace(':','_')}_{end_time.replace(':','_')}.wav")
        else:
            fileName = os.path.join(output_path, f"UnKnownSpeaker{start_time.replace(':','_')}_{end_time.replace(':','_')}.wav")
        return fileName.replace('\\','/')
    

    def run(self):
        self.is_running = True

        for result in self.segments_path_info_list:
            segments, path, _info = result
            base_path, file = os.path.split(path)
            log.info("%s", f"    current task: {file}")

            self.current_task_signal.emit(file)

            if not self.output_path:
                output_path = base_path
            else:
                output_path = self.output_path
            output_path = os.path.join(output_path, ".".join(file.split('.')[:-1]))
            output_path = output_path.replace("\\","/")

            # print(output_path)
            # 检查输出路径
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # 每个输入文件有自己的标注文件；即使列表为空也能安全关闭。
            list_path = os.path.join(output_path, "00_list.csv")
            with open(list_path, "w", encoding="utf8") as list_file:
                # 格式：vocal_path,speaker_name,language,text
                list_file.write("vocal_path,    speaker_name,    language,    text\n")

                for segment in segments:
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
                    text = segment.text.strip().replace(',', ' ')
                    list_file.write(
                        f"{output_fileName},{speaker},{self.language},{text}\n"
                    )

        # 完成后发送结果信号
        result = "over"
        self.result_signal.emit(result)
        self.stop()

    def onError(self, exc):
        self.stop()

    def stop(self):
        self.is_running = False
        self.quit()
