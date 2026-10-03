"""字幕导出任务。"""

import logging
import os
from PySide6.QtCore import Signal
from faster_whisper_GUI.tasks.base import GuardedWorker
from faster_whisper_GUI.config import SUBTITLE_FORMAT

from faster_whisper_GUI.subtitles.writers import getSaveFileName, writeSubtitles

log = logging.getLogger(__name__)

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
        try:
            self.is_running = True
            output_format = self.format
            output_dir = self.output_dir
    
            # 检查输出目录
            if output_dir != "" and not os.path.exists(output_dir):
                log.info("%s", f"\nCreate output dir : {output_dir}")
                # 给定的输出目录不存在时 创建输出目录
                os.makedirs(output_dir, exist_ok=True)
            
            if self.segments_path_info is None:
                return
            
            # 后续处理
            for segments, path, info in self.segments_path_info:
                if not self.is_running:
                    break
    
                if self.output_dir == "":
                    output_dir,_ = os.path.split(path)
    
                log.info("%s", "Output...")
                # 输出到字幕文件
                if output_format.lower() == "all":
                    output_format_ = SUBTITLE_FORMAT
                else:
                    output_format_ = [output_format]
                for format in output_format_:
                    if not self.is_running:
                        break
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
        finally:
            self.is_running = False
            self.signal_write_over.emit()
