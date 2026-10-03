"""字幕导入导出及按说话人切分音频。"""



import logging
from copy import deepcopy
import os
import av
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import MessageBox
from faster_whisper_GUI.backends.base import TranscriptionInfo
from faster_whisper_GUI.tasks.export import OutputWorker
from faster_whisper_GUI.subtitles.readers import readSRTFileToSegments, readJSONFileToSegments
from faster_whisper_GUI.config import ENCODING_DICT
from faster_whisper_GUI.runtime.diagnostics import outputWithDateTime
from faster_whisper_GUI.tasks.audio_split import SplitAudioFileWithSpeakersWorker

log = logging.getLogger(__name__)

class FileActions:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def outputOver(self):
        if self.outputWorker is None or (self.sender() is not None and self.sender() is not self.outputWorker):
            return
        self.setStateTool(self._tr("保存文件"), self._tr("结束"), True)
        if self.outputWorker is not None and self.outputWorker.lastError is not None:
            self.raiseErrorInfoBar("保存失败", str(self.outputWorker.lastError))
            self.outputWorker = None
            return
        self.raiseSuccessInfoBar(
                                title=self._tr("保存完成")
                                , content=self._tr("字幕文件已保存")
                            )
        for path, model, revision in getattr(self, '_exported_models', []):
            if self.tableModel_list.get(path) is model:
                model.markSaved(revision)
        self.outputWorker = None

    def outputSubtitleFile(self):

        if self.outputWorker is not None:
            return
        if not self.current_result:
            self.raiseErrorInfoBar("保存字幕", "没有可保存的字幕结果")
            return
        outputWithDateTime("OutputSubtitleFiles")

        format = self.page_output.combox_output_format.currentText()
        output_dir = self.page_output.outputGroupWidget.LineEdit_output_dir.text()
        code_ = self.page_output.combox_output_code.currentText()

        aggregate_contents_according_to_the_speaker = self.page_transcribes.switchButton_aggregate_contents_according_to_the_speaker.isChecked()
        result_to_write = deepcopy(self.current_result)
        self._exported_models = [(path, model, model.revision) for path, model in self.tableModel_list.items()]

        self.outputWorker = OutputWorker(result_to_write, output_dir, format, code_, aggregate_contents_according_to_the_speaker , self)
        self.outputWorker.finished.connect(self.outputOver)    
        self.outputWorker.start()
        self.setStateTool(self._tr("保存文件"), self._tr("输出字幕文件"), False)

    def is_audio_or_video(self, file_path:str) -> bool:
        try:
            with av.open(file_path, metadata_errors="ignore") as container:
                return any(getattr(stream.codec_context, 'type', None) == 'audio' for stream in container.streams)
        except Exception:
            log.warning("file open error: %s", file_path, exc_info=True)
            return False

    def openExcitedFiles(self):
        
        self.outputWithDateTime("openExcitedFiles")

        # 必须手动选择正确的语言选项
        if self.page_transcribes.combox_language.currentText().lower() == "auto":
            messageBoxDia_ = MessageBox(self._tr("选择语言"), self._tr("必须选择正确的字幕语言"),self)
            messageBoxDia_.show()
            return
        
        file,_ = QFileDialog.getOpenFileName(self, self._tr("选择音频文件"), self.page_process.fileNameListView.avDataRootDir)
        if not file:
            return

        if not self.is_audio_or_video(file):
            message_W = MessageBox(
                                    self._tr("文件无效"),
                                    self._tr("不是音视频文件或文件无法找到音频流，请检查文件及文件格式"),
                                    self
                                )
            message_W.show()
            return

        log.info("%s", f"open audio file: {file}")

        dataDir,_ = os.path.split(file)
        self.page_process.fileNameListView.avDataRootDir = dataDir
        # filesList = os.listdir(dataDir)

        # json 格式字幕文件为首选
        file_subtitle_fileName = os.path.splitext(file)[0] + ".json"
        ext_ = "json"
        # 检测 json 格式字幕文件存在性
        if not os.path.exists(file_subtitle_fileName):
            # 没有 json 格式字幕文件的时候将会尝试获取 srt 格式字幕
            file_subtitle_fileName = os.path.splitext(file)[0] + ".srt"
            ext_ = "srt"
        
        # fileName_subtitle_without_Ext = '.'.join(os.path.split(file_subtitle_fileName)[-1].split('.')[:-1])

        # 当字幕文件目录所指向的文件存在时
        if os.path.exists(file_subtitle_fileName):
            log.info("%s", f"find existed srt file: {file_subtitle_fileName}")
            # 获取文件的后缀名
            # ext_ = file_subtitle_fileName.split(".")[-1]

        else:
            file_subtitle_fileName,ext_ = QFileDialog.getOpenFileName(
                                                                    self, 
                                                                    self._tr("选择字幕文件"), 
                                                                    "", 
                                                                    # self.page_process.fileNameListView.avDataRootDir, 
                                                                    "JSON file(*.json);;SRT file(*.srt)",
                                                                )
            # print(ext_)
            
            if file_subtitle_fileName and os.path.isfile(file_subtitle_fileName):
                log.info("%s", f"get subtitle file: {file_subtitle_fileName}")
            else:
                messageBoxDia_ = MessageBox(self._tr("没有字幕文件"),self._tr("必须要有有效的字幕文件"),self)
                messageBoxDia_.show()
                return
        
        code_ = self.page_output.combox_output_code.currentText()

        try:
            if ext_ in ["JSON file(*.json)" ,"json"]:
                segments = readJSONFileToSegments(file_subtitle_fileName, file_code=ENCODING_DICT[code_])
            else:
                segments = readSRTFileToSegments(file_subtitle_fileName, file_code=ENCODING_DICT[code_])
        except Exception as e:
            log.error("%s", "read subtitle file failed:")
            log.error("%s", f"    {str(e)}")
            self.raiseErrorInfoBar(self._tr("读取失败"), self._tr("读取字幕文件失败 \n检查日志文件可能会获取更多信息"))
            return
        # 输出字幕文件内容
        # for segment in segments:
        #     print(f"[{segment.start}s --> {segment.end}s] | {segment.speaker+':'+segment.text if segment.speaker else segment.text}")

        language = self.page_transcribes.combox_language.currentText().split("-")[0]
        
        info = TranscriptionInfo(
                                    language="zh" if language in ["zhs","zht"] else language,
                                    language_probability=1,
                                      duration=max((segment.end for segment in segments), default=0.0),
                                      duration_after_vad=None,
                                )

        if language in ["zhs", "zht"]:
            self.simplifiedAndTraditionalChineseConvert(segments, language)

        if file_subtitle_fileName and file:
            self.result_whisperx_aligment = None
            self.result_whisperx_speaker_diarize = None

            self.current_result = self.mergeNewResults([(segments, file, info)])
            # self.tableModel_list[file] = file_subtitle_fileName
            
            self.showResultInTable(self.current_result)

    def outputAudioPartWithSpeaker(self):
        """
        output audio part with speaker
        """
        if self.splitAudioFileWithSpeakerWorker is not None and self.splitAudioFileWithSpeakerWorker.isRunning():
            return
        if self.current_result is None or len(self.current_result) == 0:
            self.raiseErrorInfoBar(self._tr("转写结果为空"), self._tr("没有有效的转写结果"))       
            return
        
        outputWithDateTime("SegmentAudioFileWithSpeaker")

        language = self.page_transcribes.combox_language.currentText().split("-")[0]

        output_path = self.page_output.outputGroupWidget.LineEdit_output_dir.text()
        try:
            self.splitAudioFileWithSpeakerWorker = SplitAudioFileWithSpeakersWorker(deepcopy(self.current_result),output_path,language ,self)
        except (OSError, ValueError) as error:
            self.raiseErrorInfoBar(self._tr('分割音频失败'), str(error))
            return
        self.splitAudioFileWithSpeakerWorker.finished.connect(self.splitAudioFileWithSpeakerWorkerFinished)
        self.splitAudioFileWithSpeakerWorker.current_task_signal.connect(lambda file: self.setStateTool(self._tr("分割音频"), self._tr("处理文件：") + file, False))
        self.setPageOutButtonStatus(False)
        self.splitAudioFileWithSpeakerWorker.start()

        self.setStateTool(self._tr("分割音频"), self._tr("按说话人分割音频文件"), False)

    def splitAudioFileWithSpeakerWorkerFinished(self):
        worker = self.splitAudioFileWithSpeakerWorker
        if worker is None or (self.sender() is not None and self.sender() is not worker):
            return
        self.splitAudioFileWithSpeakerWorker = None
        self.setPageOutButtonStatus(True)
        worker.deleteLater()
        if getattr(self, '_closingRequested', False):
            return
        if worker.lastError is not None:
            self.setStateTool(self._tr("分割音频"), self._tr("分割失败"), True)
            self.raiseErrorInfoBar(self._tr("分割音频失败"), str(worker.lastError))
            return
        if worker.cancelled:
            self.setStateTool(self._tr("分割音频"), self._tr("已取消"), True)
            return
        self.setStateTool(self._tr("分割音频"), self._tr("按说话人分割音频文件完成"), True)
        self.raiseSuccessInfoBar(self._tr("分割音频完成"),self._tr("按说话人分割音频文件完成"))

