"""File recognition and native streaming orchestration; all GUI callbacks are nonblocking."""
import datetime
import logging
import os
import queue
from qfluentwidgets import MessageBox
from faster_whisper_GUI.config import CAPTURE_PARA
from faster_whisper_GUI.tasks.capture import CaptureAudioWorker
from faster_whisper_GUI.transcription.file import TranscribeWorker
from faster_whisper_GUI.transcription.native_streaming import NativeStreamWorker
from faster_whisper_GUI.ui.icons import FasterWhisperGUIIcon

log = logging.getLogger(__name__)


class TranscriptionActions:
    def onButtonProcessClicked(self):
        if self.audio_stream_worker is not None or self.audio_capture_thread is not None:
            self.audioCaptureOver()
        elif self.page_process.audio_capture_RadioButton.isChecked():
            self.audioCaptureProcess()
        else:
            self.transcribeProcess()

    def _recognitionReady(self):
        identity = self.page_model.backend_combox.currentData()
        if self.FasterWhisperModel is None:
            self.raiseErrorInfoBar('模型未加载', '请先在模型页加载所选后端。')
            return False
        if getattr(self, 'loaded_backend_id', 'faster-whisper') != identity:
            self.raiseErrorInfoBar('模型已切换', '所选后端尚未加载，请重新加载模型。')
            return False
        return True

    def _lockRecognitionControls(self, locked):
        self._recognition_controls_locked = locked
        self.page_model.backend_combox.setEnabled(not locked)
        self.page_model.button_model_lodar.setEnabled(not locked)
        self.page_process.audio_capture_RadioButton.setEnabled(
            not locked and self.page_model.backend_combox.currentData() == 'sherpa')
        self.page_process.transceibe_Files_RadioButton.setEnabled(not locked)

    def audioCaptureProcess(self):
        if not self._recognitionReady():
            return
        if getattr(self.FasterWhisperModel, 'backend_id', '') != 'sherpa':
            self.raiseErrorInfoBar('原生流式识别', '录音识别请选择并加载 sherpa-onnx 后端。')
            return
        try:
            parameters = self.getParamTranscribe()
        except ValueError as error:
            self.raiseErrorInfoBar('参数错误', str(error))
            return
        audio_format = CAPTURE_PARA[self.page_process.combox_capture.currentIndex()]
        os.makedirs('./temp', exist_ok=True)
        self.audio_wav_path = os.path.abspath(os.path.join('./temp',
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.wav'))
        self.audio_queue = queue.Queue()
        self.audio_capture_thread = CaptureAudioWorker(rate=audio_format['rate'],
            channels=audio_format['channel'], dType=audio_format['dType'],
            audio_queue=self.audio_queue, wav_path=self.audio_wav_path)
        self.audio_stream_worker = NativeStreamWorker(self.FasterWhisperModel, self.audio_queue,
            audio_format['rate'], self.audio_wav_path, parameters, parent=self)
        self.audio_stream_worker.signal_update.connect(self.streamTextUpdated)
        self.audio_stream_worker.Signal_process_over.connect(self._storeStreamResult)
        self.audio_stream_worker.finished.connect(self._finishStream)
        self.audio_capture_thread.finished.connect(self._captureFinished)
        self._pending_stream_result = []
        self._stream_finishing = False
        self.page_process.processResultText.clear()
        self._lockRecognitionControls(True)
        self.page_process.button_process.setText('停止录音')
        self.audio_stream_worker.start()
        self.audio_capture_thread.start()
        self.setStateTool('原生流式识别', '录音中，文字会随识别结果修订', False)

    def audioCaptureOver(self):
        if self.audio_capture_thread is not None:
            self.audio_capture_thread.stop()
        self.page_process.button_process.setText('正在收尾')
        self.page_process.button_process.setEnabled(False)
        self.setStateTool('停止录音', '保存录音并处理剩余音频', False)

    def _captureFinished(self):
        sender = self.sender()
        if self.audio_capture_thread is None or (sender is not None and sender is not self.audio_capture_thread):
            return
        error = self.audio_capture_thread.lastError
        self.audio_capture_thread = None
        if error and not getattr(self, '_closingRequested', False):
            self.raiseErrorInfoBar('录音失败', str(error))
        if getattr(self, '_stream_finishing', False):
            self.streamTranscribeOver(self._pending_stream_result)
            self._stream_finishing = False

    def streamTextUpdated(self, update):
        # Replace the complete hypothesis: revised prefixes must never be appended twice.
        self.page_process.processResultText.setPlainText(update.text)
        self.setStateTool('原生流式识别',
            f'修订 {update.revision} · 音频 {update.audio_seconds:.1f} 秒' +
            (' · 最终结果' if update.final else ' · 识别中'), update.final)

    def streamSegmentsUpdated(self, segments):
        self.page_process.processResultText.setPlainText(''.join(s.text for s in segments))

    def _storeStreamResult(self, results):
        self._pending_stream_result = results

    def _finishStream(self):
        sender = self.sender()
        if self.audio_stream_worker is None or (sender is not None and sender is not self.audio_stream_worker):
            return
        error = self.audio_stream_worker.lastError
        self.audio_stream_worker = None
        if self.audio_capture_thread is not None:
            self.audio_capture_thread.stop()
            self._stream_finishing = True
        else:
            self.streamTranscribeOver(self._pending_stream_result)
        if error and not getattr(self, '_closingRequested', False):
            self.raiseErrorInfoBar('流式识别失败', str(error))

    def streamTranscribeOver(self, segments_path_info):
        self.page_process.transceibe_Files_RadioButton.setChecked(True)
        self.transcribeOver(segments_path_info)

    def transcribeProcess(self):
        if self.transcribe_thread is not None and self.transcribe_thread.isRunning():
            dialog = MessageBox('取消', '是否取消当前转写任务？', self)
            if dialog.exec():
                self.cancelTrancribe()
            return
        if not self._recognitionReady():
            return
        try:
            parameters = self.getParamTranscribe()
            vad = self.getVADparam() if self.page_model.backend_combox.currentData() == 'faster-whisper' else {'vad_filter': False}
            workers = int(self.page_model.LineEdit_num_workers.text()) if self.page_model.backend_combox.currentData() == 'faster-whisper' else 1
        except ValueError as error:
            self.raiseErrorInfoBar('参数错误', str(error))
            return
        if not parameters['audio'] or parameters['audio'] == ['']:
            self.raiseErrorInfoBar('没有输入', '请先添加音频或视频文件。')
            return
        if workers < 1:
            self.raiseErrorInfoBar('参数错误', '并发数必须大于 0')
            return
        self.transcribe_thread = TranscribeWorker(model=self.FasterWhisperModel, parameters=parameters,
            vad_filter=vad['vad_filter'], vad_parameters=vad.get('param', {}), num_workers=workers)
        self._pending_file_result = []
        self._transcribe_cancelled = False
        self.transcribe_thread.signal_process_over.connect(self._storeFileResult)
        self.transcribe_thread.finished.connect(self._finishFile)
        self.page_process.processResultText.clear()
        self.redirectOutput(self.setTextAndMoveCursorToProcessBrowser)
        self._lockRecognitionControls(True)
        self.page_process.button_process.setText('取消')
        self.transcribe_thread.start()
        self.setStateTool('音频处理', '识别与生成字幕', False)

    def _storeFileResult(self, results):
        self._pending_file_result = results

    def _finishFile(self):
        sender = self.sender()
        if self.transcribe_thread is None or (sender is not None and sender is not self.transcribe_thread):
            return
        error = self.transcribe_thread.lastError
        self.transcribe_thread = None
        if getattr(self.FasterWhisperModel, 'is_closed', False):
            self.FasterWhisperModel.close()
            self.FasterWhisperModel = None
            self.setModelStatusLabelTextForAll(False)
        self.transcribeOver(self._pending_file_result)
        if error and not self._transcribe_cancelled:
            self.raiseErrorInfoBar('转写失败', str(error))

    def cancelTrancribe(self):
        self._transcribe_cancelled = True
        self.transcribe_thread.stop()
        self.page_process.button_process.setEnabled(False)
        self.setStateTool('取消', '等待任务释放资源', False)

    def resetButton_process(self):
        self._lockRecognitionControls(False)
        self.page_process.button_process.setEnabled(True)
        self.page_process.button_process.setText('开始')
        self.page_process.button_process.setIcon(FasterWhisperGUIIcon.PROCESS)

    def transcribeOver(self, segments_path_info):
        self.resetButton_process()
        self.setStateTool(text='结束', status=True)
        if not segments_path_info:
            return
        for segments, path, info in segments_path_info:
            if info.language == 'zh':
                if self.page_model.backend_combox.currentData() == 'faster-whisper':
                    language = self.page_transcribes.combox_language.currentText().split('-')[0]
                    self.simplifiedAndTraditionalChineseConvert(segments, language)
        self.current_result = self.mergeNewResults(segments_path_info)
        self.result_faster_whisper = self.current_result
        self.showResultInTable(self.current_result)
        if self.page_setting.combox_autoGoToOutputPage.currentIndex() == 0:
            self.stackedWidget.setCurrentWidget(self.page_output)
        elif self.page_setting.combox_autoGoToOutputPage.currentIndex() == 2:
            if MessageBox('转写结束', '是否跳转到输出页面？', self).exec():
                self.stackedWidget.setCurrentWidget(self.page_output)
        self.raiseSuccessInfoBar('转写完成', '结果已加入字幕编辑页面')
