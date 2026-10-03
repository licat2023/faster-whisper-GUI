"""人声分离的界面编排。"""

import logging
import os
from qfluentwidgets import MessageBox, FluentIcon
import sys
from faster_whisper_GUI.tasks.roformer import RoformerWorker
from faster_whisper_GUI.runtime.paths import MODEL_CACHE_DIR

log = logging.getLogger(__name__)

class SeparationActions:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def reSetButton_demucs_process(self):
        self.page_demucs.process_button.setText(self._tr("提取"))
        self.page_demucs.process_button.setIcon( FluentIcon.IOT)
        self.page_demucs.process_button.setEnabled(True)

    def demucs_file_process_status(self, status:dict):
        labels = {'reasmple audio': '音频重采样', 'separate sources': '音轨分离',
                  'save files': '保存音频文件', 'load model': '加载模型...',
                  'file over': '结束', 'download model': '下载模型...'}
        task = status.get('task', '')
        content = self._tr(labels.get(task, str(task)))
        fileName = os.path.basename(status.get('file', ''))
        self.setStateTool(text=f"{fileName} {content}", status=bool(status.get('status', False)))

    def demucs_process_over(self, status:bool):
        if status:
            self.setStateTool(text=self._tr("分离完成"), status=True)
            self.raiseSuccessInfoBar(self._tr("人声分离"), self._tr("音轨分离成功"))

        else:
            self.setStateTool(text=self._tr("结束"), status=True)
            error = getattr(self.demucsWorker, 'lastError', None)
            self.raiseErrorInfoBar(self._tr("人声分离"), str(error) if error else self._tr("任务已取消或分离失败"))

        self.reSetButton_demucs_process()
        self.page_demucs.backend_combox.setEnabled(True)

        # print(f"+++over, model:{self.demucsWorker.model}")
        # del self.demucsWorker.model
        del self.demucsWorker
        self.demucsWorker = None
        
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def demucsProcess(self):
        
        if self.demucsWorker is not None and self.demucsWorker.isRunning():
            
            message_w = MessageBox(self._tr("取消"), self._tr('确定取消？'),self)

            if not message_w.exec():
                return

            self.outputWithDateTime("Cancel Demucs")
            self.page_demucs.process_button.setEnabled(False)
            # 取消只走 stop()，理由见 closeEvent 里的说明
            self.demucsWorker.stop()

            self.setStateTool('取消', '等待分离任务结束', False)
            return
        
        self.outputWithDateTime("Demucs")
    
        try:
            param = self.getDemucsParams()
        except ValueError as error:
            self.raiseErrorInfoBar('参数错误', str(error))
            return
        if not param['output_path']:
            self.raiseErrorInfoBar('输出目录', '请先选择分离结果的保存目录。')
            return
        if not param['audio'] or not all(param['audio']):
            self.raiseErrorInfoBar('文件错误', '没有选择有效的音视频文件')
            return
        self._separation_status = False
        self.page_demucs.backend_combox.setEnabled(False)
        if self.demucsWorker is not None:
            self.demucsWorker.deleteLater()
            self.demucsWorker = None
        if self.page_demucs.backend_combox.currentData() == 'roformer':
            self.demucsWorker = RoformerWorker(param, self)
            self.demucsWorker.file_process_status.connect(self.demucs_file_process_status)
            self.demucsWorker.signal_vr_over.connect(self._separationResult)
            self.demucsWorker.finished.connect(self._separationFinished)
            self.setStateTool('Roformer', '加载模型并分离音轨', False)
            self.demucsWorker.start()
            self.page_demucs.process_button.setText('取消')
            return

        if len(param["audio"]) < 1 or(len(param["audio"]) == 1 and param["audio"][0] == ""):
            
            self.raiseErrorInfoBar(self._tr("文件错误"), self._tr("没有选择有效的音视频文件"))
            return

        for key,value in param.items():
            log.info("%s", f"{key}: {value}")

        if self.demucsWorker is None:
            from faster_whisper_GUI.tasks.separation import DemucsWorker
            self.demucsWorker = DemucsWorker(
                                            self,
                                            param["audio"],
                                            param["stems"],
                                            str(MODEL_CACHE_DIR / "hdemucs_high_trained.pt"),
                                            segment=param["segment"],
                                            overlap=param["overlap"],
                                            output_path=param["output_path"]
                                        )
        
        else:
            self.demucsWorker.audio = param["audio"]
            self.demucsWorker.stems = param["stems"]
            self.demucsWorker.segment = param["segment"]
            self.demucsWorker.overlap = param["overlap"]
            self.demucsWorker.output_path = param["output_path"]
        
        self.demucsWorker.signal_vr_over.connect(self._separationResult)
        self.demucsWorker.finished.connect(self._separationFinished)
        self.demucsWorker.file_process_status.connect(self.demucs_file_process_status)

        self.setStateTool(self._tr("Demucs"), self._tr("音轨分离"), False)
        self.demucsWorker.start()

        self.page_demucs.process_button.setText(self._tr("取消"))
        self.page_demucs.process_button.setIcon(":/resource/Image/Cancel_red")

    def _separationResult(self, status):
        self._separation_status = status

    def _separationFinished(self):
        if self.sender() is not None and self.sender() is not self.demucsWorker:
            return
        self.demucs_process_over(getattr(self, '_separation_status', False))
