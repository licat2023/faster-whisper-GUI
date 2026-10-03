"""模型加载、转换和释放。"""

import logging
import os
from threading import Thread
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from qfluentwidgets import InfoBar, InfoBarPosition, InfoBarIcon, MessageBox
import sys
from faster_whisper_GUI.config import STR_BOOL
from faster_whisper_GUI.tasks.backend_load import BackendLoadWorker

log = logging.getLogger(__name__)

class ModelActions:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def getDownloadCacheDir(self):
        """
        get path of local model dir
        """
        if path := QFileDialog.getExistingDirectory(
                                                    self
                                                    , self._tr("选择缓存文件夹")
                                                    , self.page_model.LineEdit_download_root.text()
                                                ):
            
            self.page_model.LineEdit_download_root.setText(path)
            self.download_cache_path = path

    def onModelLoadClicked(self):
        if self.inferenceBusy():
            self.raiseErrorInfoBar('模型正在使用', '请等待当前任务结束后再加载模型。')
            return
        try:
            model_param = self.getParam_model()
        except (ValueError, KeyError) as error:
            self.raiseErrorInfoBar('参数错误', str(error))
            return
        identity = self.page_model.backend_combox.currentData()
        if identity == 'faster-whisper' and not model_param['model_size_or_path']:
            self.raiseErrorInfoBar(title=self._tr('模型名称错误'), content=self._tr('需要模型所在目录或者有效的模型名称。'))
            return
        self._loading_backend_id = identity
        if hasattr(self.FasterWhisperModel, 'close'):
            self.FasterWhisperModel.close()
        self.FasterWhisperModel = None
        if self.loadModelWorker is not None:
            self.loadModelWorker.model = None
            self.loadModelWorker.deleteLater()
            self.loadModelWorker = None
        self.outputWithDateTime("LoadModel")
        if identity != 'faster-whisper':
            self.loadModelWorker = BackendLoadWorker(model_param, parent=self)
            self.loadModelWorker.setStatusSignal.connect(self.loadModelResult)
            self.loadModelWorker.setStatusSignal.connect(self.setModelStatusLabelTextForAll)
            self._lockModelLoading()
            self.setStateTool('加载模型', identity + ' 加载中', False)
            self.loadModelWorker.start()
            return
        
        for key, value in model_param.items():
            log.info("%s", f"    -{key}: {value}")

        model_size_or_path = model_param["model_size_or_path"]

        if model_size_or_path == "":
            self.raiseErrorInfoBar(
                            title=self._tr('模型名称错误'),
                            content=self._tr("需要模型所在目录或者有效的模型名称。"),
                        )
            return

        if os.path.isdir(model_size_or_path):
            content = self._tr("加载本地模型")
        else:
            content = self._tr("在线下载模型")

        infoBar = InfoBar(
                            icon=InfoBarIcon.INFORMATION,
                            title='',
                            content=content,
                            isClosable=False,
                            orient=Qt.Orientation.Vertical,    # vertical layout
                            position=InfoBarPosition.TOP,
                            duration=2000,
                            parent=self
                        )
                
        infoBar.show()

        param_for_model_load = {
                                "model_size_or_path":model_param["model_size_or_path"],
                                "device":model_param["device"],
                                "device_index":model_param["device_index"],
                                "compute_type":model_param["compute_type"],
                                "cpu_threads":model_param["cpu_threads"],
                                "num_workers":model_param["num_workers"],
                                "download_root":model_param["download_root"],
                                "local_files_only":model_param["local_files_only"]
                            }
        
        param_for_model_load.update(backend="faster-whisper", interpreter=sys.executable,
                                    use_v3_model=model_param["use_v3_model"])
        self.loadModelWorker = BackendLoadWorker(param_for_model_load, parent=self)
        self.loadModelWorker.setStatusSignal.connect(self.loadModelResult)
        self.loadModelWorker.setStatusSignal.connect(self.setModelStatusLabelTextForAll)
        self._lockModelLoading()
        self.setStateTool(self._tr("加载模型"), self._tr("模型加载中，请稍候"), False)
        self.loadModelWorker.start()

    def onButtonConvertModelClicked(self):

        if not self.page_model.model_online_RadioButton.isChecked():
            # QMessageBox.warning(self, "错误", "必须选择在线模型时才能使用本功能", QMessageBox.Yes, QMessageBox.Yes)
            log.info("%s", self._tr("Model Convert only Work In Onlie-Mode"))
            self.raiseErrorInfoBar(
                                    self._tr("错误")
                                    , self._tr("转换功能仅在在线模式下工作")
                                )
            return

        model_name_or_path = self.page_model.combox_online_model.currentText()
        model_output_dir = self.page_model.LineEdit_model_out_dir.text().strip()
        download_cache_dir = self.page_model.LineEdit_download_root.text().strip()
        quantization = self.page_model.preciese_combox.currentText()
        use_local_files = self.page_model.combox_local_files_only.currentText()
        use_local_files = STR_BOOL[use_local_files]

        log.info("%s", self._tr("Convert Model: "))
        log.info("%s", f"  model_name_or_path : {model_name_or_path}")
        log.info("%s", f"  model_output_dir   : {model_output_dir}")
        log.info("%s", f"  download_cache_dir : {download_cache_dir}")
        log.info("%s", f"  quantization       : {quantization}")
        log.info("%s", f"  use_local_files    : {use_local_files}")

        if model_output_dir == "":
            self.raiseErrorInfoBar("转换模型", "请先选择模型输出目录")
            return
    
        from faster_whisper_GUI.tasks.conversion import ConvertModel
        thread_go = Thread(target=ConvertModel, daemon=True, args=[model_name_or_path, download_cache_dir,model_output_dir, quantization, use_local_files])
        thread_go.start()

    def loadModelResult(self, state:bool):
        if self.loadModelWorker is None:
            return
        if state:
            self.setStateTool(text=self._tr("加载完成"),status=state)
            self.raiseSuccessInfoBar(
                                        title=self._tr('加载结束'),
                                        content=self._tr("模型加载成功")
                                    )
            self.FasterWhisperModel = self.loadModelWorker.model
            self.loaded_backend_id = self._loading_backend_id
            
        elif not state:
            self.setStateTool(text=self._tr("结束"), status=True)
            self.raiseErrorInfoBar(
                                    title=self._tr("错误"),
                                    content=str(self.loadModelWorker.lastError or self._tr('加载失败，请检查设置页的日志文件。'))
                                )

    def setModelStatusLabelTextForAll(self, status:bool):
        
        for page in self.pages:
            if not hasattr(page, 'setModelStatusLabelText'):
                continue
            try:
                page.setModelStatusLabelText(status)
            except Exception as e:
                # 原先这里是裸 pass，与 docs/LOGGING.md「绝不静默吞异常」相冲突：
                # 某个页面刷新状态失败时，界面没提示、日志里也没有记录。
                log.warning("页面 %s 刷新模型状态标签失败: %s",
                            page.objectName(), e, exc_info=True)

    def getLocalModelPath(self):
        """
        get path of local model dir
        """

        if self.page_model.backend_combox.currentData() != 'faster-whisper':
            return
        path = QFileDialog.getExistingDirectory(self, self._tr("选择模型文件所在的文件夹"), self.modelRootDir)

        if path:
            self.page_model.lineEdit_model_path.setText(path)
            self.model_path = path
            self.modelRootDir = os.path.abspath(os.path.join(path, os.pardir))

    def unloadWhisperModel(self):
        """
        从内存中卸载模型
        """
        # 转写正在进行时将会直接退出
        if self.FasterWhisperModel is None:
            self.raiseErrorInfoBar(self._tr("卸载模型失败"), self._tr("未加载模型"))
            return

        self.outputWithDateTime("Unload Whisper Model")

        if self.inferenceBusy():
            # self.transcribe_thread.terminate()
            self.raiseErrorInfoBar(self._tr("模型正在使用"), self._tr("语音识别正在运行"))
            return
        
        try:
            if hasattr(self.FasterWhisperModel, 'close'):
                self.FasterWhisperModel.close()
            self.FasterWhisperModel = None
            if self.loadModelWorker is not None:
                self.loadModelWorker.model = None
                self.loadModelWorker.deleteLater()
                self.loadModelWorker = None

            self.setModelStatusLabelTextForAll(False)
            self.raiseSuccessInfoBar(self._tr("卸载模型成功"), self._tr("卸载模型成功"))
            log.info("%s", "unload model succeed")

        except Exception as e:
            log.error("%s", "unload model failed")
            log.error("%s", str(e))
            self.raiseErrorInfoBar(self._tr("卸载模型失败"), str(e))


    def _lockModelLoading(self):
        self.page_model.backend_combox.setEnabled(False)
        self.page_model.button_model_lodar.setEnabled(False)
        self.loadModelWorker.finished.connect(self._modelLoadingFinished)

    def _modelLoadingFinished(self):
        self.page_model.backend_combox.setEnabled(True)
        self.page_model.button_model_lodar.setEnabled(True)

    def inferenceBusy(self):
        return any(worker is not None and worker.isRunning() for worker in
            [getattr(self, 'transcribe_thread', None), getattr(self, 'audio_stream_worker', None),
             getattr(self, 'audio_capture_thread', None),
             getattr(self, 'loadModelWorker', None)])
