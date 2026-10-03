"""主窗口初始化、信号连接及界面生命周期。"""

from PySide6.QtWidgets import QFileDialog
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import QCoreApplication, Signal
from faster_whisper_GUI import logging_setup
from faster_whisper_GUI.runtime.diagnostics import outputWithDateTime

from .view import UIMainWin
from .signals import statusToolsSignalStore
from .parameters import WindowParameters
from .model import ModelActions
from .transcription import TranscriptionActions
from .results import ResultActions
from .postprocessing import PostprocessingActions
from .files import FileActions
from .separation import SeparationActions
from .settings import SettingsActions
from .feedback import WindowFeedback


class MainWindows(
    WindowParameters,
    ModelActions,
    TranscriptionActions,
    ResultActions,
    PostprocessingActions,
    FileActions,
    SeparationActions,
    SettingsActions,
    WindowFeedback,
    UIMainWin,
):
    """组合窗口视图与职责分组，持有所有 Qt 信号及任务状态。"""

    signal_guiLog = Signal(str)

    def _tr(self, text):
        return QCoreApplication.translate(self.__class__.__name__, text)

    def __init__(self):

        # self.translator = translator

        # 输出重定向由 logging_setup 统一负责（见 FasterWhisperGUI.py 的日志引导）。
        # 这里只做幂等兜底：万一本类被单独使用（没有经过引导），也要能正常跑。
        logging_setup.setupLogging()
        logging_setup.installOutputStreams()

        self._guiHandler = None
        self._guiLogTarget = None

        super().__init__()

        self._guiLogTarget = self.setTextAndMoveCursorToProcessBrowser
        self.signal_guiLog.connect(self._guiLogTarget)

        self.outputWithDateTime = outputWithDateTime

        self.statusToolSignalStore = statusToolsSignalStore()

        self.transcribe_thread = None
        self.audio_capture_thread = None
        self.audio_stream_worker = None
        self.audio_queue = None
        self.audio_wav_path = ""
        self.whisperXWorker = None
        self.outputWorker = None
        self.splitAudioFileWithSpeakerWorker = None
        self.demucsWorker = None
        self.loadModelWorker = None

        self.stateTool = None
        
        self.tableModel_list = {}

        self.result_whisperx_aligment = None
        self.result_faster_whisper = None
        self.result_whisperx_speaker_diarize = None
        self.current_result = None

        self.modelRootDir = r"./"

        self.singleAndSlotProcess()
        self.page_model.backend_combox.currentIndexChanged.connect(self.backendSelectionChanged)
        self.backendSelectionChanged()

        if self.page_setting.switchButton_autoLoadModel.isChecked():
            self.onModelLoadClicked()
        
        self.textOfParentClass()

    def textOfParentClass(self) -> None:
        # to fixed bug of translator 
        self.text_home = self._tr("Home")
        self.text_AVE = self._tr("声乐分离")
        self.text_modelParam = self._tr("模型参数")
        self.text_VAD = self._tr("人声活动检测")
        self.text_fwParam = self._tr("转写参数")
        self.text_process = self._tr("执行转写")
        self.text_output = self._tr("whiperX及字幕编辑")
        self.text_setting = self._tr('设置')

    def backendSelectionChanged(self, *_):
        identity = self.page_model.backend_combox.currentData()
        self.page_transcribes.selectBackend(identity)
        self.page_VAD.setEnabled(identity == 'faster-whisper')
        locked = getattr(self, '_recognition_controls_locked', False) or self.inferenceBusy()
        self.page_process.audio_capture_RadioButton.setEnabled(identity == 'sherpa' and not locked)
        if identity != 'sherpa':
            self.page_process.transceibe_Files_RadioButton.setChecked(True)
        self.page_process.audio_capture_RadioButton.setText('原生流式录音（sherpa-onnx）')

    def singleAndSlotProcess(self):
        """
        process single connect and others
        """
        self.statusToolSignalStore.LoadModelSignal.connect(self.loadModelResult)
        self.statusToolSignalStore.LoadModelSignal.connect(self.setModelStatusLabelTextForAll)

        self.page_model.toolPushButton_get_model_path.clicked.connect(self.getLocalModelPath)
        self.page_model.button_convert_model.clicked.connect(self.onButtonConvertModelClicked)

        set_model_output_dir = lambda path: path if path != "" else self.page_model.LineEdit_model_out_dir.text()
        self.page_model.button_set_model_out_dir.clicked.connect(lambda:self.page_model.LineEdit_model_out_dir.setText(set_model_output_dir(QFileDialog.getExistingDirectory(self,"选择转换模型输出目录", self.page_model.LineEdit_model_out_dir.text()))) )
        self.page_model.button_download_root.clicked.connect(self.getDownloadCacheDir)
        self.page_model.button_model_lodar.clicked.connect(self.onModelLoadClicked)
        self.page_process.button_process.clicked.connect(self.onButtonProcessClicked)
        self.page_process.processResultText.textChanged.connect(lambda: self.page_process.processResultText.moveCursor(QTextCursor.MoveOperation.End, mode=QTextCursor.MoveMode.MoveAnchor))
        self.page_process.fileNameListView.ignore_files_signal.connect(lambda ignore_files_info: self.raiseInfoBar(self._tr("忽略文件"), ignore_files_info["ignore_reason"]+"\n"+"\n".join(ignore_files_info["ignore_files"])))

        self.page_home.itemLabel_demucs.mainButton.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_demucs))
        self.page_home.itemLabel_faster_whisper.mainButton.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_process))
        self.page_home.itemLabel_whisperx.mainButton.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_output))
        self.page_home.itemLabel_faster_whisper.subButton.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_transcribes))

        self.page_output.outputSubtitleFileButton.clicked.connect(self.outputSubtitleFile)
        self.page_output.WhisperXAligmentTimeStampleButton.clicked.connect(self.whisperXAligmentTimeStample)
        self.page_output.WhisperXSpeakerDiarizeButton.clicked.connect(self.whisperXDiarizeSpeakers)
        self.page_output.tableTab.tabBar.tabAddRequested.connect(self.openExcitedFiles)
        self.page_output.tableTab.signal_delete_table.connect(self.deleteResultTableEvent)
        self.page_output.unloadWhisperModelPushbutton.clicked.connect(self.unloadWhisperModel)
        self.page_output.outputAudioPartWithSpeakerButton.clicked.connect(self.outputAudioPartWithSpeaker)

        self.page_demucs.process_button.clicked.connect(self.demucsProcess)
        self.page_demucs.fileListView.ignore_files_signal.connect(lambda ignore_files_info: self.raiseInfoBar(self._tr("忽略文件"), ignore_files_info["ignore_reason"]+"\n"+"\n".join(ignore_files_info["ignore_files"])))
        
        self.page_setting.pushButton_backupConfigFile.clicked.connect(self.backupConfigFile)
        self.page_setting.pushButton_loadConfigFile.clicked.connect(self.loadBackupConfigFile)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.stateTool is not None:
            width_tool = self.stateTool.width()
            width = self.width()
            self.stateTool.move(width-width_tool-30, 45)
        return
