"""配置备份、恢复、保存和退出。"""

import logging
import json
import os
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QTimer
from pathlib import Path
from qfluentwidgets import MessageBox, isDarkTheme
from faster_whisper_GUI import logging_setup
from faster_whisper_GUI.runtime.diagnostics import outputWithDateTime

log = logging.getLogger(__name__)

WINDOW_WORKERS = ('audio_capture_thread', 'audio_stream_worker', 'transcribe_thread',
                  'whisperXWorker', 'outputWorker', 'demucsWorker', 'loadModelWorker',
                  'splitAudioFileWithSpeakerWorker')

class SettingsActions:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def backupConfigFile(self):
        config_file_path,_ = QFileDialog.getSaveFileName(
                                                            self,
                                                            self._tr("选择保存位置"), 
                                                            r"./", 
                                                            "json file(*.json)"
                                                        )
        if not config_file_path:
            return
        
        self.saveConfig(config_file_path)

        # shutil.copy(config_file_path, config_file_path+".bak")
        self.raiseInfoBar(self._tr("备份配置文件成功"), self._tr("配置文件已备份到:\n") + config_file_path)

    def loadBackupConfigFile(self):

        if any(worker is not None and worker.isRunning()
               for worker in (getattr(self, name, None) for name in WINDOW_WORKERS)):
            self.raiseErrorInfoBar(self._tr('任务正在运行'), self._tr('请等待当前任务结束后再加载配置。'))
            return

        config_file_name, _ = QFileDialog.getOpenFileName(
                                                            self,
                                                            self._tr("选择配置文件"),
                                                            r"./",
                                                            "json file(*.json)"
                                                        )

        if not config_file_name:
            return

        try:    
            if self.readConfigJson(config_file_path=config_file_name) is False:
                raise ValueError(self._tr('配置文件不存在、格式错误或无法读取。'))
            self.setConfig()
            self.setWidgetsStatusFromConfig()
            self.raiseInfoBar(self._tr("加载配置文件成功"), self._tr("配置文件已加载:\n") + config_file_name)

        except Exception as e:
            # 原先这里调用 self.raiseErrorBar(...)，但该方法在本类里从未定义
            # （只有 raiseErrorInfoBar / raiseSuccessInfoBar / raiseInfoBar）——
            # 于是 except 里又抛 AttributeError，把真正的失败原因盖掉。
            log.error("加载配置文件失败: %s", e, exc_info=True)
            self.raiseErrorInfoBar(self._tr("加载配置文件失败"), self._tr("配置文件加载失败:\n") + str(e))

    def closeEvent(self, event) -> None:
        """Drain workers through the event loop before destroying their Qt owners."""
        if not getattr(self, '_closingRequested', False):
            if not MessageBox(self._tr('退出'), self._tr('是否要退出程序？'), self).exec():
                event.ignore()
                return
            self._closingRequested = True
            self.setEnabled(False)
            for name in WINDOW_WORKERS:
                worker = getattr(self, name, None)
                if worker is not None and worker.isRunning():
                    worker.stop()
            backend = getattr(self, 'FasterWhisperModel', None)
            if hasattr(backend, 'close'):
                backend.close()
        workers = [getattr(self, name, None) for name in WINDOW_WORKERS]
        if any(worker is not None and worker.isRunning() for worker in workers):
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        # A loader may have published its model while shutdown was pending.
        backend = getattr(self, 'FasterWhisperModel', None)
        if hasattr(backend, 'close'):
            backend.close()
        if self.page_setting.switchButton_saveConfig.isChecked():
            try:
                self.saveConfig(config_file_name=os.path.abspath('./fasterWhisperGUIConfig.json'))
            except (OSError, ValueError):
                log.error('退出时保存配置失败', exc_info=True)
        if self.page_setting.switchButton_autoClearTempFiles.isChecked():
            for subtitle in Path('./temp').glob('*.srt'):
                try:
                    subtitle.unlink()
                except OSError:
                    log.warning('清理临时字幕失败: %s', subtitle, exc_info=True)
        log.info('程序退出')
        logging_setup.shutdownLogging()
        self.FasterWhisperModel = None
        event.accept()

    def saveConfig(self, config_file_name: str = ""):
        
        if config_file_name == "":
            return
        
        outputWithDateTime("SaveConfigFile")
        model_param = self.page_model.getParam()
        setting_param = self.page_setting.getParam()    
        demucs_param = self.page_demucs.getParam()
        Transcription_param = self.page_transcribes.getParam()
        output_whisperX_param = self.page_output.getParam()
        vad_param = self.page_VAD.getParam()

        config_json = {
                        "theme":"dark" if isDarkTheme() else "light",
                        "demucs":demucs_param,
                        "model_param" : model_param,
                        "vad_param": vad_param,
                        "setting":setting_param,
                        "Transcription_param" : Transcription_param,
                        "output_whisperX":output_whisperX_param
                    }
        
        with open(os.path.abspath(config_file_name),'w',encoding='utf8')as fp:
            json.dump(
                        config_json,
                        fp,
                        ensure_ascii=False,
                        indent=4
                    )
