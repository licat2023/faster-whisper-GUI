
# coding:utf-8

import logging
import math
import os
# from pathlib import Path

from PySide6.QtCore import  ( 
                                QCoreApplication,
                                QTranslator,
                                Qt
                            )

from PySide6.QtWidgets import  (
                                # QApplication,
                                QSpacerItem,
                                QWidget
                                , QStackedWidget
                                , QVBoxLayout
                                , QHBoxLayout
                                , QGridLayout
                                # , QMainWindow
                            )

from PySide6.QtGui import QIcon

from qfluentwidgets import (
                            NavigationInterface
                            , NavigationWidget
                            , setTheme
                            , setThemeColor
                            , Theme
                            , FluentIcon
                            , NavigationItemPosition
                            
                        )

from qframelesswindow import (
                                FramelessMainWindow
                            )

from faster_whisper_GUI.config import Language_dict, Preciese_list, Model_names, Device_list

from resource import (rc_Image, rc_qss)
import json

from faster_whisper_GUI.version import __version__, __FasterWhisper_version__, __WhisperX_version__

from faster_whisper_GUI.ui.styles import StyleSheet
from faster_whisper_GUI.ui.translation import TRANSLATOR
from resource import rc_Translater

from faster_whisper_GUI.ui.pages.model import ModelNavigationInterface
from faster_whisper_GUI.ui.pages.transcription import TranscribeNavigationInterface
from faster_whisper_GUI.ui.pages.vad import VADNavigationInterface
from faster_whisper_GUI.ui.pages.process import ProcessPageNavigationInterface
from faster_whisper_GUI.ui.pages.output import OutputPageNavigationInterface
from faster_whisper_GUI.ui.pages.home import HomePageNavigationinterface
from faster_whisper_GUI.ui.pages.separation import DemucsPageNavigation
from faster_whisper_GUI.ui.icons import FasterWhisperGUIIcon
from faster_whisper_GUI.ui.widgets.titlebar import WindowTitleBar
from faster_whisper_GUI.ui.pages.settings import SettingPageNavigationInterface

log = logging.getLogger(__name__)

# =======================================================================================
# UI
# =======================================================================================
class UIMainWin(FramelessMainWindow):
    """创建窗口布局和功能页面，并将配置应用到控件。"""
    
    # def tr(self, text):
    #     return QCoreApplication.translate(self.__class__.__name__, text)
        
    def readConfigJson(self, config_file_path: str = ""):
        """
        读取配置文件；缺失或损坏时退回全默认值。

        这里必须容错，原因有两个：
         1. 该文件是程序退出时自己写的运行时状态（见 MainWindows.saveConfig），
            不该被视为"必须随源码分发"的资源；
         2. 旧实现直接 open()，文件不存在就抛 FileNotFoundError，而调用点没有
            try/except —— 新克隆 / 刚清空配置的用户会**启动即崩**。
        """
        self.default_theme = "light"
        self.model_param = {}
        self.setting = {}
        self.demucs = {}
        self.Transcription_param = {}
        self.output_whisperX_param = {}
        self.vad_param = {}
        self.use_auth_token_speaker_diarition = ""

        if not config_file_path:
            return True

        config_path = os.path.abspath(config_file_path)
        try:
            with open(config_path, "r", encoding="utf8") as fp:
                json_data = json.load(fp)
        except FileNotFoundError:
            log.info("配置文件不存在，使用默认设置: %s", config_path)
            return False
        except (OSError, ValueError) as error:
            log.warning("配置文件无法读取或 JSON 解析失败，使用默认设置: %s (%s)",
                        config_path, error)
            return False

        if not isinstance(json_data, dict):
            log.warning("配置文件顶层不是对象，使用默认设置: %s", config_path)
            return False

        sections = ('model_param', 'setting', 'demucs', 'Transcription_param', 'output_whisperX', 'vad_param')
        if any(json_data.get(name) is not None and not isinstance(json_data[name], dict) for name in sections):
            log.warning('配置节必须是 JSON 对象: %s', config_path)
            return False

        # 各配置节独立容错：某一节缺失/为 null 只影响它自己，不再整体失败。
        # 旧实现对每一节各写一个 bare except，既丢掉了出错信息，也让"缺了哪一节"
        # 变得不可见；这里用 get + 空值兜底，语义相同但可读、可日志。
        self.default_theme = json_data.get("theme") or "light"
        self.model_param = json_data.get("model_param") or {}
        self.setting = json_data.get("setting") or {}
        self.demucs = json_data.get("demucs") or {}
        self.Transcription_param = json_data.get("Transcription_param") or {}
        self.output_whisperX_param = json_data.get("output_whisperX") or {}
        self.vad_param = json_data.get("vad_param") or {}

        missing = [name for name in ("theme", "model_param", "setting", "demucs",
                                     "Transcription_param", "output_whisperX", "vad_param")
                   if not json_data.get(name)]
        if missing:
            log.info("配置文件缺少以下配置节，将使用默认值: %s", ", ".join(missing))
        return True


    def setConfig(self):

        setTheme(Theme.DARK if self.default_theme == "dark" else Theme.LIGHT, save=True, lazy=True)
        # setThemeColor("#aaff009f")
        if self.model_param != {}:
            self.page_model.setParam(self.model_param)
            if not self.model_param.get("download_root", ""):
                # 获取默认下载目录
                userDir = os.path.expanduser("~")
                cache_dir = os.path.join(userDir,".cache","huggingface","hub").replace("\\", "/")
                self.download_cache_path = cache_dir
                self.page_model.LineEdit_download_root.setText(cache_dir)
            else:
                self.download_cache_path = self.model_param.get("download_root", "")

        if self.setting != {}:
            self.page_setting.setParam(self.setting)
        
        if self.demucs != {}:
            self.page_demucs.setParam(self.demucs)

        if self.Transcription_param != {}:
            self.page_transcribes.setParam(self.Transcription_param)

        if self.output_whisperX_param != {}:
            self.page_output.setParam(self.output_whisperX_param)
        
        if self.vad_param != {}:
            self.page_VAD.setParam(self.vad_param)
    
    def __init__(self, parent=None, f=None) -> None:
        super().__init__()

        # self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)  

        self.model_path = ""
        self.model_names = Model_names

        # 模型支持的计算设备
        self.device_list = Device_list
        # 模型支持的计算精度
        self.preciese_list = Preciese_list
        # 语言支持
        self.LANGUAGES_DICT = Language_dict

        self.FasterWhisperModel = None

        # UI设置
        self.setupUI()
        self.initWin()

        # 读配置文件
        self.readConfigJson(r"./fasterWhisperGUIConfig.json")
        # 设置配置
        self.setConfig()

        try:
            self.setWidgetsStatusFromConfig()
        except Exception as e:
            log.error("%s", str(e))

    def setWidgetsStatusFromConfig(self):
        # 根据读取的配置设置完控件状态之后，根据控件状态设置相关属性
        self.page_output.tableTab.onDisplayModeChanged(self.page_output.tableTab.closeDisplayModeComboBox.currentIndex())
        self.page_output.tableTab.tabBar.setMovable(self.page_output.tableTab.movableCheckBox.isChecked())
        self.page_output.tableTab.tabBar.setScrollable(self.page_output.tableTab.scrollableCheckBox.isChecked())
        self.page_output.tableTab.tabBar.setTabShadowEnabled(self.page_output.tableTab.shadowEnabledCheckBox.isChecked())
        self.page_output.tableTab.tabBar.setTabMaximumWidth(self.page_output.tableTab.tabMaxWidthSpinBox.value())


    def initWin(self):

        self.setObjectName("FramlessMainWin")
        # setTheme(Theme.LIGHT)
        StyleSheet.MAIN_WINDOWS.apply(self)
        
        # self.resize(800, 500)
        self.setGeometry(100, 100, 1250, 915)

        # TODO: 添加标题栏 
        self.setTitleBar(WindowTitleBar(self))
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        self.setWindowTitle(f"FasterWhisperGUI-{__version__}--fw-{__FasterWhisper_version__}--WhisperX-{__WhisperX_version__}")
        
        self.setWindowIcon(QIcon(":/resource/Image/microphone.png"))
        

    def setupUI(self):
        
        # =====================================================================================
        # 创建窗体中心控件
        self.mainWindowsWidget = QWidget(self)
        self.mainWindowsWidget.setObjectName("mainWidget")

        # 创建窗体主布局
        self.mainLayout = QGridLayout()

        # 将主布局添加到窗体中心控件
        self.mainWindowsWidget.setLayout(self.mainLayout)

        # 导航布局
        self.vBoxLayout = QVBoxLayout()

        # 将导航布局添加到主布局
        self.mainLayout.addLayout(self.vBoxLayout,0,0)

        # 设置窗体中心控件
        self.setCentralWidget(self.mainWindowsWidget)

        # 创建一个空对象 用于改善布局顶部
        self.spacer_main = QSpacerItem(0,25)
        self.vBoxLayout.addItem(self.spacer_main)

        # 设置显示图层到最后避免遮挡窗体按钮
        self.mainWindowsWidget.lower()
        self.lower()

        # 创建布局用于放置导航枢和分页
        self.mainHBoxLayout = QHBoxLayout()
        self.vBoxLayout.addLayout(self.mainHBoxLayout)

        # 创建窗体导航枢 和 stacke 控件
        self.pivot = NavigationInterface(self, showMenuButton=True, showReturnButton=False)
        self.pivot.setObjectName("pivot")

        self.stackedWidget = QStackedWidget(self)

        self.mainHBoxLayout.addWidget(self.pivot)
        self.mainHBoxLayout.addWidget(self.stackedWidget)
        
        self.pages = []
        
        # 添加子界面
        self.page_home = HomePageNavigationinterface(self)
        self.addSubInterface(self.page_home, "pageHome", self.tr("Home"), icon=FluentIcon.HOME)
        self.pages.append(self.page_home)

        self.page_demucs = DemucsPageNavigation(self)
        self.addSubInterface(self.page_demucs, "pageDecums", self.tr("声乐分离"), icon=FasterWhisperGUIIcon.DEMUCS)
        self.pages.append(self.page_demucs)

        self.page_model = ModelNavigationInterface(self)
        self.addSubInterface(self.page_model, "pageModelParameter", self.tr("模型参数"), icon=FluentIcon.BOOK_SHELF)
        self.pages.append(self.page_model)

        self.page_VAD = VADNavigationInterface(self)
        self.addSubInterface(self.page_VAD, "pageVADParameter", self.tr("人声活动检测"), icon=FasterWhisperGUIIcon.VAD_PAGE)
        self.pages.append(self.page_VAD)

        self.page_transcribes = TranscribeNavigationInterface(self)
        self.addSubInterface(self.page_transcribes, "pageTranscribesParameter", self.tr("转写参数"), icon=FasterWhisperGUIIcon.TRANSCRIPTION_PAGE)
        self.pages.append(self.page_transcribes)

        self.page_process = ProcessPageNavigationInterface(self)
        self.addSubInterface(self.page_process, "pageProcess", self.tr("执行转写"), icon=FasterWhisperGUIIcon.HEAD_PHONE)
        self.pages.append(self.page_process)

        self.page_output = OutputPageNavigationInterface(self)
        self.addSubInterface(self.page_output, "pageOutput", self.tr("whiperX及字幕编辑"), icon=FluentIcon.SAVE_AS)
        self.pages.append(self.page_output)

        self.page_setting = SettingPageNavigationInterface(self)
        self.addSubInterface(
            self.page_setting, "pageSetting", self.tr('设置'), FluentIcon.SETTING, NavigationItemPosition.BOTTOM)
        self.pages.append(self.page_setting)
        self.fitNavigationWidth()
        
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.page_home)
        self.pivot.setCurrentItem(self.page_home.objectName())

    
    def fitNavigationWidth(self):
        """按最长导航内容收紧展开宽度，保留与左侧图标一致的右侧留白。"""
        panel = self.pivot.panel
        left_margin = panel.topLayout.contentsMargins().left()
        # Fluent 导航图标起点为 11.5px，文字绘制区域自带 13px 右留白。
        padding = math.ceil(panel.frameWidth() + left_margin + 11.5)
        right_margin = max(0, padding - panel.frameWidth() - 13)
        content_width = 40
        for entry in panel.items.values():
            widget = entry.widget
            if hasattr(widget, 'suitableWidth'):
                metrics = widget.itemWidget.fontMetrics()
                text_width = max(metrics.horizontalAdvance(widget.text()),
                                 metrics.boundingRect(widget.text()).width())
                margins = widget._margins()
                text_left = 44 if not widget.icon().isNull() else 16
                content_width = max(content_width, text_left + margins.left()
                                    + text_width + margins.right() + 13)

        width = content_width + left_margin + right_margin + 2 * panel.frameWidth()
        self.pivot.setExpandWidth(width)
        for layout in (panel.topLayout, panel.scrollLayout, panel.bottomLayout):
            margins = layout.contentsMargins()
            layout.setContentsMargins(margins.left(), margins.top(),
                                      right_margin, margins.bottom())
            layout.setAlignment(layout.alignment() | Qt.AlignLeft)
        # 使用实例宽度，确保文字区域完整，并保留原有折叠尺寸。
        for widget in panel.findChildren(NavigationWidget):
            widget.EXPAND_WIDTH = content_width

    def addSubInterface(self, layout: QWidget, objectName, text: str, icon:QIcon=None,position=NavigationItemPosition.TOP ):
        layout.setObjectName(objectName)
        self.stackedWidget.addWidget(layout)
        self.pivot.addItem(
            routeKey=objectName
            ,text=text
            ,onClick=lambda: self.stackedWidget.setCurrentWidget(layout)
            ,icon=icon
            ,position=position
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())
