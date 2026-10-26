
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
                            , FluentIcon
                            , NavigationItemPosition
                            
                        )

from qframelesswindow import (
                                FramelessMainWindow
                            )

from faster_whisper_GUI.config import Language_dict, Preciese_list, Model_names, Device_list

from resource import (rc_Image, rc_qss)

from faster_whisper_GUI.version import __version__, __FasterWhisper_version__, __WhisperX_version__

from faster_whisper_GUI.runtime.preferences import CONFIG_PATH, ConfigError, load_config, validate_values
from faster_whisper_GUI.runtime.paths import MODEL_CACHE_DIR
from faster_whisper_GUI.ui.appearance import apply_appearance
from .settings import configuration_schema, CONFIGURATION_PAGES
from qfluentwidgets import qconfig
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
    
        
    def readConfig(self, config_file_path=None):
        """Load settings without changing the active configuration on failure."""
        if config_file_path is None:
            self.preferences = {}
            return True
        try:
            preferences = load_config(config_file_path)
            if hasattr(self, 'page_model'):
                validate_values(preferences, configuration_schema(self))
        except ConfigError as error:
            log.warning('%s', error)
            return False
        self.preferences = preferences
        return True


    def setConfig(self):
        validate_values(self.preferences, configuration_schema(self))
        apply_appearance(self.preferences.get('ui', {}))
        for section, attribute in CONFIGURATION_PAGES.items():
            if section in self.preferences:
                getattr(self, attribute).setParam(self.preferences[section])
        self.page_setting.colorPickerButton.setColor(qconfig.get(qconfig.themeColor))
        self.page_setting.setThemeColorAndText()
        if not self.page_model.LineEdit_download_root.text().strip():
            self.page_model.LineEdit_download_root.setText(str(MODEL_CACHE_DIR))
        self.download_cache_path = self.page_model.LineEdit_download_root.text().strip()
    
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

        self.preferences = {}
        loaded = self.readConfig(CONFIG_PATH)
        apply_appearance(self.preferences.get('ui', {}))

        self.setupUI()
        self.initWin()

        try:
            self.setConfig()
        except ConfigError as error:
            log.warning('%s', error)
            loaded = False
            self.preferences = {}
        if not loaded and CONFIG_PATH.exists():
            self.page_setting.switchButton_saveConfig.setChecked(False)

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
        StyleSheet.MAIN_WINDOWS.apply(self)
        
        # self.resize(800, 500)
        self.setGeometry(100, 100, 1250, 915)

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
