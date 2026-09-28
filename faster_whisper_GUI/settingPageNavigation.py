# coding:utf-8

import logging
import os
import random
import time
from PySide6.QtCore import (QAbstractListModel, QCoreApplication, Qt)
from PySide6.QtGui import QColor, QColorSpace

from PySide6.QtWidgets import (
                                QHBoxLayout, 
                                QVBoxLayout, 
                                QWidget
                            )

from . import logging_setup

from qfluentwidgets import (
                            SwitchButton, 
                            ComboBox, 
                            LineEdit,
                            MessageBox,
                            PushButton,
                            TitleLabel,
                            ScrollArea,
                            themeColor,
                            setThemeColor,
                            ColorPickerButton,
                            FluentIcon,
                            PrimaryToolButton
                        )

from .paramItemWidget import ParamWidget
from .style_sheet import StyleSheet
from .config import default_Huggingface_user_token, THEME_COLORS

from .util import outputWithDateTime

log = logging.getLogger(__name__)

class ThemeColorModel(QAbstractListModel):
    def data(self, index, role):
        if role == Qt.DisplayRole:
            color = THEME_COLORS[index.row()] 
            return f'<div style="background-color:{color}; width:20px; height:20px"/>'

class SettingPageNavigationInterface(ScrollArea):
    def __tr(self, text):
        return QCoreApplication.translate(self.__class__.__name__, text)
    """
    This class is used to navigate to the settings page
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.themeColor_str = themeColor().name()

        # self.layout = QGridLayout(self)
        # self.setLayout(self.layout)

        self.mainWidget = QWidget(self)
        self.mainWidget.setObjectName("mainObject")
        # self.layout.addWidget(self.mainWidget, 0,0)
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidget(self.mainWidget)
        self.setWidgetResizable(True)

        self.mainLayout.setSpacing(10)
        self.mainLayout.setContentsMargins(36, 10, 36, 10)

        self.setupUI()
        self.signalAndSlotProcess()
        StyleSheet.SETTINGPAGEINTERFACE.apply(self)

        

    def addWidget(self, widget):
        self.mainLayout.addWidget(widget)
    def addLayout(self, layout):
        self.mainLayout.addLayout(layout)

    def setupUI(self):

        self.mainLayout.addSpacing(10)

        self.titleLabel_title = TitleLabel(self.__tr("软件设置"))
        self.addWidget(self.titleLabel_title)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.switchButton_saveConfig = SwitchButton()
        self.switchButton_saveConfig.setChecked(True)
        self.paramItemWidget_saveConfig = ParamWidget(self.__tr("自动保存配置"), self.__tr("程序退出时自动保存配置"),self.switchButton_saveConfig, self)
        self.addWidget(self.paramItemWidget_saveConfig)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.switchButton_autoLoadModel = SwitchButton()
        self.switchButton_autoLoadModel.setChecked(True)
        self.paramItemWidget_autoLoadModel = ParamWidget(self.__tr("自动加载模型"), self.__tr("程序启动时自动加载模型"),self.switchButton_autoLoadModel, self)
        self.addWidget(self.paramItemWidget_autoLoadModel)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.pushButton_backupConfigFile = PushButton()
        self.pushButton_backupConfigFile.setText(self.__tr("备份当前配置"))
        self.pushButton_loadConfigFile = PushButton()
        self.pushButton_loadConfigFile.setText(self.__tr("加载配置文件"))

        self.paramItemWidget_ConfigFile = ParamWidget(self.__tr("配置文件"), self.__tr("备份、加载当前配置文件，重装、升级软件时可用来备份软件配置"), self.pushButton_backupConfigFile)
        self.paramItemWidget_ConfigFile.widgetVLayout.addWidget(self.pushButton_loadConfigFile)
        self.addWidget(self.paramItemWidget_ConfigFile)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.combox_language = ComboBox()
        self.combox_language.addItems(["中文","English",self.__tr("自动")])
        self.combox_language.setCurrentIndex(2)
        self.paramItemWidget_language = ParamWidget(self.__tr("语言"), self.__tr("程序界面语言，修改后重启生效"),self.combox_language, self)
        self.addWidget(self.paramItemWidget_language)
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        self.colorPickerButton = ColorPickerButton(self.themeColor_str, "ThemeColor", parent=self)
        # self.colorPickerButton.clicked.disconnect()
        
        self.randomPickThemeColorToolButton = PrimaryToolButton()
        # self.randomPickThemeColorToolButton = ToolButton()
        self.randomPickThemeColorToolButton.setIcon(FluentIcon.BASKETBALL)
        self.randomPickThemeColorToolButton.setToolTip(self.__tr("切换预置主题色"))
        
        self.themeColorLineEdit = LineEdit()
        self.themeColorLineEdit.setText(self.themeColor_str)
        
        self.paramItemWidget_themeColor = ParamWidget(self.__tr("主题色"), self.__tr("主题配色"), self.themeColorLineEdit, self)

        hb = QHBoxLayout()
        hb.addWidget(self.randomPickThemeColorToolButton)
        hb.addWidget(self.colorPickerButton)
        hb.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # self.paramItemWidget_themeColor.widgetVLayout.addWidget(self.themeColorLineEdit)
        self.paramItemWidget_themeColor.widgetVLayout.addLayout(hb)

        self.addWidget(self.paramItemWidget_themeColor)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.LineEdit_use_auth_token = LineEdit()
        self.LineEdit_use_auth_token.setFixedWidth(330)
        self.use_auth_token_param_widget = ParamWidget(self.__tr("HuggingFace用户令牌"),
                                                        self.__tr("访问声源分析、分离模型需要提供经过许可的 HuggingFace 用户令牌。\n如果默认令牌失效可以尝试自行注册账号并生成、刷新令牌"),
                                                        self.LineEdit_use_auth_token 
                                                    )
        
        self.addWidget(self.use_auth_token_param_widget)
        self.use_auth_token_param_widget.mainHLayout.setStretch(0,4)
        self.use_auth_token_param_widget.mainHLayout.setStretch(1,1)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.combox_autoGoToOutputPage = ComboBox()
        self.combox_autoGoToOutputPage.addItems([self.__tr("总是"), self.__tr("从不"),self.__tr("询问")])
        self.combox_autoGoToOutputPage.setCurrentIndex(2)

        self.paramItemWidget_autoGoToOutputPage = ParamWidget(self.__tr("转写结束后自动跳转"), self.__tr("转写结束后是否自动跳转到输出页, 目前怀疑自动跳转功能和窗体崩溃有关"), self.combox_autoGoToOutputPage)
        self.addWidget(self.paramItemWidget_autoGoToOutputPage)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.switchButton_autoClearTempFiles = SwitchButton()
        self.switchButton_autoClearTempFiles.setOnText(self.__tr("清除"))
        self.switchButton_autoClearTempFiles.setOffText(self.__tr("不清除"))

        self.paramItemWidget_autoClearTempFiles = ParamWidget(self.__tr("自动清除临时文件"), self.__tr("程序退出时是否自动清除临时文件"), self.switchButton_autoClearTempFiles)
        self.addWidget(self.paramItemWidget_autoClearTempFiles)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------

        self.pushButton_clearTempFiles = PushButton()
        self.pushButton_clearTempFiles.setText(self.__tr("清除"))
        self.pushButton_openTempDir = PushButton()
        self.pushButton_openTempDir.setText(self.__tr("打开目录"))

        self.paramItemWidget_TempDir = ParamWidget(self.__tr("临时文件"), self.__tr("转写完成后, 临时文件将保存在该目录下, 仅保存 srt 格式文件"), self.pushButton_openTempDir)
        self.paramItemWidget_TempDir.widgetVLayout.addWidget(self.pushButton_clearTempFiles)
        self.addWidget(self.paramItemWidget_TempDir)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.pushButton_openLogFile = PushButton()
        self.pushButton_openLogFile.setText(self.__tr("打开"))
        self.paramItemWidget_logFile = ParamWidget(self.__tr("日志文件"), self.__tr("程序运行的日志将保存到该文件中"), self.pushButton_openLogFile)
        self.addWidget(self.paramItemWidget_logFile)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.pushButton_openFWLogFile = PushButton()
        self.pushButton_openFWLogFile.setText(self.__tr("打开"))
        self.paramItemWidget_FWlogFile = ParamWidget(self.__tr("faster-whisper 日志文件"), self.__tr("faster-whisper 转写的日志将保存到该文件中，\n转写过程中如果发生崩溃请参看"), self.pushButton_openFWLogFile)
        self.addWidget(self.paramItemWidget_FWlogFile)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 日志目录 / 诊断包
        # 每次运行单独一个日志文件（永不覆盖），并自动保留最近若干次。
        # 「导出诊断包」把本次日志 + 环境快照打成一个 zip，用户直接发出来即可，
        # 不必再让他去找文件、也不用来回追问环境。
        self.pushButton_openLogDir = PushButton()
        self.pushButton_openLogDir.setText(self.__tr("打开目录"))

        self.pushButton_exportDiagnostics = PushButton()
        self.pushButton_exportDiagnostics.setText(self.__tr("导出诊断包"))

        self.paramItemWidget_logDir = ParamWidget(
            self.__tr("日志目录"),
            self.__tr("每次运行生成一个日志文件，并保留最近 20 次，"
                      "不会因为重新启动而丢失上一次的记录"),
            self.pushButton_openLogDir)
        self.paramItemWidget_logDir.widgetVLayout.addWidget(
            self.pushButton_exportDiagnostics)
        self.addWidget(self.paramItemWidget_logDir)

    def setSwitchStatus(self):
        self.switchButton_autoLoadModel.setChecked(False)
        self.paramItemWidget_autoLoadModel.setEnabled(self.switchButton_saveConfig.isChecked())

    def signalAndSlotProcess(self):
        self.switchButton_saveConfig.checkedChanged.connect(self.setSwitchStatus)
        self.pushButton_openTempDir.clicked.connect(lambda: os.startfile(os.path.abspath(r"./temp/").replace("\\","/")))
        # 日志路径由 logging_setup 决定（每次运行一个文件，绝对路径，不依赖工作目录）
        self.pushButton_openLogFile.clicked.connect(self.openAppLogFile)
        self.pushButton_openFWLogFile.clicked.connect(self.openFrameworkLogFile)
        self.pushButton_openLogDir.clicked.connect(
            lambda: self.__openInExplorer(logging_setup.getLogDir()))
        self.pushButton_exportDiagnostics.clicked.connect(self.exportDiagnostics)
        self.pushButton_clearTempFiles.clicked.connect(self.deletTempFiles)
        self.colorPickerButton.colorChanged.connect(self.setThemeColorAndText)
        self.randomPickThemeColorToolButton.clicked.connect(self.setColorAndThemeColorRandom)
        self.themeColorLineEdit.textChanged.connect(self.setThemeColorWithLineEditText)
        

    def openAppLogFile(self):
        """打开本次运行的日志文件。"""
        path = logging_setup.getRunLogPath()
        if path is None or not path.exists():
            self.__openInExplorer(logging_setup.getLogDir())
            return
        os.startfile(str(path))

    def exportDiagnostics(self):
        """
        把本次日志 + faster-whisper 日志 + 环境快照打成一个 zip。

        用户把 zip 发出来即可，不必再去找文件、也不用反复追问环境。
        注意：打包内容不含配置文件（里面存着 HuggingFace token）。
        """
        try:
            bundle = logging_setup.exportDiagnostics()
        except Exception:                              # noqa: BLE001
            log.exception("导出诊断包失败")
            return
        log.info("诊断包已导出: %s", bundle)
        # 打开所在目录让用户直接看到产物（本页没有 InfoBar 辅助方法）
        self.__openInExplorer(bundle.parent)

    def openFrameworkLogFile(self):
        """打开 faster-whisper 库自身的日志。"""
        path = logging_setup.getFrameworkLogPath()
        if path is None or not path.exists():
            self.__openInExplorer(logging_setup.getLogDir())
            return
        os.startfile(str(path))

    def __openInExplorer(self, directory):
        if directory is not None and directory.exists():
            os.startfile(str(directory))

    def deletTempFiles(self):
        outputWithDateTime("clearTempFiles")
        mess_ = MessageBox(self.__tr("注意"), self.__tr("将会清除全部临时文件，是否确定？"), self)
        if mess_.exec():
            try:
                os.system(r"del .\temp\*.srt")
                mess_ = MessageBox(self.__tr("提示"), self.__tr("清除成功"), self)
                mess_.show()
                log.info("%s", "clear over")
            except Exception as e:
                mess_ = MessageBox(self.__tr("错误"), self.__tr("清除失败"), self)
                log.error("%s", f"clear temp files error: \n    {str(e)}")
        else:
            log.info("%s", "clear temp files cancel")
        
    def setParam(self,param:dict) -> None:
        try:
            self.switchButton_saveConfig.setChecked(param["saveConfig"])
            self.switchButton_autoLoadModel.setChecked(param["autoLoadModel"])
            self.combox_language.setCurrentIndex(param["language"])
            self.LineEdit_use_auth_token.setText(param["huggingface_user_token"])
            self.combox_autoGoToOutputPage.setCurrentIndex(param["autoGoToOutputPage"])
            self.switchButton_autoClearTempFiles.setChecked(param["autoClearTempFiles"])
            self.colorPickerButton.setColor(param["themeColor"])
            self.setThemeColorAndText()
            # setThemeColor(param["themeColor"])
        except Exception as e:
            # 裸 except + pass 会让「配置文件某个字段不对」变成查不出来的静默失败，
            # 与 docs/LOGGING.md「绝不静默吞异常」相冲突。
            log.warning("读取设置页配置失败，相关控件保持默认值: %s", e, exc_info=True)
    
    def getParam(self):
        param = {}
        param["saveConfig"] = self.switchButton_saveConfig.isChecked()
        param["autoLoadModel"] = self.switchButton_autoLoadModel.isChecked() if param["saveConfig"] else False
        param["language"] = self.combox_language.currentIndex()
        param["huggingface_user_token"] = self.LineEdit_use_auth_token.text().strip() or default_Huggingface_user_token 
        param["autoGoToOutputPage"] =  self.combox_autoGoToOutputPage.currentIndex()
        param["autoClearTempFiles"] = self.switchButton_autoClearTempFiles.isChecked()
        param["themeColor"] = self.themeColor_str
        return param

    def setColorAndThemeColorRandom(self):
        
        if self.themeColor_str in THEME_COLORS:
            index = THEME_COLORS.index(self.themeColor_str)
            if index != len(THEME_COLORS)-1 :
                index += 1
            else:
                index = 0
        else:
            index = 0

        
        self.themeColor_str = THEME_COLORS[index]
        self.colorPickerButton.setColor(self.themeColor_str)
        setThemeColor(self.themeColor_str, save=True, lazy=True)
        self.themeColorLineEdit.setText(self.themeColor_str)

    def setThemeColorAndText(self):
        self.themeColor_str = self.colorPickerButton.color.name()
        self.themeColorLineEdit.setText(self.themeColor_str)
        setThemeColor(self.themeColor_str, save=True, lazy=True)

    def setThemeColorWithLineEditText(self,text):
        if len(text) in [7,9]:
            self.colorPickerButton.setColor(QColor.fromString(text))
            setThemeColor(QColor.fromString(text), save=True, lazy=True)
            self.themeColor_str = text
        