"""日志投递、进度提示和消息栏。"""

import logging
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Qt
from qfluentwidgets import StateToolTip, InfoBar, InfoBarPosition
from faster_whisper_GUI import logging_setup

log = logging.getLogger(__name__)

class WindowFeedback:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def setTextAndMoveCursorToProcessBrowser(self, text:str):
        self.page_process.processResultText.moveCursor(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.MoveAnchor)
        self.page_process.processResultText.insertPlainText(text)

    def redirectOutput(self, target: callable):
        """
        让日志同时显示到界面文本框。

        文件那一份由 logging 的 FileHandler 直接写盘，不受界面影响；
        这里只是额外挂一个出口。旧实现把 sys.stdout 换成一个 Qt 信号对象，
        结果是「信号没派发出去 = 日志里也什么都没有」，排查时最难办的正是这种状态。

        注意线程：logging 的回调是在发日志的那个线程里被同步调用的，而 target 是
        界面槽。旧实现把 target 直接交给 logging，等于从工作线程操作 QTextEdit。
        现在统一 emit signal_guiLog，由 Qt 决定直连还是排队投递。
        """
        logging_setup.detachHandler(self._guiHandler)

        # 调用点每次都传同一个槽；重复 connect 会让一条日志投递多次，所以换目标时先断开
        if target != self._guiLogTarget:
            if self._guiLogTarget is not None:
                try:
                    self.signal_guiLog.disconnect(self._guiLogTarget)
                except (RuntimeError, TypeError):
                    log.debug("断开旧的日志界面出口失败", exc_info=True)
            self.signal_guiLog.connect(target)
            self._guiLogTarget = target

        self._guiHandler = logging_setup.attachGuiHandler(self.signal_guiLog.emit)

    def setStateTool(self, title:str="", text:str="", status:bool=False):

        if self.stateTool is None:
            self.stateTool = StateToolTip(title, text , self)
            self.stateTool.show()

        else:
            self.stateTool.setContent(text)

        width = self.width()
        self.stateTool.move(width-self.stateTool.width()-30, 45)
        self.stateTool.setState(status)

        if  status:    
            self.stateTool = None

    def raiseErrorInfoBar(self, title:str, content:str):
        InfoBar.error(
                        title=title
                        , content=content
                        , isClosable=True
                        , duration=-1
                        , orient=Qt.Orientation.Horizontal
                        , position=InfoBarPosition.TOP
                        # , position='Custom',   # NOTE: use custom info bar manager
                        , parent=self
                    )

    def raiseSuccessInfoBar(self, title:str, content:str):
        InfoBar.success(
                        title=title
                        , content=content
                        , isClosable=True
                        , duration=5000
                        , position=InfoBarPosition.TOP
                        , parent=self
                    )

    def raiseInfoBar(self, title:str, content:str ):
        InfoBar.info(
                title=title
                , content=content
                , isClosable=False
                , duration=2000
                , position=InfoBarPosition.TOP_RIGHT
                , parent=self
            )
