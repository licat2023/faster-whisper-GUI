"""主窗口任务状态信号。"""

from PySide6.QtCore import QObject, Signal

class statusToolsSignalStore(QObject):
    StateToolSignal = Signal(bool)
    LoadModelSignal = Signal(bool)
