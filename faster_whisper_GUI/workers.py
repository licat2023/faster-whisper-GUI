# coding:utf-8
"""
QThread 的统一异常护栏。

背景
----
Qt 在自己的事件循环里调用 Python 覆写的 QThread.run() 时，如果 run() 抛出异常，
Qt 只会用 qWarning 打一行 "Error calling Python override of QThread::run():"，
**不带 traceback**。而本程序的 sys.stdout / sys.stderr 又被重定向，于是真实的
报错信息可能既进不了日志、也送不到界面 —— 结果就是「界面永远停在正在处理中，
日志停在最后一行」这种最难排查的状态。

在此之前，只有 TranscribeWorker.run() 自己做了捕获（而且它把 traceback 直接写
文件、绕过了重定向）；OutputWorker（唯一写用户字幕文件的代码）完全没有保护；
LoadModelWorker 则是捕获后重新抛出，等于没有保护。

用法
----
把类的基类从 QThread 换成 GuardedWorker 即可，run() 不需要改名或改写：

    class DemucsWorker(GuardedWorker):
        def run(self):          # 保持原样
            ...

需要恢复界面状态时，额外实现 onError()：

    def onError(self, exc):
        self.signal_vr_over.emit(False)

护栏通过 __init_subclass__ 在类定义时自动包住 run()，因此不依赖任何调用点配合，
也不可能被忘记。
"""

from __future__ import annotations

import functools
import logging
import traceback

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


def _guard(run_func):
    """把 run() 包一层：异常一定落日志，并给子类一次恢复界面的机会。"""

    @functools.wraps(run_func)
    def wrapper(self, *args, **kwargs):
        try:
            return run_func(self, *args, **kwargs)
        except Exception as exc:                       # noqa: BLE001 - 这里就是要全接
            detail = traceback.format_exc()
            # 用 critical：线程带着异常退出属于用户可见的失败，
            # traceback 必须完整落盘，否则事后无从查起。
            log.critical("线程 %s 异常终止: %s\n%s",
                         type(self).__name__, exc, detail)
            self.lastError = exc
            self.lastTraceback = detail

            # 让子类把界面从「处理中」恢复过来。它自己再出错也不能影响日志。
            try:
                on_error = getattr(self, "onError", None)
                if callable(on_error):
                    on_error(exc)
            except Exception:                          # noqa: BLE001
                log.exception("onError() 本身抛出异常，界面状态可能未恢复")

            # 统一的失败信号，便于界面挂通用错误提示
            try:
                self.failed.emit(f"{type(self).__name__}: {exc}")
            except Exception:                          # noqa: BLE001
                log.debug("failed 信号发送失败", exc_info=True)
            return None

    wrapper._guarded = True                            # type: ignore[attr-defined]
    return wrapper


class GuardedWorker(QThread):
    """
    带异常护栏的 QThread 基类。

    子类照常实现 run()；护栏在类定义时自动生效。
    """

    #: 任何未捕获异常都会发一次这个信号（附一段可读描述）
    failed = Signal(str)

    #: 最近一次异常与其 traceback（供界面或诊断包使用）
    lastError: BaseException | None = None
    lastTraceback: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        run_func = cls.__dict__.get("run")
        if run_func is not None and not getattr(run_func, "_guarded", False):
            cls.run = _guard(run_func)

    def onError(self, exc: BaseException) -> None:
        """
        子类可覆写：把界面从「处理中」恢复过来。

        默认什么也不做 —— 日志已经记录了，界面状态由子类决定。
        """
