# coding:utf-8
"""
集中式日志配置。

设计目标
--------
1. **证据必须活下来** —— 每次运行写一个独立文件，永不覆盖上一次。
   原实现用 open(..., 'w') 打开 fasterwhispergui.log，每次启动都会清空历史，
   排查间歇性故障时最需要的「上一次运行」恰好被下一次启动抹掉。

2. **一条管道、多个出口** —— 所有输出（print、第三方库、异常、Qt 消息）统一进入
   logging，再由若干个 Handler 分别送到文件、控制台与界面。原实现是 print 直接写文件、
   错误另走一套、Qt 消息完全没人管，三套互不相干。

3. **print 不需要改也能进日志** —— LoggingStream 在「流」这一层接管，
   天然正确处理 print 的多参数、sep、end，以及第三方库直接 write 的情况。
   调用点再逐步改成带级别的 log.xxx（见 tools/convert_prints.py）。

4. **任何位置都不吞异常** —— sys.excepthook / threading.excepthook /
   qInstallMessageHandler 三个钩子全装。原实现三个都没有，导致
   "Error calling Python override of QThread::run()" 这类信息凭空消失。

日志目录
--------
优先 <程序目录>/logs；不可写时退到 %LOCALAPPDATA%/faster-whisper-GUI/logs，
再不行退到系统临时目录。绝不依赖当前工作目录（原实现全是 './xxx.log'，
从快捷方式启动就会写到别处）。
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import tempfile
import threading
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- 常量

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = (PACKAGE_DIR.parent.parent
            if PACKAGE_DIR.parent.name == 'src' and not getattr(sys, 'frozen', False)
            else Path(os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local/share')) / 'faster-whisper-GUI')

LOG_DIR_NAME = "logs"
RUN_PREFIX = "app-"
FW_PREFIX = "fw-"
LATEST_NAME = "latest.log"
KEEP_RUNS = 20                      # 保留最近多少次运行

FILE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-26s %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"
GUI_FORMAT = "%(message)s"

# 这些第三方库在 DEBUG 下极其吵，压到 WARNING
NOISY_LOGGERS = (
    "urllib3", "PIL", "matplotlib", "asyncio", "numba", "huggingface_hub",
    "filelock", "requests", "OpenGL", "torch.distributed",
)

# --------------------------------------------------------------------------- 状态

_configured = False
_log_dir: Optional[Path] = None
_run_log_path: Optional[Path] = None
_latest_log_path: Optional[Path] = None
_fw_log_path: Optional[Path] = None
_file_handlers: List[logging.Handler] = []

#: 附着在 root logger 上的「已配置」标记。
#: 为什么不用模块级变量：本模块可能被加载成两份实例 —— FasterWhisperGUI.py 为了
#: 抢在 ctranslate2 之前建立日志，是按文件路径加载的；包内其它模块走的是普通导入。
#: 一旦出现两份实例，各自的 _configured 都是 False，第二次 setupLogging() 会用
#: mode="w" 把刚写好的日志清空（实测发生过：启动日志先有 177 字节，随后变 0）。
#: root logger 是进程内唯一的，把状态挂在它上面，才能真正做到「一个进程只配置一次」。
_GUARD_ATTR = "_faster_whisper_gui_logging_state"


def getLogDir() -> Optional[Path]:
    """日志目录（setupLogging() 之后可用）。"""
    return _log_dir


def getRunLogPath() -> Optional[Path]:
    """本次运行的日志文件路径。"""
    return _run_log_path


def getLatestLogPath() -> Optional[Path]:
    """始终指向最近一次运行的文件路径。"""
    return _latest_log_path


def getFrameworkLogPath() -> Optional[Path]:
    """faster-whisper 库自身的日志路径。"""
    return _fw_log_path


# --------------------------------------------------------------------------- 目录


def resolveLogDir() -> Path:
    """
    选一个确实可写的日志目录。

    顺序：程序目录/logs -> %LOCALAPPDATA%/faster-whisper-GUI/logs -> 临时目录。
    用真实写文件来验证，而不是靠 os.access（在 Windows 上不可靠）。
    """
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    candidates = [
        BASE_DIR / LOG_DIR_NAME,
        Path(local) / "faster-whisper-GUI" / LOG_DIR_NAME,
        Path(tempfile.gettempdir()) / "faster-whisper-GUI" / LOG_DIR_NAME,
    ]
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return candidate
        except Exception as exc:                      # noqa: BLE001 - 依次降级
            last_error = exc
            continue
    raise RuntimeError(f"找不到可写的日志目录: {last_error}")


def pruneOldRuns(log_dir: Path, keep: int = KEEP_RUNS) -> List[Path]:
    """
    只保留最近 keep 次运行，其余删除。

    只删自己生成的 app-*.log / fw-*.log，latest.log 与其它文件一律不动，
    避免误删用户放进来的东西。
    """
    removed: List[Path] = []
    try:
        runs = [p for p in log_dir.iterdir()
                if p.is_file() and (p.name.startswith(RUN_PREFIX)
                                    or p.name.startswith(FW_PREFIX))
                and p.suffix == ".log"]
    except OSError:
        return removed

    # 按「运行时间戳」分组：同一个 stamp 的 app-*.log 与 fw-*.log 算一次运行
    stamps = sorted({p.stem.split("-", 1)[1] for p in runs}, reverse=True)
    for stamp in stamps[keep:]:
        for path in runs:
            if path.stem.split("-", 1)[1] == stamp:
                try:
                    path.unlink()
                    removed.append(path)
                except OSError:
                    pass
    return removed


# --------------------------------------------------------------------------- 输出流


class LoggingStream:
    """
    把 write() 变成 logging 记录的文件对象。

    为什么在「流」这一层做：print("a", b) 实际上会调用 write("a")、write(" ")、
    write("b")、write("\\n")，第三方库也是直接 write。所以只要在这里按行缓冲、
    整行发一条记录，就能天然正确处理多参数、sep、end，以及非 print 的输出 ——
    调用点一行都不用动。

    绝不能把 Handler 指回 sys.stdout，否则 write -> log -> Handler -> write 无限递归。
    控制台 Handler 必须写 sys.__stderr__，文件与 Qt 信号出口保持独立。
    """

    def __init__(self, level: int = logging.INFO,
                 fallback_name: str = "faster_whisper_GUI.output") -> None:
        self._level = level
        self._fallback_name = fallback_name
        self._buffer = ""
        self._lock = threading.RLock()

    # --- 文件对象协议 -----------------------------------------------------

    def write(self, text) -> int:
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            self._buffer += text
            self._drain()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""

    def fileno(self):
        # 没有真实文件描述符；返回 -1 让调用方知道不支持（cpython 的约定）
        return -1

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"

    def close(self) -> None:
        self.flush()

    # --- 内部 ------------------------------------------------------------

    def _drain(self) -> None:
        """把缓冲区里已完整的所有行发出去。"""
        while True:
            idx = -1
            for ch in ("\n", "\r"):
                pos = self._buffer.find(ch)
                if pos != -1 and (idx == -1 or pos < idx):
                    idx = pos
            if idx == -1:
                return
            line = self._buffer[:idx]
            rest = self._buffer[idx + 1:]
            # \r\n 视作一个换行符，不要多产生一条空记录
            if self._buffer[idx] == "\r" and rest.startswith("\n"):
                rest = rest[1:]
            self._buffer = rest
            self._emit(line)

    def _emit(self, line: str) -> None:
        # 空行不写日志（原始 print() 常见），保留可读性
        if not line.strip():
            return
        logging.getLogger(self._callerName()).log(self._level, "%s", line)

    def _callerName(self) -> str:
        """从调用栈里取发起输出的模块名，让日志能标出是谁打的。"""
        try:
            frame = sys._getframe(2)
        except ValueError:
            return self._fallback_name
        while frame is not None:
            name = frame.f_globals.get("__name__", "")
            if name and not name.startswith(__name__):
                return name
            frame = frame.f_back
        return self._fallback_name


class CallbackHandler(logging.Handler):
    """
    把日志记录转发给任意回调（界面文本框）。

    只送 message，不带时间戳/级别 —— 界面保持原来的观感。
    回调可能在任意线程被调用；调用方自己负责线程安全（Qt 用信号排队即可）。
    """

    def __init__(self, callback, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._callback = callback
        self.setFormatter(logging.Formatter(GUI_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(self.format(record) + "\n")
        except Exception:                              # noqa: BLE001 - 界面永远不能
            self.handleError(record)                   # 反过来把日志搞崩


# --------------------------------------------------------------------------- 配置


def setupLogging(level: int = logging.DEBUG,
                 gui_level: int = logging.INFO,
                 framework_log: bool = True) -> Path:
    """
    一次性配置根 logger。返回本次运行的日志文件路径。

    级别约定（写代码时请遵守，否则日志会退化成噪声）：
      DEBUG    库内部细节、DLL 注册、参数快照
      INFO     阶段标记、每一段转写结果
      WARNING  可恢复的异常（单文件失败、退回 CPU、配置节缺失）
      ERROR    用户可见的失败（写文件失败、模型加载失败、worker 崩溃）
      CRITICAL 未捕获异常、进程级故障
    """
    global _configured, _log_dir, _run_log_path, _latest_log_path
    global _fw_log_path, _file_handlers

    root = logging.getLogger()

    # 进程内只配置一次（见 _GUARD_ATTR 的说明）。重复调用必须直接返回，
    # 绝不能重新 open(mode="w") —— 那会把本次运行已经写好的日志清空。
    existing = getattr(root, _GUARD_ATTR, None)
    if existing is not None:
        _log_dir, _run_log_path, _latest_log_path, _fw_log_path = existing
        _configured = True
        return _run_log_path

    if _configured and _run_log_path is not None:
        return _run_log_path

    logging.raiseExceptions = False   # logging 自身出错时不要再往 stderr 喷

    log_dir = resolveLogDir()
    _log_dir = log_dir
    pruneOldRuns(log_dir)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}"
    _run_log_path = log_dir / f"{RUN_PREFIX}{stamp}.log"
    _latest_log_path = log_dir / LATEST_NAME

    formatter = logging.Formatter(FILE_FORMAT, datefmt=FILE_DATEFMT)

    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # 本次运行的文件（永不覆盖）+ 固定路径的最新副本（便于让用户发一份）
    for path in (_run_log_path, _latest_log_path):
        try:
            handler = logging.FileHandler(path, mode="w", encoding="utf-8",
                                          delay=False)
        except OSError:
            if path == _run_log_path:
                raise
            continue
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _file_handlers.append(handler)

    # 使用原始流，避免 installOutputStreams() 接管后形成日志递归。
    # pythonw / 无控制台启动时 __stderr__ 可能为 None，此时仅写文件与界面。
    console_handler = None
    if sys.__stderr__ is not None:
        console_handler = logging.StreamHandler(sys.__stderr__)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # 第三方库：根 logger 放 DEBUG 会把 torch/PIL 的噪声全吸进来
    root.setLevel(logging.INFO)
    logging.getLogger("faster_whisper_GUI").setLevel(level)
    logging.getLogger("Qt").setLevel(logging.DEBUG)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # faster-whisper 库自身的日志：单独一个文件，便于崩溃时对照
    if framework_log:
        try:
            _fw_log_path = log_dir / f"{FW_PREFIX}{stamp}.log"
            fw_handler = logging.FileHandler(_fw_log_path, mode="w",
                                             encoding="utf-8", delay=False)
            fw_handler.setLevel(logging.DEBUG)
            fw_handler.setFormatter(formatter)
            fw_logger = logging.getLogger("faster_whisper")
            fw_logger.setLevel(logging.DEBUG)
            fw_logger.propagate = False        # 不和 app 日志重复
            for handler in list(fw_logger.handlers):
                fw_logger.removeHandler(handler)
            fw_logger.addHandler(fw_handler)
            if console_handler is not None:
                fw_logger.addHandler(console_handler)
            _file_handlers.append(fw_handler)
        except OSError:
            _fw_log_path = None

    _configured = True

    # 把状态挂到 root logger，第二份模块实例也能看到（见 _GUARD_ATTR）
    setattr(root, _GUARD_ATTR,
            (_log_dir, _run_log_path, _latest_log_path, _fw_log_path))

    logging.getLogger(__name__).info(
        "日志已启用: %s (保留最近 %d 次运行)", _run_log_path, KEEP_RUNS)
    atexit.register(shutdownLogging)
    return _run_log_path


def installOutputStreams(level: int = logging.INFO) -> None:
    """
    把 sys.stdout / sys.stderr 换成写日志的流（幂等）。

    换掉之后，任何 print()（包括第三方库直接写 stdout/stderr）都会进入日志管道，
    调用点一行都不用改。两个流都记 INFO：真正的崩溃由异常钩子以 CRITICAL 记录，
    靠模块名就能区分是谁写的，不必用级别去猜。
    """
    if not isinstance(sys.stdout, LoggingStream):
        sys.stdout = LoggingStream(level=level,
                                   fallback_name="faster_whisper_GUI.stdout")
    if not isinstance(sys.stderr, LoggingStream):
        sys.stderr = LoggingStream(level=level,
                                   fallback_name="faster_whisper_GUI.stderr")


def attachGuiHandler(callback, level: int = logging.INFO) -> CallbackHandler:
    """把界面文本框接进同一个日志管道。返回的 handler 可用于 detachHandler()。"""
    handler = CallbackHandler(callback, level=level)
    logging.getLogger().addHandler(handler)
    return handler


def detachHandler(handler: Optional[logging.Handler]) -> None:
    """把之前挂上的 handler 摘掉（重复挂接时先摘再挂）。"""
    if handler is None:
        return
    try:
        logging.getLogger().removeHandler(handler)
        handler.close()
    except Exception:                                  # noqa: BLE001
        pass


def shutdownLogging() -> None:
    """刷新并关闭文件 Handler（幂等）。"""
    global _configured
    for handler in list(_file_handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:                              # noqa: BLE001
            pass
        logging.getLogger().removeHandler(handler)
    _file_handlers.clear()


# --------------------------------------------------------------------------- 钩子


def installExceptionHooks() -> None:
    """
    接住所有「没人管」的异常。

    原实现没有装任何钩子：主线程未捕获异常会经 sys.stderr（一个 Qt 信号对象）
    无声消失；子线程异常走 threading 默认钩子，同样未必进得了日志；
    Qt 自己捕获的 QThread.run() 异常只打印一行、不带 traceback。
    """
    crash_log = logging.getLogger("faster_whisper_GUI.crash")

    previous_hook = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_tb)
            return
        crash_log.critical("未捕获异常（主线程）", exc_info=(exc_type, exc_value, exc_tb))
        previous_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _thread_hook(args) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        thread = getattr(args, "thread", None)
        name = getattr(thread, "name", "?") if thread else "?"
        crash_log.critical("未捕获异常（线程 %s）", name,
                           exc_info=(args.exc_type, args.exc_value,
                                     args.exc_traceback))

    threading.excepthook = _thread_hook


def installQtMessageHandler() -> None:
    """
    把 Qt 自己的消息收进日志。

    这条能捞回 "Error calling Python override of QThread::run():" —— 它是 Qt 用
    qWarning 输出的，既不经过 Python 的 excepthook，也不经过被重定向的 sys.stderr。
    """
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:                                  # noqa: BLE001 - 无 Qt 时跳过
        return

    level_of = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    qt_log = logging.getLogger("Qt")

    def _handler(mode, context, message) -> None:
        qt_log.log(level_of.get(mode, logging.INFO), "%s", message)

    qInstallMessageHandler(_handler)


# --------------------------------------------------------------------------- 诊断


def environmentSnapshot() -> str:
    """
    运行环境快照。

    这段应该写在日志开头：用户发一份日志过来，就能省掉来回追问环境。
    FasterWhisperGUI.py 里那段 ROCm 诊断块（状态/根因/影响/排查）是本项目
    写得最好的日志，这里沿用它「把排查需要的东西一次说清」的思路。
    """
    lines: List[str] = []

    def add(title: str, value) -> None:
        lines.append(f"  {title:<26}: {value}")

    lines.append("---------- 运行环境 ----------")
    add("时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    add("Python", sys.version.split()[0])
    add("解释器", sys.executable)
    add("工作目录", os.getcwd())
    add("程序目录", str(BASE_DIR))
    add("日志目录", str(_log_dir) if _log_dir else "(未初始化)")
    add("平台", sys.platform)

    for package in ("faster-whisper", "ctranslate2", "torch", "PySide6",
                    "PySide6-Fluent-Widgets", "numpy", "av", "opencc",
                    "huggingface-hub"):
        try:
            add(package, _pkg_version(package))
        except PackageNotFoundError:
            add(package, "(未安装)")
        except Exception as exc:                       # noqa: BLE001
            add(package, f"(读取失败: {exc})")

    # 内存
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = status.ullTotalPhys / (1024 ** 3)
            avail = status.ullAvailPhys / (1024 ** 3)
            add("物理内存", f"{avail:.1f} GB 可用 / {total:.1f} GB 共 "
                            f"({status.dwMemoryLoad}% 已用)")
    except Exception:                                  # noqa: BLE001 - 非 Windows
        pass

    # GPU / ROCm（只读环境变量，不导入 ctranslate2，避免加载 DLL）
    for key in ("ROCM_PATH", "HIP_PATH", "ROCBLAS_USE_HIPBLASLT",
                "HSA_OVERRIDE_GFX_VERSION"):
        if os.environ.get(key):
            add(key, os.environ[key])

    lines.append("------------------------------")
    return "\n".join(lines)


def logEnvironmentSnapshot() -> None:
    """把环境快照写进日志（每次运行一次）。"""
    for line in environmentSnapshot().splitlines():
        logging.getLogger("faster_whisper_GUI.env").info("%s", line)


def exportDiagnostics(destination: Optional[Path] = None) -> Path:
    """
    打一个诊断包，用户直接发给维护者即可。

    内容：本次运行日志、faster-whisper 日志、环境快照。
    **不含配置文件**（里面有 HuggingFace token），需要的话另行说明。
    """
    if _log_dir is None:
        raise RuntimeError("日志尚未初始化")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}"
    destination = destination or (_log_dir / f"diagnostics-{stamp}.zip")

    import zipfile

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in (_run_log_path, _fw_log_path,
                     _log_dir / LATEST_NAME):
            if path and path.exists():
                bundle.write(path, arcname=path.name)
        bundle.writestr("environment.txt", environmentSnapshot())
    return destination
