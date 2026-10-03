# coding:utf-8

import sys
import os
import logging
from .runtime.paths import PROJECT_ROOT

BASE_DIR = str(PROJECT_ROOT)



def main() -> None:
    """启动桌面应用，并将运行时文件定位到项目目录。"""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.chdir(BASE_DIR)
    # 修复环境变量 - python 文件夹
    python_dir = os.pathsep + os.path.join(BASE_DIR, 'python')
    os.environ["PATH"] = os.environ.get("PATH", "") + python_dir

    # 修复环境变量 - bin 文件夹
    bin_dir = os.pathsep + os.path.join(BASE_DIR, 'bin')
    os.environ["PATH"] = os.environ.get("PATH", "") + bin_dir

    # 先注册 ROCm DLL 目录，随后建立日志；推理模块会在加载 CT2 前预加载 torch。
    # 包 __init__ 不再隐式加载 whisperx，因此这些正常导入不会抢先初始化 GPU。
    from faster_whisper_GUI.runtime import rocm as _rocm_util

    _ROCm_READY = _rocm_util.setupROCm()

    from faster_whisper_GUI import logging_setup

    try:
        LOG_PATH = logging_setup.setupLogging()
        logging_setup.installExceptionHooks()   # sys.excepthook + threading.excepthook
        logging_setup.installOutputStreams()    # print() 从此进入日志管道
        logging_setup.logEnvironmentSnapshot()
    except BaseException:
        # 日志系统自己起不来时必须让人看见 —— 否则「为什么没有日志」会变成新的谜题
        import traceback as _tb
        _detail = _tb.format_exc()
        try:
            with open(os.path.join(BASE_DIR, "logging_setup_error.txt"), "w",
                      encoding="utf-8") as _fh:
                _fh.write(_detail)
        except OSError:
            pass
        if sys.__stderr__ is not None:
            sys.__stderr__.write(_detail)
            sys.__stderr__.flush()
        raise

    # 本模块的 logger。必须在下面的 print 之前定义 —— 转换脚本只保证「在最后一个
    # 顶层 import 之后」，而这里后续还有 import，所以显式放在引导块正下方。
    log = logging.getLogger(__name__)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import (QFont, QPixmap)
    from PySide6.QtWidgets import (QApplication, QSplashScreen, QVBoxLayout)

    from qfluentwidgets import ProgressBar

    logging_setup.installQtMessageHandler()   # Qt 自己的消息也进日志

    class MySplashScreen(QSplashScreen):
        # 鼠标点击事件
        def mousePressEvent(self, event):
            pass

    from resource import rc_Image

    # 启动一个Qt程序，并使用传入的系统参数
    app = QApplication(sys.argv)
    app.setObjectName("FasterWhisperGUIAPP")

    #设置启动界面
    splash = MySplashScreen()

    #初始图片
    splash.setPixmap(QPixmap(r":/resource/Image/SplashScreen_0.4.0.png")) 

    # 设置字体
    splash.setFont(QFont('Segoe UI', 15))

    #初始文本
    splash.showMessage("Loading...", Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom, Qt.white)

    # splash.setStyleSheet("MySplashScreen{border-radius: 10px;}")

    ly = QVBoxLayout(splash)
    splash.setLayout(ly)

    pb = ProgressBar(splash)
    pb.setMaximum(100)
    pb.setMinimum(0)

    ly.addWidget(pb,alignment=Qt.AlignmentFlag.AlignBottom)
    ly.addSpacing(20)

    # 显示启动界面
    splash.show()

    app.processEvents()  # 处理主进程事件

    # ---------------------------------------------------------------------------------------------------------------------------
    # 记录 ROCm 探测结果
    #
    # setupROCm() 在日志引导之前就执行了（时序要求，不能推迟），所以它的输出没能进日志。
    # 这里补记状态，方便排查"下拉框里为什么没有 AMD ROCm 选项"这类问题。
    #
    # 这段是本项目日志的范本：状态 / 根因 / 影响 / 排查 —— 一条日志把该说的说全，
    # 用户发一份日志过来就够，不用来回追问环境。
    _rocm_log = logging.getLogger("faster_whisper_GUI.rocm")
    _rocm_log.info("---------- AMD ROCm / HIP ----------")
    if _ROCm_READY:
        _rocm_log.info("状态      : 已启用")
        _rocm_log.info("根目录    : %s", _rocm_util.ROCM_ROOT)
        for _directory in _rocm_util.ROCM_DLL_DIRECTORIES:
            _rocm_log.info("DLL 目录  : %s", _directory)
        _rocm_log.info("ROCBLAS_USE_HIPBLASLT  : %s",
                       os.environ.get('ROCBLAS_USE_HIPBLASLT'))
        _rocm_log.info("HSA_OVERRIDE_GFX_VERSION: %s",
                       os.environ.get('HSA_OVERRIDE_GFX_VERSION'))
        _rocm_log.info("设备下拉框将包含 AMD ROCm (HIP) 选项")
    else:
        _rocm_log.info("状态      : 未检测到 HIP SDK")
        _rocm_log.info("原因      : 未找到含 amdhip64_7.dll 的 ROCm 目录")
        _rocm_log.info("影响      : 设备下拉框不含 AMD ROCm (HIP) 选项（CPU / NVIDIA 不受影响）")
        _rocm_log.info("排查      : 确认已安装 AMD HIP SDK，或设置环境变量 ROCM_PATH")
    _rocm_log.info("-----------------------------------")

    from faster_whisper_GUI.version import __version__
    from faster_whisper_GUI.runtime.diagnostics import outputWithDateTime

    log.info("%s", f"faster_whisper_GUI: {__version__}")

    outputWithDateTime("Start")

    pb.setValue(10)

    from faster_whisper_GUI.ui.window.main import MainWindows

    pb.setValue(60)

    from resource import rc_Translater
    from faster_whisper_GUI.ui.translation import TRANSLATOR, language

    # 主程序入口

    # 修复程序路径依赖

    # 修复环境变量 - ffmpeg
    ffmpeg_dir = os.pathsep + os.path.join(BASE_DIR, 'ffmpeg')
    os.environ["PATH"] = os.environ.get("PATH", "") + ffmpeg_dir

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    pb.setValue(65)

    # 获取当前计算机语言
    # language_localtion, _ = locale.getdefaultlocale()
    # language = language_localtion.split("_")[0]
    # print(f"language: {language_localtion}")

    # 非中文时加载语言翻译文件, 设置英文界面
    translator = TRANSLATOR
    splash.showMessage(
                        "Install translator...", 
                        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, 
                        Qt.white
                    )

    app.installTranslator(translator)

    pb.setValue(70)
    # 注意：这里不再还原/关闭 sys.stderr。
    # 旧实现在此处关闭了日志文件并把 stdout 留在一个已关闭的对象上，
    # 之后任何 print 都会抛 "I/O operation on closed file"。
    # 现在 stdout/stderr 由 logging_setup 的 LoggingStream 一直持有到进程结束。

    # splash.showMessage("Load Windows...") #, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, Qt.white)

    splash.showMessage(
                        "Launching app...", 
                        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, 
                        Qt.white
                    )

    mainWindows = MainWindows()
    pb.setValue(100)

    # splash.requestInterruption()
    # splash.stop(mainWindows)

    splash.finish(mainWindows)
    splash.deleteLater()

    ly.deleteLater()
    pb.deleteLater()

    # 显示窗体
    mainWindows.show()

    # 退出程序，并使用app实例的退出代码
    sys.exit(app.exec())
