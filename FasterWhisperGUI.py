# coding:utf-8

import sys
import os
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath( __file__))

# 修复环境变量 - python 文件夹
python_dir = ";" + os.path.join(BASE_DIR, 'python')
os.environ["path"] += python_dir

# 修复环境变量 - bin 文件夹
bin_dir = ";" + os.path.join(BASE_DIR, 'bin')
os.environ["path"] += bin_dir

# ---------------------------------------------------------------------------------------------------------------------------
# 启用 AMD ROCm / HIP 运行时（必须放在所有 faster_whisper_GUI 导入之前）
#
# 时序很关键，原因有两条：
#   1. "from faster_whisper_GUI.xxx import ..." 会先执行 faster_whisper_GUI/__init__.py，
#      而它第 1 行就 "import whisperx"，whisperx/asr.py 又会 "import ctranslate2" ——
#      也就是说 ctranslate2.dll 的加载时机远早于脚本后面的代码。
#   2. ROCBLAS_USE_HIPBLASLT 只在 rocBLAS 首次初始化时读取，之后再设无效。
#
# 所以这里用 importlib 直接按文件路径加载 util.py，绕开包 __init__.py，
# 保证在 ctranslate2 被加载之前完成 DLL 目录注册与环境变量设置。
#
# 未检测到 HIP SDK 时会直接跳过，不影响 CPU / NVIDIA 用户。
_ROCm_READY = False
_rocm_spec = importlib.util.spec_from_file_location(
    "faster_whisper_GUI_util_early",
    os.path.join(BASE_DIR, 'faster_whisper_GUI', 'util.py'),
)
if _rocm_spec is not None and _rocm_spec.loader is not None:
    _rocm_util = importlib.util.module_from_spec(_rocm_spec)
    _rocm_spec.loader.exec_module(_rocm_util)
    # 此处 stdout 尚未重定向到日志文件，先只取结果，稍后再写入日志
    _ROCm_READY = _rocm_util.setupROCm()

from PySide6.QtCore import Qt
from PySide6.QtGui import (QFont, QPixmap)
from PySide6.QtWidgets import (QApplication, QSplashScreen, QVBoxLayout)

from qfluentwidgets import ProgressBar

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

# print输出重定向到文件
log_f = open('fasterwhispergui.log', 'w', buffering=1)
sys.stdout = log_f
sys.stderr = log_f

# ---------------------------------------------------------------------------------------------------------------------------
# 记录 ROCm 探测结果
#
# setupROCm() 在上面的日志重定向之前就执行了（时序要求，不能推迟），
# 所以它的输出此刻还进不到日志文件里。这里补记状态，方便排查
# "下拉框里为什么没有 AMD ROCm 选项" 这类问题。
log_f.write("\n---------- AMD ROCm / HIP ----------")
if _ROCm_READY:
    log_f.write(f"\n状态      : 已启用")
    log_f.write(f"\n根目录    : {_rocm_util.ROCM_ROOT}")
    for _directory in _rocm_util.ROCM_DLL_DIRECTORIES:
        log_f.write(f"\nDLL 目录  : {_directory}")
    log_f.write(f"\nROCBLAS_USE_HIPBLASLT  : {os.environ.get('ROCBLAS_USE_HIPBLASLT')}")
    log_f.write(f"\nHSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    log_f.write("\n设备下拉框将包含 AMD ROCm (HIP) 选项")
else:
    log_f.write("\n状态      : 未检测到 HIP SDK")
    log_f.write("\n原因      : 未找到含 amdhip64_7.dll 的 ROCm 目录")
    log_f.write("\n影响      : 设备下拉框不含 AMD ROCm (HIP) 选项（CPU / NVIDIA 不受影响）")
    log_f.write("\n排查      : 确认已安装 AMD HIP SDK，或设置环境变量 ROCM_PATH")
log_f.write("\n-----------------------------------\n")

from faster_whisper_GUI.version import __version__
from faster_whisper_GUI.util import outputWithDateTime

log_f.write(f"\nfaster_whisper_GUI: {__version__}")

outputWithDateTime("Start")

import logging

# faster_whisper 模块日志
logger_faster_whisper = logging.getLogger("faster_whisper")
logger_faster_whisper.setLevel(logging.DEBUG)
faster_whisper_logger_handler = logging.FileHandler(r"./faster_whisper.log", mode="w")
faster_whisper_logger_handler.setLevel(logging.DEBUG)
formatter1 = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",datefmt='%Y-%m-%d_%H:%M:%S')
faster_whisper_logger_handler.setFormatter(formatter1)
logger_faster_whisper.addHandler(faster_whisper_logger_handler)

pb.setValue(10)

from faster_whisper_GUI.mainWindows import MainWindows

pb.setValue(60)

from resource import rc_Translater
from faster_whisper_GUI.translator import TRANSLATOR, language

# 主程序入口
if __name__ == "__main__":
    
    # 修复程序路径依赖
    sys.path.append(os.path.join(BASE_DIR, 'resource'))
    sys.path.append(os.path.join(BASE_DIR, 'faster_whisper_GUI'))
    sys.path.append(os.path.join(BASE_DIR, 'whisperX'))
    sys.path.append(os.path.join(BASE_DIR, 'ffmpeg'))
    sys.path.append(os.path.join(BASE_DIR, 'cache'))
    sys.path.append(os.path.join(BASE_DIR, 'python'))
    sys.path.append(os.path.join(BASE_DIR, 'bin'))
    
    # 修复环境变量 - ffmpeg
    ffmpeg_dir = ";" + os.path.join(BASE_DIR, 'ffmpeg')
    os.environ["path"] += ffmpeg_dir

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
    sys.stderr = sys.__stderr__
    log_f.close()

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


