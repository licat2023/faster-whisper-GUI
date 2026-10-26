# coding:utf-8

import logging
from PySide6.QtCore import QTranslator
from resource import rc_Translater
import locale

from faster_whisper_GUI.runtime.preferences import ConfigError, load_config

log = logging.getLogger(__name__)

try:
    language_config = load_config().get('settings', {}).get('language', 2)
except ConfigError:
    language_config = 2

if language_config == 0:
    language = "zh"
elif language_config == 1:
    language = "en"
else:
    # 获取当前计算机语言
    language_localtion, _ = locale.getlocale()
    language = (language_localtion or "en").split("_")[0]
    log.info("%s", f"current computer language region-format: {language_localtion}")

log.info("%s", f"language: {language}")

def __translator() -> QTranslator:
    translator = QTranslator()
    if language != "zh" :  
        try:
            if not translator.load(":/resource/Translater/en.qm"):
                log.warning("英文翻译资源加载失败")
            # splash.showMessage("set Language: English") #, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, Qt.white)
        except Exception as e:
            log.error("%s", f"load translator files error: {str(e)}")
            translator.load("")
    else:
        translator.load("")

    return translator

TRANSLATOR = __translator()
