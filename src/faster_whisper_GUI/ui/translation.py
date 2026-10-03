# coding:utf-8

import logging
from PySide6.QtCore import QTranslator
from resource import rc_Translater
import locale

import json

log = logging.getLogger(__name__)

try:
    from faster_whisper_GUI.runtime.paths import PROJECT_ROOT
    with (PROJECT_ROOT / "fasterWhisperGUIConfig.json").open(encoding="utf-8") as fp:
        config_json = json.load(fp)
    language_config = config_json["setting"]["language"]
except (OSError, ValueError, KeyError, TypeError):
    log.debug("界面语言配置不可用，使用系统语言", exc_info=True)
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
