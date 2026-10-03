"""任务阶段日志。"""

import logging
import datetime

log = logging.getLogger(__name__)

def outputWithDateTime(text:str):
    dateTime_ = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info("%s", f"\n=========={dateTime_}==========")
    log.info("%s", f"=========={text}==========\n")
