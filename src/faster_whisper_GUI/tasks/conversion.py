"""conversion 任务实现。"""

from faster_whisper_GUI.runtime.inference import prepare_inference_runtime

prepare_inference_runtime()

# coding:utf-8

import logging
import os 
import time
from threading import Thread

from ctranslate2.converters import TransformersConverter as cvter

from PySide6.QtCore import QCoreApplication

from faster_whisper_GUI.config import Model_names

log = logging.getLogger(__name__)

def __tr(text:str) -> str:
    return QCoreApplication.translate("ConvertModel", text)

def ConvertModel(model_name_or_path:str,cache_dir: str, output_dir:str, quantization:str, use_local_file : bool = True):

    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    if os.path.isdir(model_name_or_path):
        source = model_name_or_path
    elif model_name_or_path in Model_names:
        source = 'openai/whisper-' + model_name_or_path
        if use_local_file:
            snapshots = os.path.join(cache_dir, 'models--openai--whisper-' + model_name_or_path.replace('.', '-'), 'snapshots')
            if os.path.isdir(snapshots):
                revisions = [os.path.join(snapshots, name) for name in os.listdir(snapshots)
                             if os.path.isdir(os.path.join(snapshots, name))]
                if revisions:
                    source = max(revisions, key=os.path.getmtime)
    else:
        raise ValueError(f'无效模型名称或目录：{model_name_or_path}')
    def wait_for(future):
        while True:
            try:
                return future.result(timeout=.5)
            except TimeoutError:
                print('.', end='', flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        converter = wait_for(executor.submit(cvter, model_name_or_path=source, copy_files=['tokenizer.json']))
        wait_for(executor.submit(converter.convert, output_dir=output_dir, quantization=quantization, force=True))
    log.info('%s', __tr('\nOver'))
