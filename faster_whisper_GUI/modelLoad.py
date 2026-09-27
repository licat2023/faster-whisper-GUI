# coding:utf-8

# from threading import Thread
from typing import (List, Optional, TypedDict, Union)
from PySide6.QtCore import QThread, Signal
from faster_whisper import WhisperModel


class modelParamDict(TypedDict):
    model_size_or_path: str
    device: str
    device_index: Union[int, List[int]] 
    compute_type: str 
    cpu_threads: int 
    num_workers: int 
    download_root: Optional[str] 
    local_files_only: bool 

class LoadModelWorker(QThread):
    setStatusSignal = Signal(bool)
    loadModelOverSignal = Signal(bool)

    def __init__(self, modelParam: modelParamDict, use_v3_model:bool = False , parent = None):
        super().__init__(parent=parent)
        self.isRunning = False
        self.model_size_or_path: str = modelParam["model_size_or_path"]
        self.device: str = modelParam["device"]
        self.device_index: Union[int, List[int]] = modelParam["device_index"]
        self.compute_type: str = modelParam["compute_type"]
        self.cpu_threads: int = modelParam["cpu_threads"]
        self.num_workers: int = modelParam["num_workers"]
        self.download_root: Optional[str] = modelParam["download_root"]
        self.local_files_only: bool = modelParam["local_files_only"]
        self.use_v3_model :bool = use_v3_model

        self.model = None
        
    def run(self) -> None:
        self.isRunning = True

        self.model = self.loadModel()
        # with futures.ThreadPoolExecutor() as executor:
        #     # 提交任务
        #     # futures_ = [executor.submit(self.loadModel)]

        #     results = executor.map(self.loadModel,[self.model_size_or_path])

        #     for result in results:
        #         self.model = result

            # for future in futures.as_completed(results):
            #     pass
                # self.model = future.result()
        
        if self.use_v3_model:
            # 修正 V3 模型的 mel 滤波器组参数
            print("\n[Using V3 model, modify  number of mel-filters to 128]")
            #
            # 注意 .astype("float32") 不能省。
            # get_mel_filters() 内部用 numpy 默认精度计算，返回的是 float64；而 FeatureExtractor
            # 的构造函数里是 "get_mel_filters(...).astype('float32')"，即官方途径会转成 float32。
            # 直接赋值会让 mel_filters 变成 float64，进而使 log_mel_spectrogram 产出的特征也是
            # float64，最终在 ctranslate2.StorageView.from_array() 处报：
            #     ValueError: Unsupported type: <f8
            # 该错误只在 use_v3_model=True（V3 模型）时出现，与设备（CPU/GPU）无关。
            _feature_extractor = self.model.feature_extractor
            self.model.feature_extractor.mel_filters = _feature_extractor.get_mel_filters(
                _feature_extractor.sampling_rate,
                _feature_extractor.n_fft,
                n_mels=128,
            ).astype("float32")

        self.isRunning = False
        # return self.model

    def stop(self):
        self.isRunning = False
    
    def loadModel(self, model_size_or_path:str=None):
        
        model = None
        try:
            if model_size_or_path is None:
                model_size_or_path = self.model_size_or_path

            # 尝试替换空格，以处理带有空格的路径
            # model_size_or_path = model_size_or_path.replace("\\", "/")
            # model_size_or_path = model_size_or_path.replace(" ", "\ ")

            # self.download_root = self.download_root.replace("\\", "/")
            # self.download_root = self.download_root.replace(" ", "\ ")

            model = WhisperModel(
                                    model_size_or_path, 
                                    device=self.device, 
                                    device_index=self.device_index,
                                    compute_type=self.compute_type,
                                    cpu_threads=self.cpu_threads,
                                    num_workers=self.num_workers,
                                    download_root=self.download_root,
                                    local_files_only=self.local_files_only
                                )
        except Exception as e:
            model = None
            self.setStatusSignal.emit(False)
            raise e

        try:
            print("\nLoad over")
            print(self.model_size_or_path)
            print(f"{'max_length: ':23}",model.max_length)
            print(f"{'num_samples_per_token: ':23}", model.num_samples_per_token)
            print("time_precision: ", model.time_precision)
            print("tokens_per_second: ", model.tokens_per_second)
            print("input_stride: ", model.input_stride)

        except Exception as e:
            
            self.setStatusSignal.emit(False)
            
            raise e
        
        self.setStatusSignal.emit(True)

        return model

