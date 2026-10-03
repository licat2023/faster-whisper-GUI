"""separation 任务实现。"""

from faster_whisper_GUI.runtime.inference import prepare_inference_runtime

prepare_inference_runtime()

import logging
# coding:utf-8
import os
from PySide6.QtCore import QThread, Signal

from faster_whisper_GUI.tasks.base import GuardedWorker
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import torch
from torchaudio.transforms import Fade
import av
import soundfile
import numpy as np
import gc

from faster_whisper import decode_audio

from faster_whisper_GUI.config import STEMS

log = logging.getLogger(__name__)


class DemucsWorker(GuardedWorker):

    signal_vr_over = Signal(bool)
    file_process_status = Signal(dict)

    def __init__(
                    self, parent, 
                    audio:list[str],
                    stems:int,
                    model_path:str,
                    *,
                    segment:float=10,
                    overlap:float=0.1,
                    sample_rate:int=44100,
                    output_path:str=""
                    ) -> None:
        
        super().__init__(parent)
        self.is_running = False
        self.model_path = model_path
        self.model = None
        self.audio = audio
        self.sampleRate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.stems = stems
        self.output_path = output_path


    def run(self) -> None:
        self.is_running = True
        try:
            if not self.audio:
                raise ValueError('没有选择有效的音频文件')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.file_process_status.emit({'file': '', 'status': False, 'task': 'load model'})
            if self.model is None:
                self.loadModel(self.model_path, device=device)
            for audio in self.audio:
                if not self.is_running:
                    self.signal_vr_over.emit(False)
                    return
                self.file_process_status.emit({'file': audio, 'status': False, 'task': 'reasmple audio'})
                samples = np.asarray(self.reSampleAudio(audio, self.sampleRate, device=device))
                if samples.ndim == 1:
                    samples = np.stack([samples, samples])
                if not self.is_running:
                    self.signal_vr_over.emit(False)
                    return
                self.file_process_status.emit({'file': audio, 'status': False, 'task': 'separate sources'})
                sources = self.separate_sources(self.model, samples[None], self.segment, self.overlap, device, self.sampleRate)
                if not self.is_running:
                    self.signal_vr_over.emit(False)
                    return
                self.file_process_status.emit({'file': audio, 'status': False, 'task': 'save files'})
                self.saveResult(model=self.model, file_path=audio, sources=sources, stems=self.stems,
                                output_path=self.output_path, sample_rate=self.sampleRate)
                self.file_process_status.emit({'file': audio, 'status': True, 'task': 'file over'})
            self.signal_vr_over.emit(True)
        finally:
            self.is_running = False
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def onError(self, exc):
        self.signal_vr_over.emit(False)

    def stop(self):
        self.is_running = False

    def separate_sources(self,
                        model,
                        mix,
                        segment=10.0,
                        overlap=0.1,
                        device=None,
                        sample_rate=44100,
                    ):
        """
        Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

        Args:
            segment (int): segment length in seconds
            device (torch.device, str, or None): if provided, device on which to
                execute the computation, otherwise `mix.device` is assumed.
                When `device` is different from `mix.device`, only local computations will
                be on `device`, while the entire tracks will be stored on `mix.device`.
        """
        if segment <= 0 or not 0 <= overlap <= 1 or sample_rate <= 0:
            raise ValueError('分段长度、采样率必须大于 0，重叠度必须在 0..1 之间')
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        batch, channels, length = mix.shape
        step = max(1, int(sample_rate * segment))
        overlap_frames = int(step * overlap)
        chunk_len = step + overlap_frames
        final = torch.zeros(batch, len(model.sources), channels, length, device=device)
        weights = torch.zeros(length, device=device)
        for start in range(0, length, step):
            if not self.is_running:
                return None
            end = min(start + chunk_len, length)
            chunk = torch.as_tensor(mix[:, :, start:end], dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model.forward(chunk)
            if not self.is_running:
                return None
            weight = torch.ones(end - start, device=device)
            fade_size = min(overlap_frames, end - start)
            if fade_size:
                ramp = torch.linspace(0, 1, fade_size + 2, device=device)[1:-1]
                if start:
                    weight[:fade_size] *= ramp
                if end < length:
                    weight[-fade_size:] *= ramp.flip(0)
            final[:, :, :, start:end] += out * weight
            weights[start:end] += weight
            if end == length:
                break
        return final / weights.clamp_min(1e-8)

    def loadModel(self, model_path:str, device=None):

        download_path = os.path.abspath(model_path)
        log.info("%s", f"download_path: {download_path}")

        if os.path.exists(download_path):
            log.info("%s", "found existed model file")
        else:
            self.file_process_status.emit({"file":"","status":False,"task":"download model"})

        import copy
        bundle = copy.copy(HDEMUCS_HIGH_MUSDB_PLUS)
        bundle._model_path = download_path
        bundle._sample_rate = 44100

        sample_rate = bundle.sample_rate
        log.info("%s", f"Sample rate: {sample_rate}")

        if not self.is_running:
            return

        self.model = bundle.get_model()

        if not self.is_running:
            return
        
        del bundle
        self.model.to(device)

    def reSampleAudio(self, audio, sample_rate, device) -> np.ndarray:
        file_path = os.path.abspath(audio)

        split_setore = True
        with av.open(file_path) as av_file:
            stream_ = next(s for s in av_file.streams if s.codec_context.type == 'audio')
            audio_channel_num = stream_.channels
            if audio_channel_num < 2:
                log.info("%s", "single-channel audio")
                split_setore = False
            else:
                log.info("%s", "multi-channel audio")

        log.info("%s", "resample audio data")
        samples = decode_audio(file_path, sample_rate, split_setore)
        # samples = np.array(samples)

        # samples_t = torch.tensor(samples,dtype=torch.float32).to(device)
        # del samples
        return samples
    
    def saveResult(self, model, file_path:str, sources:torch.Tensor, stems:int, output_path:str, sample_rate=44100):

        sources_list = model.sources
        log.info("%s", f"sources_list: {sources_list}")

        # 将不同输出音轨排列成为列表形式存储 元素为 Tensor
        sources = list(sources[0])

        # 将输出音轨整合为 字典 形式存储
        audios:dict = dict(zip(sources_list, sources))

        # 获取文件名、文件路径
        data_dir,file_name = os.path.split(file_path)
        file_output = file_name.split(".")
        file_output = ".".join(file_output[:-1])

        # 根据用户选择的不同输出音轨，进行相应处理
        if stems == 0:
            stems = STEMS[1:-1]
        
        elif stems != (len(STEMS) - 1):
            stems = [STEMS[stems]]

        # 人声、背景音乐二分输出需要进行音频内容整合
        else:
            stems = ["Vocals", "Others"]
            audios["others"] = audios["other"] 
            audios.pop("other")
            audios["others"] = audios["others"] + audios["bass"] 
            audios.pop("bass")
            audios["others"] = audios["others"] + audios["drums"]
            audios.pop("drums")
        
        log.info("%s", f"output stems: {stems}")

        if not output_path:
            output_path = os.path.join(data_dir, file_output)
        else:
            output_path = os.path.join(output_path, file_output)

        if not os.path.exists(output_path):
            log.info("%s", f"create output folder: {output_path}")
            os.makedirs(output_path, exist_ok=True)

        for stem in stems:
            spec = audios[stem.lower()][:, :].cpu()
            output_path_ = output_path
            # if not os.path.exists(output_path_):
            #     os.makedirs(output_path_)
            
            output_fileName = os.path.join(output_path_, ".".join([file_output+f"_{stem.lower()}", "wav"]))
            log.info("%s", f"save file: {output_fileName}")

            soundfile.write(output_fileName, spec.numpy().T,  sample_rate)
