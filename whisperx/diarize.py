import inspect
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
import torch

from .audio import load_audio, SAMPLE_RATE

log = logging.getLogger(__name__)


def _load_pipeline(model_name, token, cache_dir):
    """
    加载 pyannote 流水线，兼容令牌参数在 3.x / 4.x 之间的改名。

    pyannote.audio 4.0 起 Pipeline.from_pretrained 的 use_auth_token 被改名为 token，
    传旧名字会直接 TypeError: unexpected keyword argument 'use_auth_token' —— 而本方法
    的调用方（DiarizationPipeline）正好传的是旧名字，于是说话人分离必然失败。
    这里按签名探测，两个版本都能跑。
    """
    kwargs = {"cache_dir": cache_dir}

    if not token:
        # 该仓库在 HuggingFace 上是 gated 的：没有令牌必然拿不到权重。
        # 直接说清楚，比让 pyannote 抛一个难懂的网络/鉴权错误更有用。
        log.warning("未提供 HuggingFace 令牌，%s 是 gated 仓库，加载很可能失败"
                    "（设置页「HuggingFace用户令牌」可填写）", model_name)
        return Pipeline.from_pretrained(model_name, **kwargs)

    parameters = inspect.signature(Pipeline.from_pretrained).parameters
    if "use_auth_token" in parameters and "token" not in parameters:
        # pyannote.audio 3.x：只有旧名字
        return Pipeline.from_pretrained(model_name, use_auth_token=token, **kwargs)

    # pyannote.audio 4.x 用 token；签名里以 **kwargs 透传的版本也走这里，
    # 万一新名字不被接受则回退到旧名字。
    try:
        return Pipeline.from_pretrained(model_name, token=token, **kwargs)
    except TypeError as error:
        if "token" not in str(error):
            raise
        log.debug("pyannote 不接受 token 参数，回退到 use_auth_token", exc_info=True)
        return Pipeline.from_pretrained(model_name, use_auth_token=token, **kwargs)


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization@2.1",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        cache_dir = None
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = _load_pipeline(model_name, use_auth_token, cache_dir)  # .to(device)
        if self.model:
            try:
                self.model = self.model.to(device)
            except Exception as error:
                log.warning("声纹模型搬到 %s 失败，将在原设备上运行: %s", device, error)

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
        diarize_df.rename(columns={2: "speaker"}, inplace=True)
        return diarize_df


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker
        
    return transcript_result            


class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
