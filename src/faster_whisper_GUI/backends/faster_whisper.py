"""封装已加载的 faster-whisper 模型，不在接口模块中导入原生运行时。"""

from faster_whisper_GUI.domain.segments import segment_Transcribe
from .base import Recognition, TranscriptionInfo


class FasterWhisperBackend:
    def __init__(self, model):
        self.model = model

    @property
    def sample_rate(self):
        return self.model.feature_extractor.sampling_rate

    def recognize(self, audio, options):
        if "audio" in options:
            raise ValueError("audio 必须通过音频参数传入，不能重复放入 options")
        segments, native_info = self.model.transcribe(audio=audio, **dict(options))
        info = TranscriptionInfo(
            native_info.language, native_info.language_probability,
            native_info.duration, native_info.duration_after_vad,
        )
        return Recognition((segment_Transcribe(segment) for segment in segments), info)
