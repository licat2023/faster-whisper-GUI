"""可选 Qwen3-ASR 模型适配；依赖与模型加载由独立环境管理。"""

from pathlib import Path
from faster_whisper_GUI.config import Language_dict
from .base import Recognition, TranscriptionInfo


class QwenBackend:
    sample_rate = 16000

    def __init__(self, model, align=False):
        self.model = model
        self.align = align

    def recognize(self, audio, options):
        supported = {"language", "task", "word_timestamps", "max_segment_chars", "max_segment_seconds"}
        if set(options) - supported or options.get("task", "transcribe") != "transcribe":
            raise ValueError("Qwen 实验适配仅支持识别及语言选择")
        if options.get("word_timestamps") and not self.align:
            raise ValueError("Qwen ASR 需要单独的 ForcedAligner 才能返回词时间戳")
        if isinstance(audio, (str, Path)):
            import soundfile as sf
            duration = sf.info(audio).duration
            source = str(audio)
        else:
            import numpy as np
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim != 1 or not np.isfinite(samples).all():
                raise ValueError("音频数组必须为有限的单声道样本")
            duration = len(samples) / self.sample_rate
            source = (samples, self.sample_rate)
        language = options.get("language")
        if language:
            language = "Chinese" if language in {"zh", "zhs", "zht"} else Language_dict.get(language, language).strip().title()
        kwargs = {'return_time_stamps': True} if self.align else {}
        output = self.model.transcribe(audio=source, language=language, **kwargs)[0]
        detected = ("zh" if output.language.lower() == "chinese" else
                    next((code for code, name in Language_dict.items()
                          if name.strip().lower() == output.language.lower()), output.language.lower()))
        info = TranscriptionInfo(detected, None, duration, None)
        if self.align:
            from .aligned import aligned_segments
            segments = aligned_segments(output.time_stamps or [], output.text, duration,
                int(options.get('max_segment_chars', 32)), float(options.get('max_segment_seconds', 6)))
            return Recognition(segments, info, text=output.text, segment_timestamps=True)
        return Recognition([], info, text=output.text, segment_timestamps=False)
