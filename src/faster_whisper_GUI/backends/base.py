"""文件与分块识别的共同接口；原生流式会话需独立的事件协议。"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from faster_whisper_GUI.domain.segments import segment_Transcribe


@dataclass(frozen=True)
class TranscriptionInfo:
    language: str
    language_probability: float | None
    duration: float
    duration_after_vad: float | None


@dataclass
class Recognition:
    # 保持惰性迭代，Worker 可在片段之间取消，不预先耗尽推理生成器。
    segments: Iterable[segment_Transcribe]
    info: TranscriptionInfo
    # 不提供时间戳的模型只返回全文，不编造字幕时间轴。
    text: str | None = None
    segment_timestamps: bool = True


class RecognitionBackend(Protocol):
    sample_rate: int

    def recognize(self, audio: Any, options: Mapping[str, Any]) -> Recognition:
        """返回全局偏移为零的秒级时间戳；缺少词时间戳时 words 为空。"""
        ...
