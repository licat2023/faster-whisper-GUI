"""原生流式接口：明确区分可修订的文字与有时间戳的字幕片段。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StreamUpdate:
    text: str
    revision: int
    final: bool
    audio_seconds: float


class StreamingSession(Protocol):
    def accept(self, samples) -> StreamUpdate: ...
    def finish(self) -> StreamUpdate: ...


class StreamingBackend(Protocol):
    sample_rate: int
    def start(self) -> StreamingSession: ...
