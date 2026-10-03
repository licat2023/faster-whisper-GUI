"""Subtitle groups from native streaming token emission times (no word end times)."""
import math
from faster_whisper_GUI.domain.segments import segment_Transcribe


def token_segments(tokens, timestamps, duration, max_chars=32, max_seconds=6):
    if max_chars < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('字幕长度与时长必须为正数')
    if len(tokens) != len(timestamps):
        raise ValueError('流式文字与时间戳数量不匹配')
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('音频时长必须为非负有限数')
    groups, text, start = [], '', None
    previous = 0
    for token, timestamp in zip(tokens, timestamps):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < previous or timestamp < 0 or timestamp > duration + .1:
            raise ValueError('流式模型返回无效时间戳')
        previous = timestamp
        timestamp = min(timestamp, duration)
        token = token.replace('▁', ' ')
        if text and (len(text) + len(token) > max_chars or timestamp - start > max_seconds):
            groups.append(segment_Transcribe(start=start, end=timestamp, text=text.strip()))
            text, start = '', None
        if start is None:
            start = timestamp
        text += token
    if text:
        groups.append(segment_Transcribe(start=start, end=max(start, duration), text=text.strip()))
    return groups
