"""Turn measured alignment spans into editable subtitle segments."""
import math
import re
import unicodedata
from faster_whisper_GUI.domain.segments import Word, segment_Transcribe


def aligned_segments(items, text, duration, max_chars=32, max_seconds=6):
    if max_chars < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('字幕长度与时长必须为正数')
    words, cursor, previous = [], 0, 0.0
    for item in items:
        start, end = float(item.start_time), float(item.end_time)
        if not all(math.isfinite(t) for t in [start, end]) or start < previous or end < start or end > duration + .1:
            raise ValueError('对齐模型返回无效或倒序的时间戳')
        token = item.text
        position = text.find(token, cursor)
        if position < 0:
            match = re.search(re.escape(token), text[cursor:], re.IGNORECASE)
            position = cursor + match.start() if match else -1
        if not token or position < 0 or any(c.isalnum() for c in text[cursor:position]):
            raise ValueError('对齐文字与识别全文不匹配，无法生成可靠字幕')
        boundary = position + len(token)
        while boundary < len(text) and (text[boundary].isspace() or unicodedata.category(text[boundary]).startswith('P')):
            boundary += 1
        token, cursor = text[cursor:boundary], boundary
        words.append(Word(start, end, token))
        previous = start
    if not words:
        if text.strip():
            raise ValueError('识别出了文字，但对齐模型未返回时间戳')
        return []
    if cursor < len(text):
        if any(c.isalnum() for c in text[cursor:]):
            raise ValueError('识别全文含有尚未对齐的文字')
        words[-1].word += text[cursor:]
    result, group = [], []
    def flush():
        result.append(segment_Transcribe(start=group[0].start, end=max(w.end for w in group),
                                        text=''.join(w.word for w in group), words=group))
    for word in words:
        if group and (sum(len(w.word) for w in group) + len(word.word) > max_chars or
                      word.end - group[0].start > max_seconds or word.start - group[-1].end > .8):
            flush()
            group = []
        group.append(word)
    if group:
        flush()
    return result
