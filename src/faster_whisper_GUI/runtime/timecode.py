"""Subtitle timecodes; round before splitting hours, minutes and seconds."""
import math


def _ticks(value, scale):
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('时间必须是非负有限秒数') from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('时间必须是非负有限秒数')
    return round(seconds * scale)


def secondsToHMS(t) -> str:
    seconds, ms = divmod(_ticks(t, 1000), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}'


def secondsToMS(t) -> str:
    seconds, fraction = divmod(_ticks(t, 100), 100)
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes:02d}:{seconds:02d}.{fraction:02d}'


def _parse(value, fields):
    try:
        parts = value.strip().replace(',', '.').split(':')
        if len(parts) != fields:
            raise ValueError
        numbers = [float(p) for p in parts]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError
        if any(n != int(n) for n in numbers[:-1]) or any(n >= 60 for n in numbers[1:]):
            raise ValueError
        return sum(n * 60 ** (fields - i - 1) for i, n in enumerate(numbers))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f'无效时间码：{value!r}') from error


def HMSToSeconds(t: str) -> float:
    return _parse(t, 3)


def MSToSeconds(t: str) -> float:
    return _parse(t, 2)
