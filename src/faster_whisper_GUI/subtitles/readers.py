"""Read subtitle text without guessing speakers from ordinary punctuation."""
import json
from faster_whisper_GUI.domain.segments import Word, segment_Transcribe
from faster_whisper_GUI.runtime.timecode import HMSToSeconds


def readJSONFileToSegments(file, file_code='utf8'):
    with open(file, encoding=file_code) as fp:
        data = json.load(fp)['data']
    segments = []
    for index, subtitle in enumerate(data, 1):
        try:
            if 'start' in subtitle and 'end' in subtitle:
                start, end = subtitle['start']['time'] / 1000, subtitle['end']['time'] / 1000
            else:
                start, end = subtitle['from'], subtitle['to']
            segments.append(segment_Transcribe(start=start, end=end, text=subtitle['content'],
                words=[Word(w['start'], w['end'], w['word'], w.get('probability', 0), w.get('speaker'))
                       for w in subtitle.get('words', [])], speaker=subtitle.get('speaker') or None))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'JSON 字幕第 {index} 段格式无效') from error
    return segments


def readSRTFileToSegments(file, file_code='utf8'):
    with open(file, encoding=file_code) as fp:
        lines = fp.read().lstrip('\ufeff').splitlines()
    segments, index = [], 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if lines[index].strip().isdigit():
            index += 1
        if index >= len(lines) or '-->' not in lines[index]:
            raise ValueError(f'SRT 第 {index + 1} 行缺少时间范围')
        start, end = lines[index].split('-->', 1)
        start, end = HMSToSeconds(start), HMSToSeconds(end.strip().split()[0])
        if end < start:
            raise ValueError('SRT 结束时间早于开始时间')
        index += 1
        text = []
        while index < len(lines) and lines[index].strip():
            if lines[index].strip().isdigit() and index + 1 < len(lines) and '-->' in lines[index + 1]:
                break
            text.append(lines[index])
            index += 1
        segments.append(segment_Transcribe(start=start, end=end, text='\n'.join(text)))
    return segments
