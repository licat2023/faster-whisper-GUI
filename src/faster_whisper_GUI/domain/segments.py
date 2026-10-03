"""segments 任务实现。"""

# coding:utf-8

import logging
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 0.0
    speaker: str | None = None


def normalize_word(word):
    return Word(float(word.start), float(word.end), word.word,
                float(word.probability), getattr(word, "speaker", None))

log = logging.getLogger(__name__)

class segment_Transcribe():

    def __init__(self, segment: Any=None, start:float = 0, end: float = 0, text:str = "", words : list = None, speaker:str=None):
        if segment:
            self.start = float(segment.start)
            self.end = float(segment.end)
            self.text = segment.text
            self.words = [normalize_word(word) for word in (getattr(segment, "words", None) or [])]
            self.speaker = getattr(segment, "speaker", None)

        else:
            self.start = float(start)
            self.end = float(end)
            self.text = text
            self.words = [normalize_word(word) for word in (words or [])]
            self.speaker = speaker


# ---------------------------------------------------------------------------------------------------------------------------
def segmentListToDictionaryList(segment_result:List[segment_Transcribe]) -> List[dict]:
    dict_result = []
    for segment in segment_result:
        words = segment.words
        words_ = []
        if words and len(words) > 0 :
            for word in words:
                words_.append({"word":word.word, "start":word.start, "end":word.end, "score":word.probability, "speaker":word.speaker})

        dict_result.append({"start":segment.start, "end":segment.end, "text":segment.text, "words":words_, "speaker":segment.speaker})

    return dict_result

def dictionaryListToSegmentList(dict_result:List[dict]) -> List[segment_Transcribe]:
    segment_result = []
    for item in dict_result:
        words = []
        for word in item.get('words', []):
            start = word.get('start', word.get('end', item['start']))
            end = word.get('end', start)
            words.append(Word(start, end, word['word'], word.get('score', 0.0), word.get('speaker')))
        segment_result.append(segment_Transcribe(start=item['start'], end=item['end'],
            text=item['text'], words=words, speaker=item.get('speaker')))

    return segment_result

def Removerepetition(result_a):
    # 清理输出结果 
    start = -1
    end = -1
    try:
        result_a_c = {"segments":[],"word_segments":result_a['word_segments']}
    except KeyError:
        result_a_c = {"segments":[]}

    for segment in result_a["segments"]:
        
        start_, end_ = segment['start'], segment['end']
        if result_a_c['segments'] and result_a_c['segments'][-1] == segment:
            # result_a['segments'].remove(segment)
            pass
        else:
            # print(start, end)
            start, end = segment['start'], segment['end']
            result_a_c['segments'].append(segment)
            log.info("%s", f"  [{start:.2f}s --> {end:.2f}s] {segment['text']}")
    
    return result_a_c
# ---------------------------------------------------------------------------------------------------------------------------
