"""字幕序列化与输出文件命名，不依赖 Qt 或推理运行时。"""

from __future__ import annotations

import logging
import os
from typing import List
import codecs
import json
import hashlib
import webvtt
from faster_whisper_GUI.config import Language_dict, Language_without_space
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper_GUI.domain.segments import segment_Transcribe
from faster_whisper_GUI.runtime.timecode import secondsToHMS, secondsToMS
from faster_whisper_GUI.config import ENCODING_DICT

log = logging.getLogger(__name__)

def writeSubtitles(outputFileName:str, 
                    segments:List[segment_Transcribe], 
                    format:str, 
                    language:str="",
                    fileName = "",
                    file_code = "UTF-8",
                    aggregate_contents = False
                ):
    
    format = format.upper()
    file_code = ENCODING_DICT[file_code]

    if format == "SRT":
        writeSRT(outputFileName, segments, file_code = file_code)
    elif format == "TXT":
        writeTXT(outputFileName, segments, file_code=file_code, aggregate_contents = aggregate_contents)
    elif format == "VTT":
        writeVTT(outputFileName, segments,language=language, file_code=file_code)
    elif format == "LRC":
        wirteLRC(outputFileName, segments,language=language, file_code=file_code)
    elif format == "SMI":
        writeSMI(outputFileName, segments, language=language, avFile=fileName, file_code=file_code)
    elif format == "JSON":
        writeJson(outputFileName, segments, language=language, avFile=fileName, file_code=file_code)
    elif format == "ASS":
        writeASS(outputFileName, segments, file_code=file_code)

    else:
        raise ValueError(f"不支持的字幕格式：{format}")

    log.info("%s", f"write over | {os.path.abspath(outputFileName)}")

def writeJson(fileName:str, segments:List[segment_Transcribe], language:str,avFile="", file_code="utf8"):

    _id = getMd5HashId(avFile, file_code=file_code)
    
    result = {
                "id": _id,
                "title": os.path.split(avFile)[-1],
                "format": "SubRip",
                "templates": {
                                "default": "__CONTENT__",
                                # 这里原本写作 "<i>__CONTENT__<\/i>"。
                                # \/ 在 Python 里是无效转义序列（反斜杠被保留），
                                # json.dump 又把它转义一次，于是写出的字幕文件里是
                                # "<\\/i>"，解析回来是 <\/i> —— 不是合法 HTML。
                                # 已去掉反斜杠，输出 <i>__CONTENT__</i>。
                                "italic": "<i>__CONTENT__</i>"
                            },
                "styles": {
                            "default": "font-style: 10px; line-height: 1; color: #FFF;"
                            }
                }
    
    result["data"] = []
    
    for segment in segments:
        start_time_HNS = secondsToHMS(segment.start)
        end_time_HMS = secondsToHMS(segment.end)

        result["data"].append(
                                {
                                "trigger": segment.start * 1000,
                                "lang": language,
                                "styles": [
                                    "default"
                                ],
                                "templates": [
                                    "default"
                                ],
                                "start": {
                                    "time": segment.start * 1000,
                                    "hour": int(start_time_HNS.split(":")[0]),
                                    "mins": int(start_time_HNS.split(":")[1]),
                                    "secs": int(start_time_HNS.split(":")[2].split(",")[0]),
                                    "ms": int(start_time_HNS.split(":")[2].split(",")[1])
                                },
                                "end": {
                                    "time": segment.end * 1000,
                                    "hour": int(end_time_HMS.split(":")[0]),
                                    "mins": int(end_time_HMS.split(":")[1]),
                                    "secs": int(end_time_HMS.split(":")[2].split(",")[0]),
                                    "ms": int(end_time_HMS.split(":")[2].split(",")[1])
                                },
                                "duration": {
                                    "secs": round(segment.end - segment.start, 3),
                                    "ms": round(segment.end - segment.start, 3) * 1000
                                },
                                "content": segment.text,
                                "meta": {
                                    "original": {
                                        "start": start_time_HNS,
                                        "end": end_time_HMS
                                    }
                                },
                                "words":[{"start":word.start,"end":word.end,"word":word.word,"probability":word.probability }for word in segment.words],
                                "speaker":segment.speaker or "",
                            }
                        )
    with open(os.path.abspath(fileName),'w',encoding=file_code) as fp:
    
        json.dump(
                    result,
                    fp,
                    ensure_ascii=False,
                    indent=4
                )

def writeSMI(fileName:str, segments:List[segment_Transcribe], language:str, avFile = "",file_code="utf8"):

    subtitle_color_list = ["white", "red", "blue", "green", "yellow", "cyan", "magenta"]

    # 获取音频或视频的名称
    _, fileName_ = os.path.split(fileName)
    baseName = fileName_.split('.')[0]

    if avFile:
        # 带有扩展名的文件名
        _, fileName_ = os.path.split(avFile)
    else:
        fileName_ = ""

    # 创建字幕的样式类
    language_type_CC = f"{language.upper()}CC"

    # 创建一个空的 smi 字幕字符串
    smi = ""
    # 添加 smi 字幕的头部标签
    smi += "<SAMI>\n"
    # 添加 smi 字幕的元数据和样式信息
    smi += "<HEAD>\n"
    # 标题
    smi += f"<TITLE>{baseName}</TITLE>\n"
    # 参数
    smi += "<SAMIParam>\n"
    smi += f"  Media {'{'}{fileName_}{'}'}\n"
    smi += "  Metrics {time:ms;}\n"
    smi += "  Spec {MSFT:1.0;}\n"
    smi += "</SAMIParam>\n"
    # 样式
    smi += "<STYLE TYPE=\"text/css\">\n"
    smi += "<!--\n"
    smi += "  P { font-family: Arial; font-weight: normal; color: white; background-color: black; text-align: center; }\n"
    speakers = ["SUB"]
    for segment in segments:
        try:
            speaker = segment.speaker
        except:
            speaker = "SUB"
        if not(speaker in speakers):
            speakers.append(speaker)
    if len(speakers) > 1:
        i = 0
        for speaker in speakers:
            smi += f"  #{speaker} {'{'} color: {subtitle_color_list[i % len(subtitle_color_list)]}; {'}'}\n"
            i += 1
    else:
        smi += "  #SUB{color: white; background-color: black; font-family: Arial; font-size: 12pt; font-weight: normal; text-align: left;}"
    if language != "zh":
        smi += f"  .{language_type_CC} {'{'} name: {Language_dict.get(language, language or "Unknown").capitalize()}; lang: {language}; SAMIType: CC; {'}'}\n"
    else:
        smi += f"  .{language_type_CC} {'{'} name: {'Chinese'}; lang: {language}; SAMIType: CC; {'}'}\n"
    smi += "-->\n"
    smi += "</STYLE>\n"
    smi += "</HEAD>\n"
    # 添加 smi 字幕的内容和时间信息
    smi += "<BODY>\n"
    # 遍历字幕列表，每个字幕是一个字典，包含 start, end, text, words 四个键
    for segment in segments:
        try:
            speaker = segment.speaker
            if not(speaker is None):
                speaker = segment.speaker
            else:
                speaker = "SUB"
        except:
            speaker = "SUB"
        
        # 添加字幕段的开始时间标签，格式为 <SYNC Start=毫秒数>
        smi += f"<SYNC Start={segment.start * 1000}>\n"
        # 添加字幕段的文本内容标签，格式为 <P Class=样式类名>文本内容
        # 如果有单词级时间戳，则在每个单词后面添加 <SPAN Class=样式类名>标签和时间戳
        if segment.words:
            if speaker != "SUB" and not(speaker is None):
                smi += f"  <P Class={language_type_CC}>{speaker}: "
            else:
                smi += f"  <P Class={language_type_CC}>"
            for word in segment.words:
                if not(language in Language_without_space):
                    word_text = word.word + " "
                else:
                    word_text = word.word

                # word_text = word.word
                try:
                    if word.end >= segment.start and word.end <= segment.end:
                        smi += f"<{secondsToHMS(word.start).replace(',','.')}><SPAN Class={language_type_CC}>{word_text}</SPAN>"
                    else:
                        smi += f"<SPAN Class={language_type_CC}>{word_text}</SPAN>"    
                    # smi += f"<SPAN Class={language_type_CC}>{word.word}</SPAN><{secondsToHMS(word.start).replace(',','.')}>"
                    # smi += f"{word.word}<SPAN Class={language_type_CC}>{secondsToHMS(word.start).replace(',','.')}</SPAN>"
                except:
                    smi += f"<SPAN Class={language_type_CC}>{word_text}</SPAN>"
            smi += "</P>\n"
        else:
            if speaker != "SUB" and not(speaker is None):
                smi += f"<P Class={language_type_CC}>{speaker}: {segment.text}</P>\n"
            else:
                smi += f"<P Class={language_type_CC}>{segment.text}</P>\n"

        # 添加字幕段的结束时间标签，格式为 <SYNC Start=毫秒数>
        smi += f"</SYNC>\n"
    # 添加 smi 字幕的尾部标签
    smi += "</BODY>\n"
    smi += "</SAMI>\n"

    # 使用 utf-8 重新编码字幕字符串
    smi:str = smi.encode("utf8").decode("utf8")

    # 将SMI字幕写入文件
    # f = codecs.open(fileName, "w",encoding=file_code)
    with codecs.open(fileName, "w", encoding=file_code) as f:
        f.write(smi)

def wirteLRC(fileName:str, segments:List[segment_Transcribe],language:str,file_code="utf8"):
    _, baseName = os.path.split(fileName)
    baseName = baseName.split(".")[0]
    with codecs.open(fileName, "w", encoding=file_code) as f:
        f.write(f"[ti:{baseName}]\n")
        f.write(f"[re:FasterWhisperGUI]\n")
        f.write(f"[offset:0]\n\n")

        for segment in segments:
            
            start:str = secondsToMS(segment.start)
            try:
                speaker = segment.speaker + ": "
            except:
                speaker = ""

            if segment.words:
                
                text = f"[{start[:8]}]" + speaker
                length = len(segment.words)
                for i in range(length):
                    word = segment.words[i]
                    if not(language in Language_without_space):
                        word_text = word.word + " "
                    else:
                        word_text = word.word
                    try:
                        if word.start >= segment.start and word.start <= segment.end:
                            text += f"<{secondsToMS(word.start)[:8]}>{word_text}"
                        else:
                            text += f"{word_text}"
                    except:
                        text += f"{word_text}"
                text += f"<{secondsToMS(segment.end)[:8]}>"
            else:
                text:str = f"[{start[:8]}]{speaker}{segment.text}"

            # 重编码为 utf-8 
            text:str = text.encode("utf8").decode("utf8")

            f.write(f"{text} \n")

def writeVTT(fileName:str, segments:List[segment_Transcribe],language:str,file_code="utf8"):
    # 创建一个空的 VTT 字幕对象
    _, baseName = os.path.split(fileName)
    vtt = webvtt.WebVTT()
    # 遍历字幕列表，每个字幕是一个字典，包含 start, end, text, words 四个键
    for segment in segments:
        # 创建一个空的字幕段对象
        cue = webvtt.Caption()
        # 设置字幕段的开始时间和结束时间，格式为 HH:MM:SS.mmm
        cue.start = secondsToHMS(segment.start).replace(",",".")
        cue.end = secondsToHMS(segment.end).replace(",", ".")
        # 设置字幕段的文本内容，如果有单词级时间戳，则输出时间戳和单词
        text = ""
        try:
            speaker = segment.speaker + ": "
        except:
            speaker = ""

        if segment.words:
            text = speaker + text
            for i in range(len(segment.words)):
                word = segment.words[i]
                if not(language in Language_without_space):
                    word_text = word.word + " "
                else:
                    word_text = word.word

                # if i == 0:
                if i == len(segment.words) - 1:
                    text += f"{word_text}"
                else:
                    try:
                        if word.end >= segment.start and word.end <= segment.end:
                            text += f"{word_text}<{secondsToHMS(word.end).replace(',','.')}>"
                        else:
                            text += f"{word_text}"
                        # text += f"<{secondsToHMS(word.start).replace(',','.')}>{word.word}"
                    except:
                        text += f"{word_text}"
        else:
            text = speaker + segment.text
        text:str = text.encode("utf8").decode("utf8")
        cue.text = text
        # 将字幕段添加到 VTT 字幕对象中
        vtt.captions.append(cue)
        
    vtt.save(fileName, file_code)

def writeTXT(fileName:str, segments, file_code="utf8", aggregate_contents=False):
    with codecs.open(fileName, "w", encoding=file_code) as f:
        speaker_temp = ""

        for segment in segments:
            
            text:str = segment.text
            try:
                speaker = segment.speaker + ": "
            except:
                speaker = ""
            
            if speaker_temp != speaker and aggregate_contents:
                f.write(f"\n{speaker.encode('utf8').decode('utf8')} \n")
                speaker_temp = speaker
            elif not aggregate_contents:
                text = speaker + text

            # 重编码为 utf-8 
            text:str = text.encode("utf8").decode("utf8")
            f.write(f"{text} \n")

def writeSRT(fileName:str, segments, file_code="UTF-8"):
    index = 1
    # encoding = ENCODING_DICT[file_code]
    with codecs.open(fileName, "w", encoding=file_code) as f:
        for segment in segments:
            start_time:float = segment.start
            end_time:float = segment.end
            text:str = segment.text

            try:
                speaker = segment.speaker + ": "
            except:
                speaker = ""

            text = speaker + text

            # 重编码为 utf-8 
            text:str = text.encode("utf8").decode("utf8")

            start_time:str = secondsToHMS(start_time)
            end_time:str = secondsToHMS(end_time)
            f.write(f"{index}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n")
            
            index += 1

def writeASS(fileName:str, segments, file_code="UTF-8"):
    with codecs.open(fileName, "w", encoding=file_code) as f:
        f.write("[Script Info]\n")
        f.write("; This file was generated by FasterWhisperGUI\n")
        f.write("; https://github.com/CheshireCC/faster-whisper-GUI\n")
        f.write("Original Script: FasterWhisperGUI\n")
        f.write("ScriptType: v4.00+\n")
        f.write("Collisions: Normal\n")
        f.write("PlayDepth: 0\n")
        f.write("Timer: 100.0000\n")
        f.write("WrapStyle: 0\n")

        f.write("\n[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: fwgDefault, Microsoft YaHei, 16, &H00FFFFFF, &H00FFFFFF, &H00000000, &H00000000, 0, 0, 0, 0, 100, 100, 0.00, 0.00, 1, 1, 0, 2, 20, 20, 20, 1\n")
        
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n")
        for segment in segments:
            # segment_Transcribe 始终带 speaker 字段，但 writeSubtitles 也会处理从
            # 外部读入（SRT/JSON）或由 whisperx 转换来的结果；用 getattr 兜底，
            # 避免因为缺一个字段在整个导出环节抛 AttributeError。
            speaker = getattr(segment, "speaker", None) or ""
            f.write(f'Dialogue: 0,{secondsToHMS(segment.start).replace(",",".")[:-1]},{secondsToHMS(segment.end).replace(",",".")[:-1]},fwgDefault,{speaker},0000,0000,0000,,{segment.text}\n')

def getSaveFileName(audioFile: str, format:str = "srt", rootDir:str = ""):
    path, fileName = os.path.split(audioFile)
    fileName = fileName.split(".")

    fileName[-1] = format.lower()

    fileName = ".".join(fileName)

    if rootDir != "":
        path = rootDir

    saveFileName = os.path.join(path, fileName).replace("\\", "/")
    return saveFileName

def getMd5HashId(fileName:str, file_code) -> str:
    
    md5 = hashlib.md5()
    md5.update(fileName.encode(file_code))
    id = md5.hexdigest()
    return id
