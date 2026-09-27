# coding:utf-8

import logging
from PySide6.QtCore import (QThread, Signal)

from faster_whisper_GUI.workers import GuardedWorker
import torch

import whisperx
from .seg_ment import (
                        Removerepetition
                        , dictionaryListToSegmentList
                        , segmentListToDictionaryList
                    )
import gc

log = logging.getLogger(__name__)

class WhisperXWorker(GuardedWorker):
    signal_process_over = Signal(list)

    def __init__(
                self
                , segments_path_info:list
                , alignment:bool
                , speaker_diarize:bool
                , use_auth_token:str=None
                , min_speaker:int=None
                , max_speaker:int=None
                , parent=None
            ) -> None:
        
        super().__init__(parent)
        self.is_running = False

        self.alignment = alignment
        self.speaker_diarize = speaker_diarize

        self.use_auth_token = use_auth_token
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker

        self.model_alignment = None
        self.metadata_alignment = None
        self.diarize_model = None

        self.segments_path_info = segments_path_info

    def stop(self):
        self.is_running = False

    def run(self):
        self.is_running = True
        self.result_segments_path_info = []
        # audio = None
        for (segments, path, info) in self.segments_path_info:
            audio = None
            # wav2vec2 对齐
            if self.alignment:
                try:
                    audio = whisperx.load_audio(path)
                    # 重新获取当前系统支持的设备
                    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                    log.info("%s", "\nTimeStample alignment")

                    # 获取字典格式的转写结果
                    log.info("%s", "transform transcript result...")
                    segment_dict_list = segmentListToDictionaryList(segments)

                    log.info("%s", "process audio...")

                    if self.model_alignment is None :
                        log.info("%s", "load wav2vec2 model...")
                        
                        self.setStateTool(text="load wav2vec2 model...",status=False)
                        self.model_alignment, self.metadata_alignment = whisperx.load_align_model(language_code=info.language
                                                                                                ,device=device
                                                                                                ,model_dir=r"./cache"
                                                                                                ,cache_dir=r"./cache"
                                                                                            )
                        
                    log.info("%s", "start alignment...")
                    self.setStateTool(text="start alignment...",status=False)
    
                    result_a = whisperx.align(segment_dict_list, self.model_alignment, self.metadata_alignment, audio, device, return_char_alignments=False)

                    # 清理结果
                    # print("after alignment: ")

                    # 清理可能存在的重复内容 时间戳完全一致的将会被合并 开启 faster-whisper 时间戳细分模式的情况下可能会出现此类结果
                    result_a_c = Removerepetition(result_a=result_a)

                except Exception as e:
                    log.error("%s", "alignment Error")
                    log.error("%s", f"Error: {e}")
                    self.alignment = False
                    self.signal_process_over.emit(None)
                    result_a_c = segments
                    del audio
                    return
            else:
                del audio
                audio = None
                result_a_c = segments

            if self.speaker_diarize:
                if audio is None:
                    audio = whisperx.load_audio(path)
                try:
                    log.info("%s", "\nSpeaker diarize and alignment")
                    # 重新获取当前系统支持的设备
                    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

                    if self.diarize_model is None:
                        log.info("%s", "load speaker brain model...")
                        self.setStateTool("load models...", False)
                        self.diarize_model = whisperx.DiarizationPipeline(
                                                                            use_auth_token=self.use_auth_token
                                                                            , device=device
                                                                            , cache_dir=r"./cache"
                                                                        )
                        
                    # 检查结果
                    if self.diarize_model is None:
                        log.error("%s", "load speaker brain model failed...")
                        self.setStateTool("load model failed", False)
                        self.signal_process_over.emit(None)
                        return

                    log.info("%s", "speaker diarize...")
                    self.setStateTool("start diarize...", False)
                    self.diarize_segments = self.diarize_model(audio
                                                            , min_speakers=self.min_speaker
                                                            , max_speakers=self.max_speaker
                                                        )
                    if not self.alignment:
                        log.info("%s", "process transcription result...")
                        result_a_c = {"segments":segmentListToDictionaryList(result_a_c)}

                    log.info("%s", "speaker alignment...")
                    self.setStateTool("assign speakers to words...")
                    result_s = whisperx.assign_word_speakers(self.diarize_segments, result_a_c)

                    # 检查结果
                    if result_s is None:
                        log.error("%s", "assign speakers to words failed...")
                        self.setStateTool("assign speakers to words failed", False)
                        self.signal_process_over.emit(None)
                        return
                    
                    # print("alignment result: ")
                    # for segment in result_s['segments']:
                    #     try:
                    #         print(f"  [{segment['start']:.2f}s -> {segment['end']:.2f}s] | {segment['speaker']}: {segment['text']}")
                    #     except Exception:
                    #         print(f"  [{segment['start']:.2f}s -> {segment['end']:.2f}s] | {segment['text']}")

                except Exception as e:
                    log.error("%s", "failed to diarize speaker!")
                    log.error("%s", f"Error: {e}")
                    result_s = result_a_c
                    self.speaker_diarize = False
                    self.signal_process_over.emit(None)
                    return

            else:
                result_s = result_a_c

            if not(audio is None):
                del audio

            try:
                if self.alignment or self.speaker_diarize:
                    # 字典列表转换回对象列表
                    segments = dictionaryListToSegmentList(result_s['segments'])
            except Exception as e:
                log.error("%s", "failed to transform alignment result!")
                log.error("%s", str(e))
                self.signal_process_over.emit(None)
                return

            self.result_segments_path_info.append((segments, path, info))

        self.signal_process_over.emit(self.result_segments_path_info)

        try:
            del self.model_alignment
            self.model_alignment = None
        except:
            pass
        
        try:
            del self.metadata_alignment
            self.metadata_alignment = None
        except:
            pass

        try:
            del self.diarize_segments
            self.diarize_segments = None
        except:
            pass

        # 清除显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # gc强制回收，避免内存泄露
        gc.collect()


    def setStateTool(self, text:str , status:bool=False):
        try:
            self.parent().setStateTool(text=text,status=status)
        except Exception as e:
            log.error("%s", f"To set StateTool Error: {e}")







