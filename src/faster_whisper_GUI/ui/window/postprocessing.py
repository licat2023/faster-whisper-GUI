"""对齐与说话人分离的界面编排。"""

import logging

log = logging.getLogger(__name__)

class PostprocessingActions:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def aligmentOver(self, segments_path_info:list):

        self.setPageOutButtonStatus(True)
        
        self.setStateTool(title=self._tr("WhisperX"), text=self._tr("结束"), status=True)
        # if segments_path_info is None:
        #     self.raiseErrorInfoBar(self._tr("错误"), content=self._tr("对齐失败，退出软件后检查 日志文件（设置页「日志文件」可打开）可能会获取错误信息"))
        #     return

        self.result_whisperx_aligment = segments_path_info
        if self.result_whisperx_aligment is not None:
            self.current_result = self.result_whisperx_aligment
            self.showResultInTable(results=self.current_result)
            self.raiseSuccessInfoBar(
                                    title=self._tr("WhisperX")
                                    , content=self._tr("时间戳对齐结束")
                                )

        else:
            self.raiseErrorInfoBar(
                                    self._tr("错误"),
                                    content=self._tr("对齐失败，检查 日志文件（设置页「日志文件」可打开）可能会获取更多信息")
                                )
        try:
            del self.whisperXWorker.model_alignment
        except Exception:
            pass
        try:
            del self.whisperXWorker.diarize_model
        except Exception:
            pass

        self.whisperXWorker = None

    def whisperXAligmentTimeStample(self):
        if self.result_faster_whisper is None and self.current_result is None:
            self.raiseErrorInfoBar(
                                    self._tr("错误"),
                                    self._tr("没有有效的 音频-字幕 转写结果，无法进行对齐")
            )
            return

        elif self.current_result is None :
            self.current_result = self.result_faster_whisper
        
        self.setPageOutButtonStatus(False)
        self.outputWithDateTime("TimeStample_Alignment")

        self.setStateTool(title=self._tr("WhisperX"), text=self._tr("时间戳对齐"), status=False)

        if self.whisperXWorker is None:
            from faster_whisper_GUI.tasks.alignment import WhisperXWorker
            self.whisperXWorker = WhisperXWorker(self.current_result, alignment=True, speaker_diarize=False, parent=self)
            self.whisperXWorker.stateToolRequest.connect(self.setStateTool)
        else:
            # 原先这里赋值给 result_segments_path_info —— 那是 run() 内部用来收集
            # 「输出」的属性（每次 run() 开头都会被重置成空列表），而 run() 读取的是
            # segments_path_info。于是第二次点「时间戳对齐」时，实际处理的是上一次
            # 遗留的旧结果，本次选中的转写结果被完全忽略。
            self.whisperXWorker.segments_path_info = self.current_result
            self.whisperXWorker.alignment = True
            self.whisperXWorker.speaker_diarize = False
        
        self._connectWhisperXFinished(self.aligmentOver)
        self.whisperXWorker.start()

    def whisperXDiarizeSpeakers(self):
        self.outputWithDateTime("Speaker_Diarize")
        
        whisperParams = self.getParamWhisperX()

        result_needed = next((result for result in (self.current_result, self.result_whisperx_aligment, self.result_faster_whisper) if result is not None), None)
        # print(f"result_useing: {result_needed}")
        # try:
        #     print(len(result_needed))
        # except:
        #     pass

        if result_needed is None:
            self.raiseErrorInfoBar(
                                    self._tr("错误"),
                                    self._tr("没有有效的 音频-字幕 转写结果，无法输出人声分离结果")
            )
            return

        self.setPageOutButtonStatus(False)

        if self.whisperXWorker is None:

            log.info("%s", f"min_speaker: {whisperParams['min_speaker']}")
            log.info("%s", f"max_speaker: {whisperParams['max_speaker']}")

            from faster_whisper_GUI.tasks.alignment import WhisperXWorker
            self.whisperXWorker = WhisperXWorker(result_needed
                                                , alignment=False
                                                , speaker_diarize=True
                                                , use_auth_token=whisperParams["use_auth_token"]
                                                , min_speaker=whisperParams["min_speaker"]
                                                , max_speaker=whisperParams["max_speaker"]
                                                , parent=self
                                            )
            self.whisperXWorker.stateToolRequest.connect(self.setStateTool)

        else:
            self.whisperXWorker.segments_path_info = result_needed
            self.whisperXWorker.alignment = False
            self.whisperXWorker.speaker_diarize = True
            self.whisperXWorker.use_auth_token = whisperParams['use_auth_token']
            self.whisperXWorker.min_speaker = whisperParams['min_speaker']
            self.whisperXWorker.max_speaker = whisperParams['max_speaker']

        self._connectWhisperXFinished(self.speakerDiarizeOver)
        self.setStateTool(title=self._tr("WhisperX"), text=self._tr("声源分离"), status=False)
        self.whisperXWorker.start()

    def _connectWhisperXFinished(self, slot):
        """
        保证 whisperXWorker.signal_process_over 只连到本次要用的那个结束回调。

        旧实现只在「说话人分离」分支里断开 aligmentOver，「时间戳对齐」分支从不断开；
        结束回调只应执行一次，避免重复展示结果和完成提示。
        """
        for candidate in (self.aligmentOver, self.speakerDiarizeOver):
            try:
                self.whisperXWorker.signal_process_over.disconnect(candidate)
            except (RuntimeError, TypeError):
                pass        # 本来就没连上
        self.whisperXWorker.signal_process_over.connect(slot)

    def setPageOutButtonStatus(self, enabled):
        for name in ['WhisperXAligmentTimeStampleButton', 'outputSubtitleFileButton',
                     'WhisperXSpeakerDiarizeButton', 'outputAudioPartWithSpeakerButton',
                     'unloadWhisperModelPushbutton']:
            getattr(self.page_output, name).setEnabled(enabled)

    def speakerDiarizeOver(self, segments_path_info:list):
        self.setPageOutButtonStatus(True)

        self.setStateTool(title=self._tr("WhisperX"), text=self._tr("结束"), status=True)
        if segments_path_info is None:
            self.raiseErrorInfoBar(self._tr("错误"),content=self._tr("声源分离失败，退出软件后检查 日志文件（设置页「日志文件」可打开）可能会获取错误信息"))
            return
        
        self.result_whisperx_speaker_diarize = segments_path_info
        if self.result_whisperx_speaker_diarize is not None:
            self.current_result = self.result_whisperx_speaker_diarize
            self.showResultInTable(results=self.current_result)
            self.raiseSuccessInfoBar(
                                    title=self._tr("WhisperX")
                                    , content=self._tr("声源分离结束")
                                )
            
        # for segments in self.result_whisperx_speaker_diarize:
        #         segment_, path, info = segments
        #         print(path, info.language)
        #         print(f"len:{len(segment_)}")
        #         for segment in segment_:
        #             try:
        #                 print(f"[{segment.start}s --> {segment.end}] | {segment['speaker']}:{segment.text}")
        #             except:
        #                 print(f"[{segment.start}s --> {segment.end}] | {segment.text}")

        #             print(f"len_words: {len(segment.words)}")

        self.whisperXWorker = None
