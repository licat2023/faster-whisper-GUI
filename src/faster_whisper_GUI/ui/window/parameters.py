"""从界面读取、校验和转换任务参数。"""

from faster_whisper_GUI.runtime.timecode import HMSToSeconds, MSToSeconds
from faster_whisper_GUI.domain.parameters import WhisperParameters, VADParameters

class WindowParameters:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def getParam_model(self) -> dict:
        """
        获取模型参数
        """
        
        if self.page_model.backend_combox.currentData() != 'faster-whisper':
            return self.page_model.backend_options.settings()
        if self.page_model.model_local_RadioButton.isChecked():
            model_size_or_path = self.page_model.lineEdit_model_path.text()
        else:
            model_size_or_path = self.page_model.combox_online_model.currentText()
        # 取下拉框的 userData（设备值），而非显示文本。
        # "AMD ROCm (HIP)" 选项的值为 "rocm"，但 CTranslate2 的 HIP 后端沿用 CUDA 的 API 命名，
        # 实际必须传 "cuda"，所以在这里做一次映射。
        device: str = self.page_model.device_combox.currentData()
        if device == "rocm":
            device = "cuda"
        device_index:str = self.page_model.LineEdit_device_index.text().replace(" ", "")
        device_index = [int(index) for index in device_index.split(",")]
        if len(device_index) == 1:
            device_index = device_index[0]

        compute_type: str = self.page_model.preciese_combox.currentText()
        cpu_threads: int = int(self.page_model.LineEdit_cpu_threads.text().replace(" ", ""))
        num_workers: int = int(self.page_model.LineEdit_num_workers.text().replace(" ", ""))
        download_root: str = self.page_model.LineEdit_download_root.text().strip()
        local_files_only: bool = self.page_model.switchButton_local_files_only.isChecked()
        use_v3_model: bool = self.page_model.switchButton_use_v3.isChecked()

        model_dict : dict = {
                    "model_size_or_path" : model_size_or_path,
                    "device" : device,
                    "device_index" : device_index,
                    "compute_type" : compute_type,
                    "cpu_threads" : cpu_threads,
                    "num_workers" : num_workers,
                    "download_root" : download_root,
                    "local_files_only" : local_files_only,
                    "use_v3_model" : use_v3_model
        }

        return model_dict

    def getParamWhisperX(self) -> dict:
        dict_WhisperXParams = {}
        dict_WhisperXParams["use_auth_token"] = self.page_setting.LineEdit_use_auth_token.text()

        dict_WhisperXParams["min_speaker"] = int(self.page_output.SpinBox_min_speaker.text())
        dict_WhisperXParams["max_speaker"] = int(self.page_output.SpinBox_max_speaker.text())

        if dict_WhisperXParams["min_speaker"] > dict_WhisperXParams["max_speaker"]:
            dict_WhisperXParams["max_speaker"] = dict_WhisperXParams["min_speaker"]
            
        if dict_WhisperXParams["min_speaker"] == 0 and dict_WhisperXParams["max_speaker"] == 0:
            dict_WhisperXParams["min_speaker"] = None
            dict_WhisperXParams["max_speaker"] = None

        return dict_WhisperXParams

    def getParamTranscribe(self) -> dict:

        if self.page_model.backend_combox.currentData() != 'faster-whisper':
            parameters = self.page_transcribes.backend_options.settings()
            parameters['audio'] = self.page_process.fileNameListView.FileNameModle.stringList()
            parameters['task'] = 'transcribe'
            return parameters

        Transcribe_params = WhisperParameters()

        # audio = self.page_process.LineEdit_audio_fileName.text().strip()
        # audio = audio.split(";;") if audio != "" else []

        # 从数据模型获取文件列表
        audio = self.page_process.fileNameListView.FileNameModle.stringList()

        if self.page_process.audio_capture_RadioButton.isChecked():
            audio = "AudioStream"
        
        Transcribe_params["audio"] = audio

        language = self.page_transcribes.combox_language.currentText().split("-")[0]
        if language == "Auto":
            language = None
        if language in ["zht","zhs"]:
            language = "zh"

        Transcribe_params["language"] = language

        task = self.page_transcribes.switchButton_Translate_to_English.isChecked()
        # task = STR_BOOL[task]
        # task = Task_list[int(task)]
        Transcribe_params["task"] = task

        Transcribe_params["log_progress"] = False

        beam_size = int(self.page_transcribes.LineEdit_beam_size.text().replace(" ", ""))
        Transcribe_params["beam_size"] = beam_size

        best_of = int(self.page_transcribes.LineEdit_best_of.text().replace(" ", ""))
        Transcribe_params["best_of"] = best_of

        patience = float(self.page_transcribes.LineEdit_patience.text().replace(" ", ""))
        Transcribe_params["patience"] = patience

        length_penalty = float(self.page_transcribes.LineEdit_length_penalty.text().replace(" ", ""))
        Transcribe_params["length_penalty"] = length_penalty

        temperature = self.page_transcribes.LineEdit_temperature.text().replace(" ", "")
        temperature = [float(t) for t in temperature.split(",")]
        Transcribe_params["temperature"] = temperature 

        compression_ratio_threshold = float(self.page_transcribes.LineEdit_compression_ratio_threshold.text().replace(" ", ""))
        Transcribe_params["compression_ratio_threshold"] = compression_ratio_threshold

        log_prob_threshold = float(self.page_transcribes.LineEdit_log_prob_threshold.text().replace(" ", ""))
        Transcribe_params["log_prob_threshold"] = log_prob_threshold

        no_speech_threshold = float(self.page_transcribes.LineEdit_no_speech_threshold.text().replace(" ", ""))
        Transcribe_params["no_speech_threshold"] = no_speech_threshold

        condition_on_previous_text = self.page_transcribes.switchButton_condition_on_previous_text.isChecked()
        # condition_on_previous_text = STR_BOOL[condition_on_previous_text]
        Transcribe_params["condition_on_previous_text"] = condition_on_previous_text

        initial_prompt = self.page_transcribes.LineEdit_initial_prompt.text().strip()
        pieces = initial_prompt.split(',')
        if initial_prompt and all(piece.strip().lstrip('-').isdigit() for piece in pieces):
            initial_prompt = [int(piece) for piece in pieces]
        else:
            initial_prompt = initial_prompt or None
        Transcribe_params["initial_prompt"] = initial_prompt

        prefix = self.page_transcribes.LineEdit_prefix.text().replace(" ", "") or None
        Transcribe_params["prefix"] = prefix

        suppress_blank = self.page_transcribes.switchButton_suppress_blank.isChecked()
        # suppress_blank = STR_BOOL[suppress_blank]
        Transcribe_params["suppress_blank"] = suppress_blank

        suppress_tokens = self.page_transcribes.LineEdit_suppress_tokens.text().replace(" ", "")
        suppress_tokens = [int(s) for s in suppress_tokens.split(",")]
        Transcribe_params["suppress_tokens"] = suppress_tokens

        without_timestamps = self.page_transcribes.switchButton_without_timestamps.isChecked()
        # without_timestamps = STR_BOOL[without_timestamps]
        Transcribe_params["without_timestamps"] = without_timestamps

        max_initial_timestamp = self.page_transcribes.LineEdit_max_initial_timestamp.text().replace(" ", "")
        max_initial_timestamp = float(max_initial_timestamp)
        Transcribe_params["max_initial_timestamp"] = max_initial_timestamp

        word_timestamps = self.page_transcribes.switchButton_word_level_timestampels.isChecked()
        # word_timestamps = STR_BOOL[word_timestamps]
        Transcribe_params["word_timestamps"] = word_timestamps

        prepend_punctuations = self.page_transcribes.LineEdit_prepend_punctuations.text().replace(" ", "")
        Transcribe_params["prepend_punctuations"] = prepend_punctuations

        append_punctuations = self.page_transcribes.LineEdit_append_punctuations.text().replace(" ","")
        Transcribe_params["append_punctuations"] = append_punctuations

        multilingual = self.page_transcribes.switchButton_multilingual.isChecked()
        Transcribe_params["multilingual"] = multilingual

        repetition_penalty = self.page_transcribes.LineEdit_repetition_penalty.text().strip()
        repetition_penalty = float(repetition_penalty)
        Transcribe_params['repetition_penalty'] = repetition_penalty  

        no_repeat_ngram_size = self.page_transcribes.LineEdit_no_repeat_ngram_size.text().strip()
        no_repeat_ngram_size = int(no_repeat_ngram_size)
        Transcribe_params["no_repeat_ngram_size"]  = no_repeat_ngram_size 

        prompt_reset_on_temperature  = self.page_transcribes.LineEdit_prompt_reset_on_temperature.text().strip()
        prompt_reset_on_temperature = float(prompt_reset_on_temperature)
        Transcribe_params['prompt_reset_on_temperature']  = prompt_reset_on_temperature 

        max_new_tokens = self.page_transcribes.LineEdit_max_new_tokens.text().strip()
        if max_new_tokens != "":
            if max_new_tokens.isdigit():
                max_new_tokens = int(max_new_tokens)
                if max_new_tokens == 448:
                    max_new_tokens = None
            else:
                max_new_tokens = None
        else :
            max_new_tokens = None
        Transcribe_params["max_new_tokens"] = max_new_tokens

        chunk_length = self.page_transcribes.LineEdit_chunk_length.text().strip()
        if chunk_length != "":
            if chunk_length.isdigit():
                chunk_length = int(chunk_length)
            else:
                chunk_length = None
        else :
            chunk_length = None
        Transcribe_params["chunk_length"] = chunk_length

        clip_mode = self.page_transcribes.ComboBox_clip_mode.currentIndex()
        Transcribe_params["clip_mode"] = clip_mode

        clip_timestamps = self.page_transcribes.LineEdit_clip_timestamps.text().strip()
        clip_timestamps = self.getClipTimestamps(clip_mode, clip_timestamps)
        Transcribe_params["clip_timestamps"] = clip_timestamps

        hallucination_silence_threshold = self.page_transcribes.lineEdit_hallucination_silence_threshold.text().strip()
        Transcribe_params["hallucination_silence_threshold"] = float(hallucination_silence_threshold)

        hotwords = self.page_transcribes.LineEdit_hotwords.text().strip()
        Transcribe_params["hotwords"] = hotwords

        language_detaction_th = self.page_transcribes.LineEdit_language_detection_threshold.text().strip()
        Transcribe_params["language_detection_threshold"] = float(language_detaction_th) if language_detaction_th != "" else None

        language_detaction_segments = self.page_transcribes.lienEdit_language_detection_segments.text().strip()
        Transcribe_params["language_detection_segments"] = int(language_detaction_segments) if language_detaction_segments != "" else None

        return Transcribe_params

    def getClipTimestamps(self, clip_mode:int, clip_timestamps:str):
        if clip_mode not in (0, 1, 2):
            raise ValueError("无效的剪辑时间模式")
        if clip_timestamps == "0":
            return clip_timestamps
        
        if clip_mode == 0:
            clip_timestamps_ = "0"
        elif clip_mode == 1:
            clip_timestamps_ = []
            clip_timestamps = clip_timestamps.split(";")
            for item in clip_timestamps:
                items = item.split("-")
                for item in items:
                    clip_timestamps_.append(item)
            
        elif clip_mode == 2:
            clip_timestamps_ = []
            clip_timestamps = clip_timestamps.split(";")
            for item in clip_timestamps:
                items = item.split("-")
                for item in items:
                    if len(item.split(":")) == 2:
                        clip_timestamps_.append(float(MSToSeconds(item)))
                    elif len(item.split(":")) == 3:
                        clip_timestamps_.append(float(HMSToSeconds(item)))

        return clip_timestamps_

    def getVADparam(self) -> dict:
        """
        get param of VAD
        """

        vad_filter = self.page_VAD.VAD_check_switchButton.isChecked()
        # print(vad_filter)
        VAD_param = {"vad_filter":vad_filter} 

        if not vad_filter:
            return VAD_param
        
        onset = round(self.page_VAD.doubleSpin_VAD_param_threshold.value(),2)
        min_speech_duration_ms = int(self.page_VAD.LineEdit_VAD_param_min_speech_duration_ms.text().replace(" ", ""))
        max_speech_duration_s = float(self.page_VAD.LineEdit_VAD_param_max_speech_duration_s.text().replace(" ", ""))
        min_silence_duration_ms = int(self.page_VAD.LineEdit_VAD_param_min_silence_duration_ms.text().replace(" ", ""))
        # window_size_samples = int(self.page_VAD.combox_VAD_param_window_size_samples.currentText())
        speech_pad_ms = int(self.page_VAD.LineEdit_VAD_param_speech_pad_ms.text().replace(" ", ""))

        VAD_param["param"] = VADParameters()
        # 注意：faster-whisper 1.x 已把该参数由 "onset" 改名为 "threshold"
        # （VADParameters 里定义的字段名也是 threshold，这里必须保持一致）
        VAD_param["param"]["threshold"] = onset
        VAD_param["param"]["min_speech_duration_ms"] = min_speech_duration_ms
        VAD_param["param"]["max_speech_duration_s"] = max_speech_duration_s
        VAD_param["param"]["min_silence_duration_ms"] = min_silence_duration_ms
        # VAD_param["param"]["window_size_samples"] = window_size_samples
        VAD_param["param"]["speech_pad_ms"] = speech_pad_ms

        return VAD_param

    def getDemucsParams(self):
        if self.page_demucs.backend_combox.currentData() == 'roformer':
            return {**self.page_demucs.roformer_options.settings(),
                    'output_path': self.page_demucs.outputGroupWidget.LineEdit_output_dir.text().strip(),
                    'audio': self.page_demucs.fileListView.avFileList}
        demucs_param = {}
        output_path = self.page_demucs.outputGroupWidget.LineEdit_output_dir.text().strip()
        demucs_param["output_path"] = output_path

        overlap = self.page_demucs.demucs_param_widget.spinBox_overlap.value()
        demucs_param["overlap"] = overlap

        segment = self.page_demucs.demucs_param_widget.spinBox_segment.value()
        demucs_param["segment"] = segment

        stems = self.page_demucs.demucs_param_widget.comboBox_stems.currentIndex()
        demucs_param["stems"] = stems

        audio = self.page_demucs.fileListView.avFileList
        demucs_param["audio"] = audio

        return demucs_param
