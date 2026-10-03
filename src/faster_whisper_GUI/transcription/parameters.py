"""文件与实时转录共用的参数转换。"""

from faster_whisper_GUI.config import Task_list

def buildTranscribeKwargs(parameters: dict, vad_filter: bool, vad_parameters: dict) -> dict:
    """
    把 GUI 的参数字典翻译成 WhisperModel.transcribe() 的关键字参数。

    文件转写与实时流式转写共用这一份：原先这 30 多个参数只写在 TranscribeWorker
    里，实时转写再抄一份的话两边会逐渐漂移。
    """
    return dict(
        language = parameters["language"],
        task = Task_list[int(parameters["task"])],
        log_progress = False,
        beam_size = parameters["beam_size"],
        best_of = parameters["best_of"],
        patience = parameters["patience"],
        length_penalty = parameters["length_penalty"],
        temperature = parameters["temperature"],
        compression_ratio_threshold = parameters["compression_ratio_threshold"],
        log_prob_threshold = parameters["log_prob_threshold"],
        no_speech_threshold = parameters["no_speech_threshold"],
        condition_on_previous_text = parameters["condition_on_previous_text"],
        initial_prompt = parameters["initial_prompt"],
        prefix = parameters["prefix"],
        repetition_penalty = parameters["repetition_penalty"],
        no_repeat_ngram_size = parameters["no_repeat_ngram_size"],
        prompt_reset_on_temperature = parameters["prompt_reset_on_temperature"],
        suppress_blank = parameters["suppress_blank"],
        suppress_tokens = parameters["suppress_tokens"],
        without_timestamps = parameters["without_timestamps"],
        max_initial_timestamp = parameters["max_initial_timestamp"],
        word_timestamps = parameters["word_timestamps"],
        prepend_punctuations = parameters["prepend_punctuations"],
        append_punctuations = parameters["append_punctuations"],
        multilingual = parameters["multilingual"],
        max_new_tokens = parameters["max_new_tokens"],
        chunk_length = parameters["chunk_length"],
        clip_timestamps = parameters["clip_timestamps"],
        hallucination_silence_threshold = parameters["hallucination_silence_threshold"],
        hotwords = parameters["hotwords"],
        language_detection_threshold = parameters["language_detection_threshold"],
        language_detection_segments = parameters["language_detection_segments"],
        vad_filter = vad_filter,
        vad_parameters = vad_parameters,
    )
