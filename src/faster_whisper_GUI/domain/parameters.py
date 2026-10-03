"""转录和 VAD 参数类型。"""

from typing import List, TypedDict, Union

class VADParameters(TypedDict):
    threshold:float = 0.5
    min_speech_duration_ms:float = 250
    max_speech_duration_s:float = float("inf")
    min_silence_duration_ms:float = 2000
    window_size_samples:int = 1024
    speech_pad_ms:float = 400

class WhisperParameters(TypedDict):
    language:str = ""
    task:str = "transcribe"
    log_progress: bool = False
    beam_size:int = 5
    best_of:int = 5
    patience:float = 0.0
    length_penalty:float = 1.0
    temperature:list = [0.0]
    compression_ratio_threshold:float = 1.0
    log_prob_threshold:float = -1.0
    no_speech_threshold:float = 0.6
    condition_on_previous_text:bool = True
    initial_prompt:list = []
    prefix:str = ""
    repetition_penalty:float = 1.0
    no_repeat_ngram_size:int = 0
    prompt_reset_on_temperature:float = 0.5
    suppress_blank:bool = True
    suppress_tokens:list = []
    without_timestamps:bool = False
    max_initial_timestamp:float = 0.0
    word_timestamps:bool = False
    prepend_punctuations:str = ""
    append_punctuations:str = ""
    multilingual: bool = False
    max_new_tokens:int = None
    chunk_length:int = None
    clip_mode:int = 0
    clip_timestamps:Union[str, List[float]] = "0"
    hallucination_silence_threshold:float = None
    hotwords: str = None
    language_detection_threshold:float = None
    language_detection_segments:int = 2
