"""Backend identities, supported recognition options and installation locations."""
from pathlib import Path
from faster_whisper_GUI.runtime.paths import PROJECT_ROOT

BACKENDS = [('faster-whisper', 'faster-whisper'), ('whisper.cpp', 'whisper.cpp'),
            ('Qwen3-ASR + 时间戳对齐', 'qwen'), ('sherpa-onnx 原生流式', 'sherpa')]
CPP_EXECUTABLES = {device: str(PROJECT_ROOT / f'.cache/experiments/whisper.cpp/build-{device}/bin/whisper-cli.exe')
                   for device in ['vulkan', 'rocm']}


def default_settings(backend):
    root = PROJECT_ROOT / '.cache'
    settings = {'backend': backend, 'language': 'zh', 'threads': 4, 'beam_size': 5, 'best_of': 1,
                'max_segment_chars': 32, 'max_segment_seconds': 6}
    if backend == 'whisper.cpp':
        settings.update(model=str(root / 'models/ggml-large-v3-turbo.bin'),
                        executable=CPP_EXECUTABLES['vulkan'], device='vulkan')
    elif backend == 'qwen':
        settings.update(model='Qwen/Qwen3-ASR-0.6B', aligner='Qwen/Qwen3-ForcedAligner-0.6B',
                        device='cuda', max_new_tokens=2048, max_segment_chars=32, max_segment_seconds=6,
                        interpreter=str(root / 'experiments/qwen-env/Scripts/python.exe'))
    elif backend == 'sherpa':
        settings.update(model=str(root / 'models/streaming/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23'),
                        interpreter=str(root / 'experiments/streaming-env/Scripts/python.exe'), device='cpu')
    return settings


def recognition_options(backend, parameters, vad=False, vad_parameters=None):
    identity = getattr(backend, 'backend_id', 'faster-whisper')
    if identity == 'faster-whisper':
        from faster_whisper_GUI.transcription.parameters import buildTranscribeKwargs
        return buildTranscribeKwargs(parameters, vad, vad_parameters)
    if identity == 'whisper.cpp':
        keys = ['language', 'beam_size', 'best_of', 'initial_prompt']
        result = {key: parameters[key] for key in keys if key in parameters}
        result['task'] = 'translate' if parameters.get('task') in (True, 'translate') else parameters.get('task', 'transcribe') or 'transcribe'
        return result
    return {key: parameters[key] for key in ['language', 'max_segment_chars', 'max_segment_seconds'] if key in parameters}


def load_backend(settings):
    identity = settings['backend']
    if identity == 'whisper.cpp':
        from .whisper_cpp import WhisperCppBackend
        for key in ['model', 'executable']:
            if not Path(settings[key]).is_file():
                raise ValueError(f'{key} 文件不存在：{settings[key]}')
        result = WhisperCppBackend(settings['executable'], settings['model'], device=settings['device'],
                                   threads=int(settings['threads']), timeout=3600)
        result.backend_id = identity
        return result
    from .remote import RemoteBackend
    return RemoteBackend(settings)
