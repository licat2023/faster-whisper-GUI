"""Isolated model runtime. Stdout is reserved exclusively for JSON responses."""
import base64
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faster_whisper_GUI.runtime.paths import PROJECT_ROOT as ROOT


class Server:
    def stream_result(self, session, options):
        from faster_whisper_GUI.backends.token_segments import token_segments
        native = session.recognizer.get_result_all(session.stream)
        duration = session.samples / session.sample_rate
        segments = token_segments(native.tokens, native.timestamps, duration,
            int(options.get('max_segment_chars', 32)), float(options.get('max_segment_seconds', 6)))
        return {'segments': [dict(start=s.start, end=s.end, text=s.text, words=[]) for s in segments],
                'text': session.text, 'segment_timestamps': bool(segments) or not session.text,
                'info': dict(language=options.get('language') or 'zh', language_probability=None,
                             duration=duration, duration_after_vad=None)}

    def dispatch(self, request):
        action = request['action']
        if action == 'load':
            self.settings = settings = request['settings']
            backend = settings['backend']
            if backend == 'sherpa':
                from faster_whisper_GUI.backends.sherpa import SherpaStreamingBackend
                self.backend = SherpaStreamingBackend.from_directory(settings['model'], int(settings.get('threads', 4)))
            else:
                from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
                prepare_inference_runtime()
                import torch
                if backend == 'faster-whisper':
                    from faster_whisper import WhisperModel
                    from faster_whisper_GUI.backends.faster_whisper import FasterWhisperBackend
                    keys = ['model_size_or_path', 'device', 'device_index', 'compute_type',
                            'cpu_threads', 'num_workers', 'download_root', 'local_files_only']
                    model = WhisperModel(**{key: settings[key] for key in keys})
                    if settings.get('use_v3_model'):
                        extractor = model.feature_extractor
                        extractor.mel_filters = extractor.get_mel_filters(
                            extractor.sampling_rate, extractor.n_fft, n_mels=128).astype('float32')
                    self.backend = FasterWhisperBackend(model)
                elif backend == 'qwen':
                    from qwen_asr import Qwen3ASRModel
                    from faster_whisper_GUI.backends.qwen import QwenBackend
                    device = settings.get('device', 'cuda')
                    if device == 'cuda' and not torch.cuda.is_available():
                        raise ValueError('当前独立环境没有可用 GPU；请选择 CPU')
                    kwargs = dict(dtype=torch.float16 if device == 'cuda' else torch.float32,
                                  device_map='cuda:0' if device == 'cuda' else 'cpu',
                                  cache_dir=str(ROOT / '.cache/models/qwen'), attn_implementation='eager',
                                  local_files_only=settings.get('local_files_only', False))
                    model = Qwen3ASRModel.from_pretrained(settings['model'],
                        forced_aligner=settings['aligner'], forced_aligner_kwargs=kwargs,
                        max_new_tokens=int(settings['max_new_tokens']), max_inference_batch_size=1, **kwargs)
                    self.backend = QwenBackend(model, align=True)
                elif backend == 'roformer':
                    from audio_separator.separator import Separator
                    self.separator = Separator(model_file_dir=str(ROOT / '.cache/models/separation'),
                        output_dir=settings['output_path'], output_format='WAV', use_soundfile=True,
                        use_autocast=True, mdxc_params={'segment_size': int(settings['segment_size']),
                        'override_model_segment_size': True, 'batch_size': 1,
                        'overlap': int(settings['overlap']), 'pitch_shift': 0})
                    self.separator.load_model(settings['model'])
                else:
                    raise ValueError(f'未知独立后端：{backend}')
            return {'loaded': backend}
        if action == 'recognize':
            import soundfile as sf
            # Decode media in the main adapter, keeping sherpa's environment lightweight.
            if self.settings['backend'] == 'sherpa':
                audio, rate = sf.read(request['audio'], dtype='float32')
                if rate != 16000 or audio.ndim != 1:
                    raise ValueError('流式文件识别需要 16kHz 单声道')
                session = self.backend.start()
                for offset in range(0, len(audio), 1600):
                    session.accept(audio[offset:offset + 1600])
                final = session.finish()
                return self.stream_result(session, request['options'])
            result = self.backend.recognize(request['audio'], request['options'])
            return {'segments': [dict(start=s.start, end=s.end, text=s.text,
                    words=[asdict(w) for w in s.words], speaker=s.speaker) for s in result.segments],
                    'info': asdict(result.info), 'text': result.text, 'segment_timestamps': result.segment_timestamps}
        if action == 'start':
            self.session = self.backend.start()
            return True
        if action == 'accept':
            import numpy as np
            samples = np.frombuffer(base64.b64decode(request['samples'], validate=True), dtype='<f4')
            return asdict(self.session.accept(samples))
        if action == 'finish':
            return asdict(self.session.finish())
        if action == 'stream_result':
            return self.stream_result(self.session, request.get('options', {}))
        if action == 'separate':
            files = self.separator.separate(request['audio'])
            return [str(Path(self.settings['output_path']) / name) for name in files]
        raise ValueError(f'未知请求：{action}')


def main():
    output = sys.stdout
    server = Server()
    with redirect_stdout(sys.stderr):
        for line in sys.stdin:
            try:
                response = {'result': server.dispatch(json.loads(line))}
            except Exception as error:
                traceback.print_exc()
                response = {'error': f'{type(error).__name__}: {error}'}
            output.write(json.dumps(response, ensure_ascii=False) + '\n')
            output.flush()


if __name__ == '__main__':
    main()
