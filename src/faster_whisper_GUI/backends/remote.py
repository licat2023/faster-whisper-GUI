"""ASR adapters backed by isolated persistent Python environments."""
import base64
from pathlib import Path
from .base import Recognition, TranscriptionInfo
from .streaming import StreamUpdate
from faster_whisper_GUI.domain.segments import segment_Transcribe, Word
from faster_whisper_GUI.runtime.model_process import ModelProcess


class RemoteBackend:
    sample_rate = 16000

    def __init__(self, settings):
        self.backend_id = settings['backend']
        self.is_closed = False
        self.settings = dict(settings)
        self.transport = ModelProcess(settings['interpreter'])
        try:
            self.transport.request('load', settings=settings)
        except Exception:
            self.transport.close()
            raise

    def recognize(self, audio, options):
        try:
            return self._recognize(audio, options)
        except RuntimeError:
            # A failed native child/protocol must be reloadable through the GUI.
            self.cancel()
            raise

    def _recognize(self, audio, options):
        if not isinstance(audio, (str, Path)):
            raise ValueError('独立模型文件识别需要音频文件路径')
        # faster-whisper owns media decoding, VAD and timestamps in its process.
        if self.backend_id == 'faster-whisper':
            return self._recognition(self.transport.request(
                'recognize', audio=str(audio), options=dict(options)))
        from .audio import prepared_audio
        with prepared_audio(audio) as source:
            data = self.transport.request('recognize', audio=source, options=dict(options))
        return self._recognition(data)

    @staticmethod
    def _recognition(data):
        try:
            segments = [segment_Transcribe(start=s['start'], end=s['end'], text=s['text'],
                        words=[Word(**w) for w in s.get('words', [])], speaker=s.get('speaker')) for s in data['segments']]
            return Recognition(segments, TranscriptionInfo(**data['info']), text=data.get('text'),
                               segment_timestamps=data['segment_timestamps'])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError('模型进程返回了无效识别结果，请重新加载模型') from error

    def start(self):
        if self.backend_id != 'sherpa':
            raise ValueError('当前后端不提供原生流式会话，请选择 sherpa-onnx')
        self.transport.request('start')
        return RemoteSession(self.transport)

    def close(self):
        self.is_closed = True
        self.transport.close()

    def cancel(self):
        self.is_closed = True
        self.transport.terminate()

    def stream_result(self, options):
        return self._recognition(self.transport.request('stream_result', options=options))


class RemoteSession:
    def __init__(self, transport):
        self.transport = transport

    def accept(self, samples):
        import numpy as np
        from .audio import mono_float_samples
        samples = mono_float_samples(samples).astype('<f4', copy=False)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError('流式音频必须是有限的单声道样本')
        encoded = base64.b64encode(samples.tobytes()).decode('ascii')
        return StreamUpdate(**self.transport.request('accept', samples=encoded))

    def finish(self):
        return StreamUpdate(**self.transport.request('finish'))
