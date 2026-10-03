"""Normalize arbitrary audio/video input for backends that accept WAV only."""
from contextlib import contextmanager
from pathlib import Path
import tempfile


def mono_float_samples(samples):
    """Accept mono float audio or normalize integer PCM to [-1, 1]."""
    import numpy as np
    data = np.asarray(samples)
    if data.ndim != 1:
        raise ValueError('音频必须是单声道样本')
    if np.issubdtype(data.dtype, np.signedinteger):
        data = data.astype(np.float32) / float(-np.iinfo(data.dtype).min)
    elif np.issubdtype(data.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(data.dtype).max + 1) / 2
        data = (data.astype(np.float32) - midpoint) / midpoint
    else:
        data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError('音频必须是有限样本')
    return data


class StreamingResampler:
    """Retain FFmpeg filter state across microphone frames, including the final tail."""
    def __init__(self, source_rate, target_rate=16000):
        import av
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.samples = 0
        self.resampler = av.AudioResampler(format='flt', layout='mono', rate=target_rate)

    def process(self, samples):
        import av
        import numpy as np
        from fractions import Fraction
        frame = av.AudioFrame.from_ndarray(mono_float_samples(samples).reshape(1, -1), format='flt', layout='mono')
        frame.sample_rate = self.source_rate
        frame.time_base = Fraction(1, self.source_rate)
        frame.pts = self.samples
        self.samples += len(samples)
        return self._samples(self.resampler.resample(frame))

    def finish(self):
        return self._samples(self.resampler.resample(None))

    @staticmethod
    def _samples(frames):
        import numpy as np
        return np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames]) if frames else np.empty(0, np.float32)


@contextmanager
def prepared_audio(source):
    import soundfile as sf
    if isinstance(source, (str, Path)):
        ready = False
        try:
            info = sf.info(source)
            ready = info.samplerate == 16000 and info.channels == 1 and info.format == 'WAV' and info.subtype == 'PCM_16'
        except (OSError, sf.LibsndfileError):
            pass
        if ready:
            yield str(Path(source).resolve())
            return
        import av
        resampler = av.AudioResampler(format='flt', layout='mono', rate=16000)
        with tempfile.TemporaryDirectory(prefix='asr-audio-') as folder:
            output = Path(folder) / 'audio.wav'
            count = 0
            with sf.SoundFile(output, 'w', samplerate=16000, channels=1, subtype='PCM_16') as wav:
                with av.open(str(source), metadata_errors='ignore') as container:
                    if not container.streams.audio:
                        raise ValueError('文件不包含音频流')
                    for frame in container.decode(audio=0):
                        frame.pts = None
                        for resampled in resampler.resample(frame):
                            samples = resampled.to_ndarray().reshape(-1)
                            wav.write(samples)
                            count += len(samples)
                for resampled in resampler.resample(None):
                    samples = resampled.to_ndarray().reshape(-1)
                    wav.write(samples)
                    count += len(samples)
            if not count:
                raise ValueError('文件不包含可解码的音频样本')
            yield str(output)
    else:
        with tempfile.TemporaryDirectory(prefix='asr-audio-') as folder:
            output = Path(folder) / 'audio.wav'
            sf.write(output, mono_float_samples(source), 16000, subtype='PCM_16')
            yield str(output)
