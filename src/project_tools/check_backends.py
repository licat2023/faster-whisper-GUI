"""统一结果与 community-1 返回契约；不下载模型，不进行真实推理。"""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class BackendChecks(unittest.TestCase):
    def test_text_only_backend_keeps_missing_timestamps_explicit(self):
        from faster_whisper_GUI.backends.qwen import QwenBackend
        import numpy as np
        calls = []
        def transcribe(**kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(language="Chinese", text="测试文本")]
        result = QwenBackend(SimpleNamespace(transcribe=transcribe)).recognize(
            np.zeros(16000, dtype=np.float32), {"language": "zh"})
        self.assertEqual(result.info.language, "zh")
        self.assertEqual(calls[0]["language"], "Chinese")
        self.assertEqual(result.text, "测试文本")
        self.assertFalse(result.segment_timestamps)
        self.assertEqual(result.segments, [])
        self.assertIsNone(result.info.language_probability)

    def test_stream_sessions_revision_finish_and_isolation(self):
        from faster_whisper_GUI.backends.sherpa import SherpaStreamingBackend
        import numpy as np
        class Stream:
            def __init__(self):
                self.frames = []
                self.finished = False
                self.pending = False
            def accept_waveform(self, rate, samples):
                self.frames.append(len(samples))
                self.pending = True
            def input_finished(self):
                self.finished = True
        class Recognizer:
            def create_stream(self): return Stream()
            def is_ready(self, stream): return stream.pending
            def decode_stream(self, stream): stream.pending = False
            def get_result(self, stream): return "recognized" if stream.frames else ""
        backend = SherpaStreamingBackend(Recognizer())
        first = backend.start()
        update = first.accept(np.zeros(1600, dtype=np.float32))
        self.assertEqual(update.revision, 1)
        final = first.finish()
        self.assertTrue(final.final)
        self.assertEqual(final.audio_seconds, 0.1)
        self.assertEqual(final.revision, 1)
        with self.assertRaises(RuntimeError): first.accept(np.zeros(1))
        with self.assertRaises(RuntimeError): first.finish()
        second = backend.start()
        self.assertEqual(second.samples, 0)
        self.assertEqual(second.text, "")
        with self.assertRaises(ValueError): second.accept(np.array([float("nan")]))

    def test_pure_result_and_readers(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
from faster_whisper_GUI.domain.segments import Word, segment_Transcribe
from faster_whisper_GUI.backends.base import Recognition, TranscriptionInfo
from faster_whisper_GUI.subtitles.readers import readJSONFileToSegments
assert not any(name in sys.modules for name in
               ['torch', 'faster_whisper', 'ctranslate2', 'PySide6'])
assert segment_Transcribe().words == []
"""], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_segments_are_lazy_and_independent(self):
        from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
        prepare_inference_runtime()
        from faster_whisper.transcribe import Segment, Word
        from faster_whisper_GUI.backends.faster_whisper import FasterWhisperBackend
        from faster_whisper_GUI.domain.segments import Word as ResultWord
        native = Segment(id=0, seek=0, start=0.25, end=1.0, text="hello", tokens=[],
                         avg_logprob=0.0, compression_ratio=1.0, no_speech_prob=0.0,
                         temperature=0.0, words=[Word(0.25, 1.0, "hello", 0.9)])
        consumed = []
        def generate():
            consumed.append(True)
            yield native
        model = SimpleNamespace(transcribe=lambda **kwargs: (
            generate(), SimpleNamespace(language="en", language_probability=0.9,
                                        duration=1.0, duration_after_vad=1.0)))
        result = FasterWhisperBackend(model).recognize("sample.wav", {})
        self.assertEqual(consumed, [])
        segment = next(iter(result.segments))
        self.assertIsInstance(segment.words[0], ResultWord)
        segment.words[0].word = "changed"
        self.assertEqual(native.words[0].word, "hello")
        self.assertEqual(result.info.language, "en")

    def test_injected_backend_and_chunk_offset(self):
        from faster_whisper_GUI.backends.base import Recognition, TranscriptionInfo
        from faster_whisper_GUI.domain.segments import Word, segment_Transcribe
        from faster_whisper_GUI.transcription.streaming import AudioStreamTranscribeWorker
        import numpy as np
        options = []
        original = segment_Transcribe(start=0.25, end=1.0, text="hello",
                                     words=[Word(0.25, 1.0, "hello", 0.9)], speaker="S0")
        class Backend:
            sample_rate = 16000
            def recognize(self, audio, parameters):
                options.append(parameters)
                return Recognition([original], TranscriptionInfo("en", 1.0, 1.0, 1.0))
        worker = AudioStreamTranscribeWorker(model=Backend(), source_rate=16000)
        with patch("faster_whisper_GUI.transcription.streaming.buildTranscribeKwargs", return_value={}):
            shifted = worker._transcribeChunk(np.zeros(16000, dtype=np.float32), 30.0)[0]
        self.assertEqual(shifted.start, 30.25)
        self.assertEqual(shifted.words[0].start, 30.25)
        self.assertEqual(shifted.speaker, "S0")
        self.assertEqual(original.start, 0.25)
        self.assertFalse(options[0]["without_timestamps"])

    def test_community_and_legacy_diarization(self):
        from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
        prepare_inference_runtime()
        from pyannote.core import Annotation, Segment
        from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        import numpy as np
        overlap = Annotation()
        overlap[Segment(0, 2), "a"] = "S0"
        overlap[Segment(1, 3), "b"] = "S1"
        exclusive = Annotation()
        exclusive[Segment(0, 1.5)] = "S0"
        exclusive[Segment(1.5, 3)] = "S1"
        output = DiarizeOutput(overlap, exclusive)
        for native, use_exclusive in [(output, True), (output, False), (overlap, True),
                                      (DiarizeOutput(Annotation(), Annotation()), True)]:
            pipeline = DiarizationPipeline.__new__(DiarizationPipeline)
            pipeline.exclusive = use_exclusive
            pipeline.model = lambda *args, **kwargs: native
            frame = pipeline(np.zeros(16000, dtype=np.float32))
            self.assertEqual(list(frame.columns), [0, 1, "speaker", "start", "end"])
            self.assertIs(pipeline.last_output, native)
            if native is output and use_exclusive:
                self.assertEqual(frame.iloc[0]["end"], 1.5)
                transcript = {"segments": [{"start": 2.0, "end": 2.5, "text": "hi"}]}
                self.assertEqual(assign_word_speakers(frame, transcript)["segments"][0]["speaker"], "S1")
            elif native is output or native is overlap:
                self.assertEqual(frame.iloc[0]["end"], 2.0)
            else:
                self.assertTrue(frame.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
