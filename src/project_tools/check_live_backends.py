"""Opt-in installed-model checks; reports beneath .cache, temporary subtitles in temp/."""

if not __debug__:
    raise RuntimeError("Run verification without -O or PYTHONOPTIMIZE; assertions are required")

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('PYTHONPYCACHEPREFIX', str(ROOT / '.cache/pycache'))


def check(identity):
    from faster_whisper_GUI.backends.catalog import default_settings, load_backend
    from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
    prepare_inference_runtime()
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from faster_whisper_GUI.transcription.file import TranscribeWorker
    from faster_whisper_GUI.subtitles.writers import writeSubtitles
    import numpy as np
    import soundfile as sf
    destination = ROOT / '.cache/checks/gui-backends' / identity
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if identity == 'roformer':
        from faster_whisper_GUI.tasks.roformer import RoformerWorker
        settings = dict(backend=identity, model='vocals_mel_band_roformer.ckpt', segment_size=64, overlap=2,
            interpreter=str(ROOT / '.cache/experiments/separation-env/Scripts/python.exe'),
            output_path=str(destination), audio=[str(ROOT / '.cache/fixtures/hdemucs_mix_150_155.wav')])
        worker = RoformerWorker(settings)
        worker.run()
        assert worker.lastError is None, worker.lastError
        files = list(destination.glob('*.wav'))
        assert len(files) == 2, files
        for file in files:
            samples, rate = sf.read(file)
            assert np.isfinite(samples).all() and len(samples) / rate > 4.9
        report = {'backend': identity, 'files': [str(file) for file in files]}
    else:
        backend = load_backend(default_settings(identity))
        try:
            audio = ROOT / '.cache/fixtures/qwen-zh.wav'
            worker = TranscribeWorker(model=backend, parameters={'audio':[str(audio)], 'language':'zh',
                'task':'transcribe', 'beam_size':5, 'best_of':1, 'max_segment_chars':16, 'max_segment_seconds':6})
            results = []
            worker.signal_process_over.connect(results.extend)
            worker.run()
            assert worker.lastError is None, worker.lastError
            assert len(results) == 1, results
            segments, path, info = results[0]
            assert segments and all(0 <= s.start <= s.end <= info.duration + .1 for s in segments)
            if identity == 'qwen':
                assert all(s.words for s in segments)
            for format in ['SRT', 'JSON', 'VTT', 'ASS', 'LRC']:
                output = destination / ('result.' + format.lower())
                writeSubtitles(str(output), segments, format=format, language=info.language, fileName=path)
                assert output.stat().st_size > 0
            report = {'backend': identity, 'text': ''.join(s.text for s in segments),
                'segments': len(segments), 'words': sum(len(s.words) for s in segments), 'duration': info.duration}
            if identity == 'sherpa':
                from faster_whisper_GUI.transcription.native_streaming import NativeStreamWorker
                import queue
                samples, rate = sf.read(audio, dtype='float32')
                frames = queue.Queue()
                for offset in range(0, len(samples), 2048):
                    frames.put(samples[offset:offset+2048])
                frames.put(None)
                stream = NativeStreamWorker(backend, frames, rate, str(audio), {'language':'zh'})
                updates, streamed = [], []
                stream.signal_update.connect(updates.append)
                stream.Signal_process_over.connect(streamed.extend)
                stream.run()
                assert stream.lastError is None, stream.lastError
                assert streamed and updates[-1].final
                assert updates[-1].text == ''.join(s.text for s in streamed[0][0])
                report['revisions'] = [dict(text=u.text, revision=u.revision, final=u.final) for u in updates]
        finally:
            if hasattr(backend, 'close'):
                backend.close()
    report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['qwen', 'sherpa', 'whisper.cpp', 'roformer'])
    check(parser.parse_args().backend)
