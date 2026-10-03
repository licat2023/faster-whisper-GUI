"""GUI backend selection, alignment, streaming revisions and subtitle export checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import queue

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_alignment_preserves_punctuation_and_measured_times(self):
        from faster_whisper_GUI.backends.aligned import aligned_segments
        items = [SimpleNamespace(text=t, start_time=i, end_time=i+.5) for i,t in enumerate('你好世界')]
        segments = aligned_segments(items, '你好，世界！', 4, max_chars=4)
        self.assertEqual(''.join(s.text for s in segments), '你好，世界！')
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[-1].end, 3.5)
        self.assertTrue(all(s.words for s in segments))

    def test_alignment_refuses_missing_or_invalid_spans(self):
        from faster_whisper_GUI.backends.aligned import aligned_segments
        with self.assertRaises(ValueError):
            aligned_segments([], '有文字', 3)
        with self.assertRaises(ValueError):
            aligned_segments([SimpleNamespace(text='字', start_time=2, end_time=1)], '字', 3)
        with self.assertRaises(ValueError):
            aligned_segments([SimpleNamespace(text='错误文字', start_time=0, end_time=1)], '原文字', 3)

    def test_backend_options_filter_whisper_parameters(self):
        from faster_whisper_GUI.backends.catalog import recognition_options
        backend = SimpleNamespace(backend_id='whisper.cpp')
        options = recognition_options(backend, {'language':'zh','beam_size':5,'task':False,'temperature':[0], 'audio':['file']})
        self.assertEqual(options, {'language':'zh','beam_size':5,'task':'transcribe'})

    def test_model_page_selection_and_persistence(self):
        from faster_whisper_GUI.ui.pages.model import ModelNavigationInterface
        page = ModelNavigationInterface()
        page.lineEdit_model_path.setText('existing-ct2')
        page.backend_combox.setCurrentIndex(page.backend_combox.findData('qwen'))
        page.backend_options.controls['model'].setText('local-qwen')
        saved = page.getParam()
        other = ModelNavigationInterface()
        other.setParam(saved)
        self.assertEqual(other.backend_combox.currentData(), 'qwen')
        self.assertEqual(other.backend_options.settings()['model'], 'local-qwen')
        other.backend_combox.setCurrentIndex(0)
        self.assertEqual(other.lineEdit_model_path.text(), 'existing-ct2')
        page.deleteLater()
        other.deleteLater()

    def test_transcription_and_separation_parameter_panels(self):
        from faster_whisper_GUI.ui.pages.transcription import TranscribeNavigationInterface
        from faster_whisper_GUI.ui.pages.separation import DemucsPageNavigation
        page = TranscribeNavigationInterface()
        page.selectBackend('whisper.cpp')
        self.assertEqual(set(page.backend_options.controls), {'language','beam_size','best_of','initial_prompt'})
        page.selectBackend('qwen')
        self.assertNotIn('beam_size', page.backend_options.controls)
        page.selectBackend('sherpa')
        self.assertEqual(page.backend_options.settings()['max_segment_chars'], 32)
        self.assertEqual(page.backend_options.settings()['max_segment_seconds'], 6)
        separation = DemucsPageNavigation()
        separation.backend_combox.setCurrentIndex(1)
        settings = separation.roformer_options.settings()
        self.assertEqual(settings['model'], 'vocals_mel_band_roformer.ckpt')
        self.assertEqual(int(settings['segment_size']), 64)
        saved = separation.getParam()
        separation.backend_combox.setCurrentIndex(0)
        separation.setParam(saved)
        self.assertEqual(separation.backend_combox.currentData(), 'roformer')
        page.deleteLater()
        separation.deleteLater()

    def test_native_stream_drains_queue_and_flushes_resampler(self):
        import numpy as np
        from faster_whisper_GUI.transcription.native_streaming import NativeStreamWorker
        from faster_whisper_GUI.backends.streaming import StreamUpdate
        from faster_whisper_GUI.backends.base import Recognition, TranscriptionInfo
        from faster_whisper_GUI.domain.segments import segment_Transcribe
        samples = []
        class Session:
            def accept(self, frame):
                samples.extend(frame)
                return StreamUpdate('修订文字', 1, False, len(samples)/16000)
            def finish(self):
                return StreamUpdate('最终文字', 2, True, len(samples)/16000)
        backend = SimpleNamespace(sample_rate=16000, start=Session, stream_result=lambda options:
            Recognition([segment_Transcribe(start=0,end=.1,text='最终文字')], TranscriptionInfo('zh',None,.1,None)))
        frames = queue.Queue()
        frames.put(np.zeros(4800, dtype=np.float32))
        frames.put(None)
        worker = NativeStreamWorker(backend, frames, 48000, 'capture.wav', {})
        updates, results = [], []
        worker.signal_update.connect(updates.append)
        worker.Signal_process_over.connect(results.append)
        worker.run()
        self.assertEqual(len(samples), 1600)
        self.assertTrue(updates[-1].final)
        self.assertEqual(updates[-1].text, '最终文字')
        self.assertEqual(results[0][0][0][0].text, '最终文字')

    def test_sherpa_subtitles_use_native_token_times(self):
        from faster_whisper_GUI.backends.token_segments import token_segments
        segments = token_segments(['你','好','世','界'], [.2,.4,1,1.2], 2, max_chars=2)
        self.assertEqual([(s.start,s.end,s.text) for s in segments], [(.2,1,'你好'),(1,2,'世界')])
        self.assertEqual(segments[0].words, [])

    def test_aligned_result_exports_srt_and_json(self):
        from faster_whisper_GUI.backends.aligned import aligned_segments
        from faster_whisper_GUI.subtitles.writers import writeSubtitles
        items = [SimpleNamespace(text='你好', start_time=.3, end_time=1.1)]
        segments = aligned_segments(items, '你好！', 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'aligned.srt'
            writeSubtitles(str(path), segments=segments, format='SRT', language='zh', fileName='input.wav')
            self.assertIn('00:00:00,300 --> 00:00:01,100', path.read_text(encoding='utf-8-sig'))
            writeSubtitles(str(Path(folder)/'aligned.json'), segments=segments, format='JSON', language='zh', fileName='input.wav')
            self.assertTrue((Path(folder)/'aligned.json').is_file())

    def test_prepared_audio_does_not_swallow_backend_io_errors(self):
        import numpy as np
        import soundfile as sf
        from faster_whisper_GUI.backends.audio import prepared_audio
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.wav'
            sf.write(path, np.zeros(1600), 16000)
            with self.assertRaisesRegex(OSError, 'backend failed'):
                with prepared_audio(path):
                    raise OSError('backend failed')

    def test_capture_start_failure_closes_device_and_ends_queue(self):
        from faster_whisper_GUI.tasks.capture import CaptureAudioWorker
        closed = []
        class Input:
            def start(self): raise RuntimeError('input unavailable')
            def close(self): closed.append(True)
        frames = queue.Queue()
        worker = CaptureAudioWorker(audio_queue=frames)
        with patch('faster_whisper_GUI.tasks.capture.sd.RawInputStream', return_value=Input()), self.assertLogs(level='CRITICAL'):
            worker.run()
        self.assertEqual(closed, [True])
        self.assertIsNone(frames.get_nowait())
        self.assertFalse(worker.is_running)

    def test_model_process_cancel_unblocks_request_and_closes_idempotently(self):
        import subprocess
        import threading
        from faster_whisper_GUI.runtime.model_process import ModelProcess
        popen = subprocess.Popen
        def delayed(command, **kwargs):
            return popen([sys.executable, '-u', '-c', 'import time; time.sleep(60)'], **kwargs)
        with patch('faster_whisper_GUI.runtime.model_process.subprocess.Popen', delayed):
            transport = ModelProcess(sys.executable)
        errors = []
        def request():
            try: transport.request('load')
            except RuntimeError as error: errors.append(error)
        thread = threading.Thread(target=request)
        thread.start()
        transport.terminate()
        thread.join(timeout=3)
        transport.close()
        transport.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsNotNone(transport.process.poll())

    def test_whisper_cpp_device_changes_select_correct_default_binary(self):
        from faster_whisper_GUI.ui.widgets.backend_options import BackendOptions, MODEL_FIELDS
        from faster_whisper_GUI.backends.catalog import CPP_EXECUTABLES
        panel = BackendOptions(MODEL_FIELDS)
        panel.select('whisper.cpp')
        panel.controls['device'].setCurrentText('rocm')
        self.assertEqual(panel.settings()['executable'], CPP_EXECUTABLES['rocm'])
        panel.controls['executable'].setText('custom-cli.exe')
        panel.controls['device'].setCurrentText('cpu')
        self.assertEqual(panel.settings()['executable'], 'custom-cli.exe')
        panel.deleteLater()

    def test_close_waits_for_workers_without_destroying_them(self):
        from faster_whisper_GUI.ui.window.settings import SettingsActions
        active, stopped, ignored, accepted = [True], [], [], []
        worker = SimpleNamespace(isRunning=lambda: active[0], stop=lambda: stopped.append(True))
        off = SimpleNamespace(isChecked=lambda: False)
        window = SimpleNamespace(_tr=lambda text: text, setEnabled=lambda state: None,
            audio_capture_thread=worker, close=lambda: None, FasterWhisperModel=None,
            page_setting=SimpleNamespace(switchButton_saveConfig=off, switchButton_autoClearTempFiles=off))
        event = SimpleNamespace(ignore=lambda: ignored.append(True), accept=lambda: accepted.append(True))
        with patch('faster_whisper_GUI.ui.window.settings.MessageBox') as dialog, \
             patch('faster_whisper_GUI.ui.window.settings.QTimer.singleShot') as deferred, \
             patch('faster_whisper_GUI.ui.window.settings.logging_setup.shutdownLogging') as shutdown:
            dialog.return_value.exec.return_value = True
            SettingsActions.closeEvent(window, event)
            self.assertEqual(stopped, [True])
            self.assertEqual(ignored, [True])
            self.assertFalse(accepted)
            deferred.assert_called_once()
            shutdown.assert_not_called()
            active[0] = False
            SettingsActions.closeEvent(window, event)
            self.assertEqual(accepted, [True])
            dialog.assert_called_once()
            shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
