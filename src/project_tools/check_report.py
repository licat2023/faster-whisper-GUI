"""Regression cases for independently verified report.html findings."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import queue
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch


class ReportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from faster_whisper_GUI.runtime.rocm import setupROCm
        setupROCm()
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_timecode_carry_and_precision(self):
        from faster_whisper_GUI.runtime.timecode import secondsToHMS, secondsToMS, HMSToSeconds
        self.assertEqual(secondsToHMS(59.99996), '00:01:00,000')
        self.assertEqual(secondsToHMS(3599.99996), '01:00:00,000')
        self.assertEqual(secondsToHMS(5.12345), '00:00:05,123')
        self.assertEqual(secondsToMS(59.99996), '01:00.00')
        self.assertAlmostEqual(HMSToSeconds('00:00:05,123'), 5.123)
        for value in [None, float('nan'), -1]:
            with self.assertRaises(ValueError): secondsToHMS(value)

    def test_srt_multiline_colons_and_missing_final_blank(self):
        from faster_whisper_GUI.subtitles.readers import readSRTFileToSegments
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'input.srt'
            file.write_text('1\n00:00:00,123 --> 00:00:01,456\nhttps://example.com\nsecond line\n\n2\n00:00:02,000 --> 00:00:03,000\nNote: keep this', encoding='utf-8')
            result = readSRTFileToSegments(file)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].text, 'https://example.com\nsecond line')
            self.assertAlmostEqual(result[0].start, .123)
            self.assertEqual(result[1].text, 'Note: keep this')
            self.assertIsNone(result[1].speaker)

    def test_segment_roundtrip_preserves_speakers_and_distinct_words(self):
        from faster_whisper_GUI.domain.segments import Word, segment_Transcribe, segmentListToDictionaryList, dictionaryListToSegmentList
        original = segment_Transcribe(start=1, end=2, text='Hello world', speaker='A', words=[Word(1, 2, 'Hello', .9, 'A'), Word(1, 2, ' world', .8, 'A')])
        result = dictionaryListToSegmentList(segmentListToDictionaryList([original]))[0]
        self.assertEqual(result.speaker, 'A')
        self.assertEqual([w.word for w in result.words], ['Hello', ' world'])
        self.assertEqual(result.words[0].speaker, 'A')

    def test_partial_word_times_do_not_inherit_previous_end(self):
        from faster_whisper_GUI.domain.segments import dictionaryListToSegmentList
        result = dictionaryListToSegmentList([dict(start=0,end=2,text='ab',words=[dict(start=0,end=1,word='a'),dict(start=1.5,word='b')])])[0]
        self.assertEqual((result.words[1].start,result.words[1].end), (1.5,1.5))

    def test_dedup_keeps_different_text_at_same_times(self):
        from faster_whisper_GUI.domain.segments import Removerepetition
        a=dict(start=0,end=0,text='a'); b=dict(start=0,end=0,text='b')
        self.assertEqual(Removerepetition({'segments':[a,a.copy(),b]})['segments'], [a,b])

    def test_json_milliseconds_and_smi_many_speakers(self):
        from faster_whisper_GUI.domain.segments import segment_Transcribe
        from faster_whisper_GUI.subtitles.writers import writeSubtitles
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'output.json'
            segments=[segment_Transcribe(start=5.5,end=6.75,text='text',speaker=str(i)) for i in range(9)]
            writeSubtitles(str(path),segments,'JSON',fileName='input.wav')
            data=json.loads(path.read_text(encoding='utf-8-sig'))['data'][0]
            self.assertEqual(data['start']['ms'],500)
            writeSubtitles(str(Path(folder)/'output.smi'),segments,'SMI')
            with self.assertRaises(ValueError): writeSubtitles(str(path),segments,'invalid')

    def test_integer_stream_samples_are_normalized(self):
        import numpy as np
        from faster_whisper_GUI.backends.audio import StreamingResampler
        from faster_whisper_GUI.backends.remote import RemoteSession
        import base64
        samples=np.array([-32768,0,16384],dtype=np.int16)
        output=StreamingResampler(16000).process(samples)
        np.testing.assert_allclose(output,[-1,0,.5])
        sent=[]
        transport=NS(request=lambda action,**p: (sent.append(p) or dict(text='',revision=0,final=False,audio_seconds=0)))
        RemoteSession(transport).accept(samples)
        np.testing.assert_allclose(np.frombuffer(base64.b64decode(sent[0]['samples']),dtype='<f4'),[-1,0,.5])

    def test_float_wav_is_converted_to_pcm16(self):
        import numpy as np
        import soundfile as sf
        from faster_whisper_GUI.backends.audio import prepared_audio
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'float.wav'; sf.write(file,np.zeros(1600),16000,subtype='FLOAT')
            with prepared_audio(file) as result: self.assertEqual(sf.info(result).subtype,'PCM_16')

    def test_unicode_alignment_indices_and_cjk_tokens(self):
        from faster_whisper_GUI.backends.aligned import aligned_segments
        spans=[NS(text=t,start_time=i,end_time=i+.5) for i,t in enumerate(['İ','A'])]
        result=aligned_segments(spans,'İ a',2)
        self.assertEqual(''.join(s.text for s in result),'İ a')
        result=aligned_segments([NS(text=t,start_time=i,end_time=i+.5) for i,t in enumerate('你好')],'你好',2)
        self.assertEqual(''.join(s.text for s in result),'你好')

    def test_capture_error_sends_one_sentinel_and_prestart_stop(self):
        from faster_whisper_GUI.tasks.capture import CaptureAudioWorker
        frames=queue.Queue(); worker=CaptureAudioWorker(audio_queue=frames)
        with patch('faster_whisper_GUI.tasks.capture.sd.RawInputStream',side_effect=RuntimeError('device')):
            with self.assertLogs(level='CRITICAL'): worker.run()
        self.assertEqual(list(frames.queue),[None])
        worker=CaptureAudioWorker(audio_queue=queue.Queue()); worker.stop()
        with patch('faster_whisper_GUI.tasks.capture.sd.RawInputStream') as capture:
            worker.run(); capture.assert_not_called()

    def test_export_finishes_on_empty_and_io_failure(self):
        from faster_whisper_GUI.tasks.export import OutputWorker
        for result in [None, [([], 'input.wav', NS(language='zh'))]]:
            worker=OutputWorker(result,'','SRT'); done=[]; worker.signal_write_over.connect(lambda:done.append(True))
            with patch('faster_whisper_GUI.tasks.export.writeSubtitles',side_effect=OSError('disk full')):
                worker.run()
            self.assertEqual(done,[True]); self.assertFalse(worker.is_running)

    def test_bad_remote_response_becomes_reloadable_error(self):
        from faster_whisper_GUI.backends.remote import RemoteBackend
        with self.assertRaises(RuntimeError): RemoteBackend._recognition({'segments':[{}]})

    def test_output_drop_ignores_remote_empty_and_missing_urls(self):
        from PySide6.QtCore import QUrl
        from faster_whisper_GUI.ui.widgets.output_path import OutputGroupWidget
        widget=OutputGroupWidget(None); widget.LineEdit_output_dir.setText('original')
        for urls in [[],[QUrl('https://example.com')],[QUrl.fromLocalFile('Z:/missing/file')]]:
            widget.dropEvent(NS(mimeData=lambda:NS(urls=lambda:urls)))
            self.assertEqual(widget.LineEdit_output_dir.text(),'original')

    def test_media_probe_handles_zero_streams_and_errors(self):
        from faster_whisper_GUI.ui.widgets.file_list import FileNameListView
        from unittest.mock import MagicMock
        widget=FileNameListView(None)
        for streams in [[],[NS(codec_context=None)]]:
            container=MagicMock(); container.__enter__.return_value=container; container.streams=streams
            with patch('faster_whisper_GUI.ui.widgets.file_list.av.open',return_value=container):
                self.assertEqual(widget.testFileWithAudioTrackOrNot(['empty.wav'])[0],[])
            self.assertTrue(container.__exit__.called)

    def test_actual_icon_enum(self):
        from qfluentwidgets import FluentIcon
        self.assertTrue(hasattr(FluentIcon,'CONSTRACT'))


    def test_partial_config_preserves_defaults(self):
        from faster_whisper_GUI.ui.pages.model import ModelNavigationInterface
        from faster_whisper_GUI.ui.pages.transcription import TranscribeNavigationInterface
        from faster_whisper_GUI.ui.pages.vad import VADNavigationInterface
        from faster_whisper_GUI.ui.pages.separation import DemucsPageNavigation
        from faster_whisper_GUI.ui.pages.output import OutputPageNavigationInterface
        for cls in [ModelNavigationInterface, TranscribeNavigationInterface, VADNavigationInterface, DemucsPageNavigation, OutputPageNavigationInterface]:
            with self.subTest(page=cls.__name__):
                page=cls(); before=page.getParam(); page.setParam({}); self.assertEqual(page.getParam(),before)
                page.deleteLater()

    def test_result_delete_removes_all_matches_and_is_idempotent(self):
        from faster_whisper_GUI.ui.window.results import ResultActions
        result=[([], 'a.wav',None),([], 'a.wav',None),([], 'b.wav',None)]
        window=NS(tableModel_list={},current_result=result,result_faster_whisper=result,
                  result_whisperx_aligment=None,result_whisperx_speaker_diarize=None)
        ResultActions.deleteResultTableEvent(window,'tab_a.wav')
        self.assertEqual([r[1] for r in window.current_result],['b.wav'])
        ResultActions.deleteResultTableEvent(window,'tab_a.wav')

    def test_gui_log_target_is_connected_once(self):
        from PySide6.QtCore import QObject, Signal
        from faster_whisper_GUI.ui.window.feedback import WindowFeedback
        from faster_whisper_GUI import logging_setup
        class Window(QObject,WindowFeedback):
            signal_guiLog=Signal(str)
            def target(self,text): self.lines.append(text)
        window=Window(); window.lines=[]; window._guiHandler=None
        window._guiLogTarget=window.target; window.signal_guiLog.connect(window.target)
        try:
            window.redirectOutput(window.target); window.redirectOutput(window.target)
            window.signal_guiLog.emit('once'); self.assertEqual(window.lines,['once'])
        finally: logging_setup.detachHandler(window._guiHandler)

    def test_invalid_model_does_not_unload_existing(self):
        from faster_whisper_GUI.ui.window.model import ModelActions
        closed=[]; errors=[]; old=NS(close=lambda:closed.append(True))
        window=NS(inferenceBusy=lambda:False,getParam_model=lambda:{'model_size_or_path':''},
            FasterWhisperModel=old,loadModelWorker=None,page_model=NS(backend_combox=NS(currentData=lambda:'faster-whisper')),
            outputWithDateTime=lambda *a:None,_tr=lambda t:t,raiseErrorInfoBar=lambda **kw:errors.append(kw))
        ModelActions.onModelLoadClicked(window)
        self.assertIs(window.FasterWhisperModel,old); self.assertFalse(closed); self.assertTrue(errors)

    def test_rocm_child_cannot_trust_parent_flag(self):
        from faster_whisper_GUI.runtime import rocm
        with patch.object(rocm,'ROCM_AVAILABLE',False),patch.dict(os.environ,{rocm.ROCM_ENV_FLAG:'1',rocm.ROCM_PROCESS_FLAG:'-1'}):
            self.assertFalse(rocm.isROCmAvailable())

    def test_cpp_integer_task_index(self):
        from faster_whisper_GUI.backends.catalog import recognition_options
        self.assertEqual(recognition_options(NS(backend_id='whisper.cpp'),{'task':1})['task'],'translate')

    def test_speaker_setting_preserves_autoload_choice(self):
        from faster_whisper_GUI.ui.pages.settings import SettingPageNavigationInterface
        page=SettingPageNavigationInterface(); page.switchButton_autoLoadModel.setChecked(True)
        page.switchButton_saveConfig.setChecked(False); page.switchButton_saveConfig.setChecked(True)
        self.assertTrue(page.switchButton_autoLoadModel.isChecked())
        old=page.themeColor_str; page.setThemeColorWithLineEditText('#12ZZ45'); self.assertEqual(page.themeColor_str,old)

    def test_table_edits_emit_model_notifications(self):
        from faster_whisper_GUI.ui.widgets.result_table import CustomTableView
        from faster_whisper_GUI.ui.models.segments import TableModel
        from faster_whisper_GUI.domain.segments import segment_Transcribe
        view=CustomTableView(); model=TableModel([segment_Transcribe(text='a'),segment_Transcribe(text='b')]); view.setModel(model)
        reset=[]; model.modelReset.connect(lambda:reset.append(True))
        view.selectRow(0); view.delete_subtitles_line()
        self.assertEqual(model.rowCount(),1); self.assertTrue(reset)

    def test_demucs_short_audio_keeps_tail_and_amplitude(self):
        from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
        prepare_inference_runtime()
        import numpy as np
        import torch
        from faster_whisper_GUI.tasks.separation import DemucsWorker
        worker=DemucsWorker(None,[],1,'fake'); worker.is_running=True
        class Model:
            sources=['vocals']
            def to(self,device): return self
            def forward(self,chunk): return chunk[:,None]
        for length in [10,80,330]:
            with self.subTest(length=length):
                mix=np.ones((1,2,length),dtype=np.float32)
                out=worker.separate_sources(Model(),mix,segment=1,overlap=.5,sample_rate=100,device='cpu')
                np.testing.assert_allclose(out.numpy(),mix[:,None],atol=1e-6)

    def test_whisperx_missing_word_end_and_dataframe_unchanged(self):
        from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
        prepare_inference_runtime()
        import pandas as pd
        from whisperx.diarize import assign_word_speakers
        frame=pd.DataFrame([dict(start=0,end=2,speaker='A')]); before=frame.copy()
        result=assign_word_speakers(frame,{'segments':[dict(start=0,end=2,words=[dict(start=1,word='x')])]})
        self.assertEqual(result['segments'][0]['speaker'],'A'); pd.testing.assert_frame_equal(frame,before)

    def test_whisperx_single_frame_vad(self):
        from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
        prepare_inference_runtime()
        import numpy as np
        from pyannote.core import SlidingWindow,SlidingWindowFeature
        from whisperx.vad import Binarize
        result=Binarize(pad_offset=.01)(SlidingWindowFeature(np.array([[.9]]),SlidingWindow(duration=.02,step=.02)))
        self.assertTrue(list(result.itertracks()))

    def test_whisperx_generator_alignment(self):
        from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
        prepare_inference_runtime()
        import numpy as np
        from whisperx.alignment import align
        result=align(iter([dict(start=0,end=.5,text='!')]),None,{'dictionary':{},'language':'en','type':'huggingface'},np.zeros(16000,dtype=np.float32),'cpu')
        self.assertEqual(len(result['segments']),1)

    def test_split_filenames_stay_inside_output(self):
        from faster_whisper_GUI.tasks.audio_split import SplitAudioFileWithSpeakersWorker
        worker=SplitAudioFileWithSpeakersWorker([], '')
        with tempfile.TemporaryDirectory() as folder:
            name=worker.getOutPutFileName(folder,'00:00:01.000','00:00:02.000','../../bad:name')
            self.assertEqual(Path(name).resolve().parent,Path(folder).resolve())

    def test_malformed_model_protocol_is_runtime_error(self):
        from faster_whisper_GUI.runtime.model_process import ModelProcess
        import threading
        import io
        transport=ModelProcess.__new__(ModelProcess);transport.lock=threading.Lock(); transport.timeout=1
        transport.process=NS(poll=lambda:None,stdin=io.StringIO(),terminate=lambda:None);transport.responses=queue.Queue();transport.responses.put([])
        transport.log_path='test.log'
        with self.assertRaises(RuntimeError): transport.request('load')

    def test_whisperx_reuses_supplied_model(self):
        from whisperx import asr
        from unittest import mock
        supplied=object()
        with mock.patch.object(asr,'WhisperModel',side_effect=AssertionError('duplicate load')), mock.patch.object(asr,'load_vad_model',return_value=object()), mock.patch.object(asr,'FasterWhisperPipeline',side_effect=lambda **kwargs:kwargs):
            result=asr.load_model('tiny','cpu',model=supplied)
        self.assertIs(result['model'],supplied)

    def test_whisperx_subtitle_spans_all_segments(self):
        from whisperx.utils import WriteSRT
        result={'language':'en','segments':[{'start':0,'end':1,'words':[{'word':'a','start':0,'end':1}]},{'start':1,'end':2,'words':[{'word':'b','start':1,'end':2}]}]}
        with tempfile.TemporaryDirectory() as folder:
            cues=list(WriteSRT(folder).iterate_result(result,{'max_line_width':80,'max_line_count':2,'highlight_words':False}))
        self.assertEqual(cues[0][1],'00:00:02,000')

    def test_chinese_scoring_missing_native_total_and_empty_reference(self):
        from project_tools import score_chinese as scorer
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); fixtures=root/'.cache/fixtures/chinese';fixtures.mkdir(parents=True)
            reports=root/'.cache/reports/chinese';reports.mkdir(parents=True)
            panel={'id':'p','reference':'','audio_seconds':0,'clips':[]}
            (fixtures/'manifest.json').write_text(json.dumps({'panels':[panel],'revision':'test'}),encoding='utf-8')
            (reports/'p-model.json').write_text(json.dumps({'backend':'test','runs':[{'text':'','elapsed_seconds':3,'native_timings_seconds':{'load_time':2}}]}),encoding='utf-8')
            with patch.object(scorer,'ROOT',root):scorer.main()
            result=json.loads((root/'.cache/reports/chinese-comparison.json').read_text(encoding='utf-8'))['results'][0]
            self.assertEqual(result['processing_seconds'],3)
            self.assertIsNone(result['score']['cer']);self.assertIsNone(result['rtf'])

    def test_long_summary_handles_missing_metrics(self):
        from project_tools import summarize_long_benchmarks as summary
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixtures=root/'.cache/fixtures';fixtures.mkdir(parents=True)
            (fixtures/'librispeech-long.json').write_text(json.dumps({'clips':[]}))
            reports=root/'.cache/reports';reports.mkdir()
            (reports/'whisper-cpp-long.json').write_text(json.dumps({'backend':'whisper.cpp','runs':[]}))
            with patch.object(summary,'ROOT',root),patch.object(summary,'FOLDER',reports):summary.main()
            rows=json.loads((reports/'long-comparison.json').read_text())['results']
            self.assertEqual(rows[0]['status'],'invalid')
            self.assertIsNone(summary.wer([],[]))

    def test_cleanup_refuses_linked_ancestors_and_descendants(self):
        import shutil, subprocess
        shell=shutil.which('pwsh') or shutil.which('powershell')
        script=Path(__file__).with_name('clean_generated.ps1')
        for relative in ('src/linked','build/nested','.cache'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as folder:
                base=Path(folder).resolve();root=base/'repo';outside=base/'external';outside.mkdir()
                (outside/'keep.txt').write_text('keep')
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                script_dir=root/'src/project_tools';script_dir.mkdir(parents=True,exist_ok=True)
                copied=script_dir/script.name;shutil.copyfile(script,copied)
                (root/'src/check.pyc').write_bytes(b'keep until validation succeeds')
                command="New-Item -ItemType Junction -Path '"+str(target).replace("'","''")+"' -Target '"+str(outside).replace("'","''")+"' | Out-Null"
                subprocess.run([shell,'-NoProfile','-Command',command],check=True,capture_output=True)
                result=subprocess.run([shell,'-NoProfile','-File',str(copied),'-ClearBytecodeCache'],capture_output=True)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertEqual((outside/'keep.txt').read_text(),'keep')
                self.assertTrue((root/'src/check.pyc').exists())

    def test_native_stream_stop_before_run_does_not_start_backend(self):
        from faster_whisper_GUI.transcription.native_streaming import NativeStreamWorker
        worker=NativeStreamWorker(NS(start=lambda:self.fail('started cancelled backend')),queue.Queue(),16000,'',{})
        completed=[];worker.Signal_process_over.connect(completed.append)
        worker.stop();worker.run()
        self.assertEqual(completed,[[]])

    def test_stale_finished_signal_keeps_current_workers(self):
        from faster_whisper_GUI.ui.window.transcription import TranscriptionActions
        old=object();current=object()
        receiver=NS(sender=lambda:old,audio_stream_worker=current,audio_capture_thread=current,transcribe_thread=current)
        TranscriptionActions._captureFinished(receiver)
        TranscriptionActions._finishStream(receiver)
        TranscriptionActions._finishFile(receiver)
        self.assertIs(receiver.transcribe_thread,current)
        self.assertIs(receiver.audio_stream_worker,current)
        self.assertIs(receiver.audio_capture_thread,current)

    def test_whisperx_state_restored_after_inference_failure(self):
        from whisperx.asr import FasterWhisperPipeline
        import threading
        pipeline=FasterWhisperPipeline.__new__(FasterWhisperPipeline)
        pipeline._transcription_lock=threading.Lock()
        options=object();tokenizer=object();pipeline.options=options;pipeline.tokenizer=tokenizer
        def fail(*args,**kwargs):
            pipeline.options=object();pipeline.tokenizer=object()
            raise RuntimeError('failed inference')
        pipeline._transcribe=fail
        with self.assertRaises(RuntimeError):pipeline.transcribe([])
        self.assertIs(pipeline.options,options);self.assertIs(pipeline.tokenizer,tokenizer)

    def test_installed_paths_are_independent_of_launch_directory(self):
        from faster_whisper_GUI.runtime import paths
        source=Path(paths.__file__).read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for frozen in (False,True):
                namespace={'__file__':str(root/'site-packages/faster_whisper_GUI/runtime/paths.py')}
                with patch.object(sys,'frozen',frozen,create=True),patch.dict(os.environ,{'LOCALAPPDATA':str(root/'userdata')}):
                    exec(compile(source,'paths.py','exec'),namespace)
                self.assertEqual(namespace['PROJECT_ROOT'],root/'userdata/faster-whisper-GUI')

    def test_demucs_failure_releases_model(self):
        from faster_whisper_GUI.tasks.separation import DemucsWorker
        worker=DemucsWorker(None,['file.wav'],0,'unused')
        worker.loadModel=lambda *a,**k: (_ for _ in ()).throw(ValueError('invalid model'))
        status=[];worker.signal_vr_over.connect(status.append);worker.run()
        self.assertIsNone(worker.model);self.assertFalse(worker.is_running);self.assertEqual(status,[False])

    def test_editable_model_fields_survive_configuration_roundtrip(self):
        from faster_whisper_GUI.ui.pages.model import ModelNavigationInterface
        first=ModelNavigationInterface();second=ModelNavigationInterface()
        first.combox_online_model.setText('custom-model')
        first.preciese_combox.setText('custom-precision')
        second.setParam(first.getParam())
        self.assertEqual(second.combox_online_model.currentText(),'custom-model')
        self.assertEqual(second.preciese_combox.currentText(),'custom-precision')

    def test_cpp_device_diagnostics_variants_reject_cpu_fallback(self):
        from faster_whisper_GUI.backends.whisper_cpp import WhisperCppBackend
        backend=WhisperCppBackend('unused','unused','vulkan')
        backend._verify_device('ggml_vulkan: Found 1 Vulkan devices:')
        with self.assertRaises(RuntimeError):backend._verify_device('CPU only')
        backend.device='rocm';backend._verify_device('HIP initialized; using CUDA backend')
        with self.assertRaises(RuntimeError):backend._verify_device('using CUDA backend')

    def test_backend_loader_cancellation_before_and_during_load(self):
        from faster_whisper_GUI.tasks import backend_load
        worker=backend_load.BackendLoadWorker({});worker.stop()
        with patch.object(backend_load,'load_backend',side_effect=AssertionError('started')):worker.run()
        self.assertIsNone(worker.model)
        worker=backend_load.BackendLoadWorker({});closed=[]
        def load(settings):
            worker.stop();return NS(close=lambda:closed.append(True))
        with patch.object(backend_load,'load_backend',side_effect=load):worker.run()
        self.assertEqual(closed,[True]);self.assertIsNone(worker.model)

    def test_setup_rejects_missing_or_mismatched_dll_before_replacement(self):
        import shutil, subprocess
        shell=shutil.which('pwsh') or shutil.which('powershell')
        for exists in (False,True):
            with self.subTest(exists=exists),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();script=root/'setup.ps1'
                shutil.copyfile(Path(__file__).resolve().parents[2]/'setup.ps1',script)
                (root/'pyproject.toml').write_text('');(root/'uv.lock').write_text('')
                package=root/'.venv/Lib/site-packages/ctranslate2';package.mkdir(parents=True)
                (package/'version.py').write_text('__version__ = "4.8.2"')
                (package/'_ext.test.pyd').write_bytes(b'binding')
                destination=package/'ctranslate2.dll';destination.write_bytes(b'original')
                source=root/'replacement.dll'
                if exists:source.write_bytes(b'incompatible')
                marker=root/'uv-called'
                def quote(path):return "'"+str(path).replace("'","''")+"'"
                command='function uv { Set-Content -LiteralPath '+quote(marker)+" -Value called; $global:LASTEXITCODE=0 }; & "+quote(script)+' -RocmDll '+quote(source)+" -RocmDllVersion '9.9.9'"
                result=subprocess.run([shell,'-NoProfile','-Command',command],capture_output=True)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(destination.read_bytes(),b'original')
                if not exists:self.assertFalse(marker.exists())

if __name__ == '__main__': unittest.main(verbosity=2)
