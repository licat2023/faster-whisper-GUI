"""GUI interaction regressions: synthetic data, real Qt controls, isolated outputs."""
if not __debug__:
    raise RuntimeError('Run verification without -O; assertions are required')
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
DEST = ROOT / '.cache/checks/gui-interactions'
DEST.mkdir(parents=True, exist_ok=True)
from PySide6.QtCore import QEventLoop, QTimer, Qt, QItemSelectionModel
from PySide6.QtWidgets import QApplication
from qfluentwidgets import qconfig
from faster_whisper_GUI import logging_setup
logging_setup.resolveLogDir = lambda: DEST
logging_setup.pruneOldRuns = lambda *a, **k: []
from faster_whisper_GUI.ui.window.main import MainWindows
from faster_whisper_GUI.ui.window.view import UIMainWin
from faster_whisper_GUI.ui.window.settings import SettingsActions
from faster_whisper_GUI.domain.segments import segment_Transcribe

class Window(MainWindows):
    def readConfigJson(self, config_file_path=''):
        return UIMainWin.readConfigJson(self, config_file_path if getattr(self, '_audit_ready', False) else '')
    def onModelLoadClicked(self):
        pass
    def closeEvent(self, event):
        event.accept()
    def raiseInfoBar(self, title, content):
        self.notices.append(('info', title, content))
    def raiseSuccessInfoBar(self, title, content):
        self.notices.append(('success', title, content))
    def raiseErrorInfoBar(self, title, content):
        self.notices.append(('error', title, content))

def result(path, texts=('DELETE_ME', 'KEEP_ME')):
    return ([segment_Transcribe(start=i * 2, end=i * 2 + 1.5, text=t) for i, t in enumerate(texts)],
            str(DEST / path), NS(language='en', duration=4, language_probability=1))

class Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.save_patch = patch.object(qconfig, 'save', lambda *a, **k: None)
        cls.save_patch.start()
    @classmethod
    def tearDownClass(cls):
        cls.save_patch.stop()
    def setUp(self):
        self.w = Window()
        self.w.notices = []
        self.w._audit_ready = True
        self.w.page_setting.combox_autoGoToOutputPage.setCurrentIndex(1)
        self.w.page_setting.switchButton_saveConfig.setChecked(False)
        self.w.page_setting.switchButton_autoClearTempFiles.setChecked(False)
    def tearDown(self):
        self.w.close()
        self.app.processEvents()
    def install(self, results):
        self.w.current_result = results
        self.w.result_faster_whisper = results
        self.w.showResultInTable(results)
    def wait(self, worker):
        loop, timer = QEventLoop(), QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        worker.finished.connect(loop.quit)
        timer.start(3000)
        if worker.isRunning():
            loop.exec()
        self.app.processEvents()
        from shiboken6 import isValid
        self.assertTrue(not isValid(worker) or not worker.isRunning(), 'Worker did not finish')
    def select_rows(self, table, rows):
        for row in rows:
            table.selectionModel().select(table.model().index(row, 3),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def test_delete_row_is_reflected_in_gui_export(self):
        self.install([result('delete.wav')])
        table = self.w.page_output.tableTab.stackedWidget.widget(0)
        self.select_rows(table, [0])
        table.delete_subtitles_line()
        self.assertEqual(table.model().rowCount(), 1)
        self.w.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(DEST / 'export'))
        self.w.page_output.combox_output_format.setCurrentText('SRT')
        self.w.page_output.outputSubtitleFileButton.click()
        worker = self.w.outputWorker
        self.wait(worker)
        self.assertIsNone(worker.lastError)
        outputs = list((DEST / 'export').glob('*.srt'))
        self.assertTrue(outputs)
        self.assertNotIn('DELETE_ME', outputs[0].read_text(encoding='utf-8-sig'),
            'Deleted row is still present in the exported SRT')

    def test_close_after_tab_reordering_closes_matching_table(self):
        self.install([result('a.wav'), result('b.wav')])
        tabs = self.w.page_output.tableTab
        tabs.movableCheckBox.setChecked(True)
        tabs.tabBar.setCurrentIndex(0)
        # This is the installed TabBar's actual drag swap callback; emits tabMoved.
        tabs.tabBar._swapItem(1)
        self.assertEqual(tabs.tabBar.tabItem(0).routeKey(), 'tab_' + str(DEST / 'b.wav').replace('\\', '/'))
        tabs.tabBar.tabCloseRequested.emit(0)
        remaining_tabs = [item.routeKey() for item in tabs.tabBar.items]
        remaining_views = [tabs.stackedWidget.widget(i).objectName() for i in range(tabs.stackedWidget.count())]
        self.assertEqual(remaining_views, remaining_tabs,
            'Closing B removes A table while removing B result data')

    def test_split_failure_restores_buttons_and_reports_error(self):
        self.install([result('split.wav')])
        self.w.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(DEST / 'split-failure'))
        with patch('faster_whisper_GUI.tasks.audio_split.subprocess.run',
                   return_value=NS(returncode=1, stderr='AUDIT simulated ffmpeg failure')):
            self.w.page_output.outputAudioPartWithSpeakerButton.click()
            worker = self.w.splitAudioFileWithSpeakerWorker
            self.wait(worker)
        self.assertIsNotNone(worker.lastError)
        self.assertTrue(self.w.page_output.outputSubtitleFileButton.isEnabled(),
            'ffmpeg failure permanently disables all five output actions')
        self.assertTrue(any(n[0] == 'error' for n in self.w.notices))

    def test_close_waits_for_running_audio_split(self):
        self.install([result('close.wav')])
        self.w.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(DEST / 'split-close'))
        entered, release = threading.Event(), threading.Event()
        def blocked(*a, **k):
            entered.set()
            release.wait(2)
            return NS(returncode=0, stderr='')
        class Event:
            accepted = False
            def accept(self): self.accepted = True
            def ignore(self): self.accepted = False
        event = Event()
        try:
            with patch('faster_whisper_GUI.tasks.audio_split.subprocess.run', side_effect=blocked):
                self.w.page_output.outputAudioPartWithSpeakerButton.click()
                worker = self.w.splitAudioFileWithSpeakerWorker
                self.assertTrue(entered.wait(1))
                with patch('faster_whisper_GUI.ui.window.settings.MessageBox') as dialog, \
                     patch.object(logging_setup, 'shutdownLogging'):
                    dialog.return_value.exec.return_value = True
                    SettingsActions.closeEvent(self.w, event)
                running_on_accept = worker.isRunning() and event.accepted
                release.set()
                self.wait(worker)
        finally:
            release.set()
        self.assertFalse(running_on_accept,
            'Close accepts while child QThread audio split is still running')

    def test_new_transcription_preserves_unsaved_prior_result(self):
        self.install([result('old.wav')])
        table = self.w.page_output.tableTab.stackedWidget.widget(0)
        table.model().setData(table.model().index(0, 3), 'UNSAVED_EDIT', Qt.ItemDataRole.EditRole)
        self.w.transcribeOver([result('new.wav', ('NEW',))])
        self.assertIn(str(DEST / 'old.wav').replace('\\', '/'), self.w.tableModel_list,
            'New task silently removes the previous file and its unsaved edit')

    def test_restore_config_cannot_change_backend_during_recognition(self):
        self.w.page_model.backend_combox.setCurrentIndex(self.w.page_model.backend_combox.findData('qwen'))
        self.w.transcribe_thread = NS(isRunning=lambda: True)
        self.w._lockRecognitionControls(True)
        config = DEST / 'restore-sherpa.json'
        config.write_text(json.dumps({'model_param': {'backend': 'sherpa'}}), encoding='utf-8')
        with patch('faster_whisper_GUI.ui.window.settings.QFileDialog.getOpenFileName',
                   return_value=(str(config), 'json file(*.json)')):
            self.w.page_setting.pushButton_loadConfigFile.click()
        self.w.transcribe_thread = None
        self.assertEqual(self.w.page_model.backend_combox.currentData(), 'qwen',
            'Restore bypasses disabled backend selection during a running task; recording enabled=' + str(self.w.page_process.audio_capture_RadioButton.isEnabled()))
        self.assertFalse(self.w.page_process.audio_capture_RadioButton.isEnabled())

    def test_invalid_config_does_not_report_success(self):
        config = DEST / 'broken.json'
        config.write_text('{broken', encoding='utf-8')
        with patch('faster_whisper_GUI.ui.window.settings.QFileDialog.getOpenFileName',
                   return_value=(str(config), 'json file(*.json)')):
            self.w.page_setting.pushButton_loadConfigFile.click()
        self.assertTrue(any(n[0] == 'error' for n in self.w.notices),
            'Malformed JSON notices=' + repr(self.w.notices))

    def test_merge_notifies_view_of_removed_rows(self):
        self.install([result('merge.wav')])
        table = self.w.page_output.tableTab.stackedWidget.widget(0)
        notifications = []
        for signal in [table.model().modelReset, table.model().rowsRemoved, table.model().dataChanged]:
            signal.connect(lambda *a: notifications.append(True))
        self.select_rows(table, [0, 1])
        table.merge_subtitles()
        self.assertEqual(table.model().rowCount(), 1)
        self.assertTrue(notifications, 'Merge mutates model row count without notifying Qt')

    def test_navigation_and_backend_capabilities(self):
        for page in self.w.pages:
            self.w.stackedWidget.setCurrentWidget(page)
            self.assertIs(self.w.pivot.panel.currentItem(), self.w.pivot.panel.widget(page.objectName()))
        for identity in ['faster-whisper', 'whisper.cpp', 'qwen', 'sherpa']:
            self.w.page_model.backend_combox.setCurrentIndex(self.w.page_model.backend_combox.findData(identity))
            self.assertEqual(self.w.page_transcribes.backend_options.identity, identity)
            self.assertEqual(self.w.page_VAD.isEnabled(), identity == 'faster-whisper')
            self.assertEqual(self.w.page_process.audio_capture_RadioButton.isEnabled(), identity == 'sherpa')

    def test_same_file_replacement_requires_confirmation_for_edits(self):
        self.install([result('same.wav'), result('other.wav')])
        table = self.w.page_output.tableTab.stackedWidget.widget(0)
        model = table.model()
        model.setData(model.index(0, 3), 'UNSAVED_EDIT', Qt.ItemDataRole.EditRole)
        with patch('faster_whisper_GUI.ui.window.results.MessageBox') as dialog:
            dialog.return_value.exec.return_value = False
            self.w.transcribeOver([result('same.wav', ('NEW',))])
            dialog.assert_called_once()
        self.assertEqual(self.w.current_result[0][0][0].text, 'UNSAVED_EDIT')
        self.assertTrue(model.isModified)
        with patch('faster_whisper_GUI.ui.window.results.MessageBox') as dialog:
            dialog.return_value.exec.return_value = True
            self.w.transcribeOver([result('same.wav', ('NEW',))])
            dialog.assert_called_once()
        self.assertEqual(self.w.current_result[0][0][0].text, 'NEW')
        self.assertEqual(len(self.w.current_result), 2)
        self.assertEqual(self.w.page_output.tableTab.stackedWidget.count(), 2)

    def test_export_snapshot_keeps_later_edits_unsaved(self):
        self.install([result('snapshot.wav')])
        model = self.w.page_output.tableTab.stackedWidget.widget(0).model()
        model.setData(model.index(0, 3), 'SAVED_TEXT', Qt.ItemDataRole.EditRole)
        self.w.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(DEST / 'snapshot'))
        self.w.page_output.combox_output_format.setCurrentText('SRT')
        self.w.page_output.outputSubtitleFileButton.click()
        worker = self.w.outputWorker
        model.setData(model.index(0, 3), 'LATER_EDIT', Qt.ItemDataRole.EditRole)
        self.wait(worker)
        exported = next((DEST / 'snapshot').glob('*.srt')).read_text(encoding='utf-8-sig')
        self.assertIn('SAVED_TEXT', exported)
        self.assertNotIn('LATER_EDIT', exported)
        self.assertTrue(model.isModified)
        self.w.page_output.outputSubtitleFileButton.click()
        self.wait(self.w.outputWorker)
        self.assertFalse(model.isModified)

    def test_moved_tabs_refresh_matching_pages(self):
        self.install([result('refresh-a.wav'), result('refresh-b.wav')])
        tabs = self.w.page_output.tableTab
        tabs.tabBar.setCurrentIndex(0)
        tabs.tabBar._swapItem(1)
        new = result('refresh-b.wav', ('UPDATED_B',))
        self.w.current_result = [new]
        self.w.showResultInTable([new])
        self.assertEqual(tabs.stackedWidget.count(), 1)
        self.assertEqual(tabs.stackedWidget.widget(0).objectName(), tabs.tabBar.tabItem(0).routeKey())
        self.assertEqual(tabs.stackedWidget.widget(0).model().data(
            tabs.stackedWidget.widget(0).model().index(0, 3), Qt.ItemDataRole.DisplayRole), 'UPDATED_B')

    def test_split_retry_and_late_finished_callback(self):
        self.install([result('retry.wav')])
        self.w.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(DEST / 'split-retry'))
        for returncode in [1, 0]:
            with patch('faster_whisper_GUI.tasks.audio_split.subprocess.run',
                       return_value=NS(returncode=returncode, stderr='simulated failure' if returncode else '')):
                self.w.page_output.outputAudioPartWithSpeakerButton.click()
                worker = self.w.splitAudioFileWithSpeakerWorker
                self.wait(worker)
            self.assertIsNone(self.w.splitAudioFileWithSpeakerWorker)
            self.assertTrue(self.w.page_output.outputSubtitleFileButton.isEnabled())
        self.assertTrue(any(n[0] == 'success' and n[1] == '分割音频完成' for n in self.w.notices))
        self.w.setPageOutButtonStatus(False)
        self.w.splitAudioFileWithSpeakerWorkerFinished()
        self.assertFalse(self.w.page_output.outputSubtitleFileButton.isEnabled())

    def test_split_stop_before_start_does_not_run_ffmpeg(self):
        from faster_whisper_GUI.tasks.audio_split import SplitAudioFileWithSpeakersWorker
        worker = SplitAudioFileWithSpeakersWorker([result('cancel-before-start.wav')], str(DEST / 'prestart'))
        worker.stop()
        with patch('faster_whisper_GUI.tasks.audio_split.subprocess.run') as run:
            worker.run()
            run.assert_not_called()
        self.assertTrue(worker.cancelled)

    def test_split_invalid_output_directory_does_not_lock_buttons(self):
        self.install([result('invalid-dir.wav')])
        parent_file = DEST / 'not-a-directory'
        parent_file.write_text('fixture', encoding='utf-8')
        self.w.page_output.outputGroupWidget.LineEdit_output_dir.setText(str(parent_file / 'child'))
        self.w.page_output.outputAudioPartWithSpeakerButton.click()
        self.assertIsNone(self.w.splitAudioFileWithSpeakerWorker)
        self.assertTrue(self.w.page_output.outputSubtitleFileButton.isEnabled())
        self.assertTrue(any(n[0] == 'error' for n in self.w.notices))

if __name__ == '__main__':
    unittest.main(verbosity=2)
