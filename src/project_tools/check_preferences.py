"""YAML configuration storage and GUI persistence checks."""

import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class StorageChecks(unittest.TestCase):
    def test_personal_configuration_and_write_temporaries_are_ignored(self):
        root = Path(__file__).resolve().parents[2]
        for path in ('config/config.yml', 'config/.config-interrupted.tmp'):
            result = subprocess.run(['git', 'check-ignore', '--quiet', path], cwd=root)
            self.assertEqual(result.returncode, 0)

    def test_roundtrip_keeps_unicode_paths_text_and_token(self):
        from faster_whisper_GUI.runtime.preferences import load_config, save_config
        data = {'ui': {'theme': 'dark', 'theme_color': '#12abcd', 'font_families': ['微软雅黑']},
                'model': {'model_path': 'D:/模型/whisper'},
                'transcription': {'initial_prompt': 'yes', 'prefix': '2026-10-26'},
                'settings': {'huggingface_user_token': 'dummy-token', 'saveConfig': False}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nested/config.yml'
            save_config(data, path)
            self.assertEqual(load_config(path), data)
            self.assertIn('微软雅黑', path.read_text(encoding='utf-8'))

    def test_invalid_documents_fail_without_exposing_values(self):
        from faster_whisper_GUI.runtime.preferences import ConfigError, load_config
        samples = ['[1, 2]', 'model: []', 'ui: {theme: purple}',
                   'ui: {font_families: text}', 'ui: {theme_color: invalid}',
                   'settings: 1\nsettings: 2', 'obsolete: {}',
                   '!!python/object/apply:os.system [echo unsafe]',
                   'settings: {huggingface_user_token: "dummy-private-token']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yml'
            for value in samples:
                with self.subTest(value=value):
                    path.write_text(value, encoding='utf-8')
                    with self.assertRaises(ConfigError) as raised:
                        load_config(path)
                    self.assertNotIn('dummy-private-token', str(raised.exception))

    def test_manual_strings_are_not_implicitly_booleans_or_dates(self):
        from faster_whisper_GUI.runtime.preferences import load_config
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yml'
            path.write_text('transcription: {initial_prompt: yes, prefix: 2026-10-26}\nsettings: {saveConfig: false}', encoding='utf-8')
            result = load_config(path)
            self.assertEqual(result['transcription']['initial_prompt'], 'yes')
            self.assertEqual(result['transcription']['prefix'], '2026-10-26')
            self.assertIs(result['settings']['saveConfig'], False)

    def test_failed_replace_preserves_previous_configuration(self):
        from faster_whisper_GUI.runtime.preferences import ConfigError, load_config, save_config
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yml'
            save_config({'model': {'device': 'cpu'}}, path)
            with patch('faster_whisper_GUI.runtime.preferences.os.replace', side_effect=OSError('blocked')):
                with self.assertRaises(ConfigError):
                    save_config({'model': {'device': 'rocm'}}, path)
            self.assertEqual(load_config(path), {'model': {'device': 'cpu'}})
            self.assertEqual(list(Path(folder).iterdir()), [path])


class GuiChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        with patch('faster_whisper_GUI.runtime.preferences.load_config', return_value={}):
            from faster_whisper_GUI.ui import translation
        from faster_whisper_GUI import logging_setup
        destination = Path(__file__).resolve().parents[2] / '.cache/checks/config-yaml/gui'
        destination.mkdir(parents=True, exist_ok=True)
        cls.log_patch = patch.multiple(logging_setup, resolveLogDir=lambda: destination,
                                      pruneOldRuns=lambda *args, **kwargs: [])
        cls.log_patch.start()
        cls.addClassCleanup(cls.log_patch.stop)

    def test_language_selection_reads_the_yaml_settings(self):
        import importlib
        from faster_whisper_GUI.ui import translation
        try:
            for index, expected in ((0, 'zh'), (1, 'en')):
                with patch('faster_whisper_GUI.runtime.preferences.load_config',
                           return_value={'settings': {'language': index}}):
                    self.assertEqual(importlib.reload(translation).language, expected)
        finally:
            with patch('faster_whisper_GUI.runtime.preferences.load_config', return_value={}):
                importlib.reload(translation)

    def test_appearance_and_color_controls_do_not_autosave(self):
        from qfluentwidgets import qconfig
        from faster_whisper_GUI.ui.appearance import apply_appearance, appearance_settings
        from faster_whisper_GUI.ui.pages.settings import SettingPageNavigationInterface
        from faster_whisper_GUI.ui.widgets.navigation import ToolBar
        original = appearance_settings()
        try:
            with patch.object(qconfig, 'save', side_effect=AssertionError('framework autosave')) as autosave:
                apply_appearance({'theme': 'dark', 'theme_color': '#80123456',
                                  'font_families': ['Microsoft YaHei', 'Segoe UI']})
                self.assertEqual(appearance_settings()['theme_color'], '#80123456')
                page = SettingPageNavigationInterface()
                page.setColorAndThemeColorRandom()
                page.setThemeColorWithLineEditText('#123456')
                page.setThemeColorAndText()
                toolbar = ToolBar('Test', '')
                toolbar.themeButton.click()
                autosave.assert_not_called()
                self.assertNotIn('themeColor', page.getParam())
                page.deleteLater()
                toolbar.deleteLater()
        finally:
            apply_appearance(original)

    def test_real_window_roundtrip_preserves_all_sections_and_appearance(self):
        from faster_whisper_GUI.runtime.preferences import load_config
        from faster_whisper_GUI.ui.appearance import apply_appearance, appearance_settings
        from faster_whisper_GUI.ui.window.main import MainWindows
        from faster_whisper_GUI.ui.window.view import UIMainWin

        class Window(MainWindows):
            def readConfig(self, path=None):
                return UIMainWin.readConfig(self)
            def onModelLoadClicked(self):
                pass
            def closeEvent(self, event):
                event.accept()

        original = appearance_settings()
        window = Window()
        try:
            window.page_setting.switchButton_autoLoadModel.setChecked(False)
            window.page_setting.LineEdit_use_auth_token.setText('dummy-token')
            window.page_model.combox_online_model.setText('custom-model')
            apply_appearance({'theme': 'dark', 'theme_color': '#ff123456', 'font_families': ['微软雅黑']})
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.yml'
                window.saveConfig(path)
                data = load_config(path)
                self.assertEqual(set(data), {'ui', 'model', 'transcription', 'vad', 'separation', 'settings', 'output'})
                self.assertEqual(data['ui']['font_families'], ['微软雅黑'])
                self.assertNotIn('themeColor', data['settings'])
                self.assertNotIn('modelName', data['model'])
                self.assertNotIn('preciese', data['model'])
                self.assertEqual(data['settings']['huggingface_user_token'], 'dummy-token')
                apply_appearance({'theme': 'light', 'theme_color': '#ff654321'})
                window.page_model.combox_online_model.setText('changed')
                self.assertTrue(UIMainWin.readConfig(window, path))
                window.setConfig()
                self.assertEqual(window.page_model.combox_online_model.currentText(), 'custom-model')
                self.assertEqual(appearance_settings(), data['ui'])
                window.saveConfig(path)
                self.assertEqual(load_config(path), data)
                path.write_text('ui: []', encoding='utf-8')
                before = window.preferences
                self.assertFalse(UIMainWin.readConfig(window, path))
                self.assertIs(window.preferences, before)
                path.write_text('settings: {saveConfig: "false"}', encoding='utf-8')
                self.assertFalse(UIMainWin.readConfig(window, path))
                self.assertIs(window.preferences, before)
        finally:
            window.close()
            self.app.processEvents()
            apply_appearance(original)

    def test_backend_options_survive_loading_into_a_fresh_window(self):
        from faster_whisper_GUI.ui.window.main import MainWindows
        from faster_whisper_GUI.ui.window.view import UIMainWin
        from faster_whisper_GUI.ui.window.settings import collect_config

        class Window(MainWindows):
            def readConfig(self, path=None):
                return UIMainWin.readConfig(self)
            def onModelLoadClicked(self):
                pass
            def closeEvent(self, event):
                event.accept()

        first, second = Window(), Window()
        try:
            for identity in ('whisper.cpp', 'sherpa', 'qwen'):
                first.page_model.backend_combox.setCurrentIndex(first.page_model.backend_combox.findData(identity))
                first.page_model.backend_options.controls['model'].setText('custom-' + identity)
            first.page_demucs.backend_combox.setCurrentIndex(1)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.yml'
                first.saveConfig(path)
                self.assertTrue(UIMainWin.readConfig(second, path))
                second.setConfig()
                self.assertEqual(collect_config(second), collect_config(first))
        finally:
            first.close()
            second.close()
            self.app.processEvents()

    def test_invalid_startup_configuration_is_preserved_without_autosave(self):
        from faster_whisper_GUI.ui.window.view import UIMainWin
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yml'
            contents = 'settings: {saveConfig: "invalid"}'
            path.write_text(contents, encoding='utf-8')
            with patch('faster_whisper_GUI.ui.window.view.CONFIG_PATH', path):
                window = UIMainWin()
                try:
                    self.assertFalse(window.page_setting.switchButton_saveConfig.isChecked())
                    self.assertEqual(path.read_text(encoding='utf-8'), contents)
                finally:
                    window.close()
                    self.app.processEvents()


if __name__ == '__main__':
    unittest.main(verbosity=2)
