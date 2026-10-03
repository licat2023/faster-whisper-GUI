"""验证标题栏在启动和主题切换时的文字颜色。"""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import Theme, qconfig, setTheme

from faster_whisper_GUI.ui.widgets.titlebar import WindowTitleBar


class TitleBarChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.originalTheme = qconfig.get(qconfig.themeMode)
        self.window = QWidget()

    def tearDown(self):
        self.window.deleteLater()
        self.app.processEvents()
        setTheme(self.originalTheme, save=False)

    def assertTitleColor(self, bar, expected):
        self.app.processEvents()
        bar.titleLabel.ensurePolished()
        self.assertEqual(
            bar.titleLabel.palette().color(QPalette.WindowText).name(), expected)

    def test_initial_theme(self):
        for theme, color in ((Theme.LIGHT, '#000000'), (Theme.DARK, '#ffffff')):
            with self.subTest(theme=theme):
                setTheme(theme, save=False)
                bar = WindowTitleBar(self.window)
                self.assertTitleColor(bar, color)

    def test_live_theme_switches(self):
        setTheme(Theme.LIGHT, save=False)
        bar = WindowTitleBar(self.window)
        initialStyle = bar.titleLabel.styleSheet()
        for lazy in (True, False):
            for theme, color in ((Theme.DARK, '#ffffff'), (Theme.LIGHT, '#000000')):
                with self.subTest(theme=theme, lazy=lazy):
                    setTheme(theme, save=False, lazy=lazy)
                    self.assertTitleColor(bar, color)
                    self.assertIn("Segoe UI", bar.titleLabel.styleSheet())
        self.assertEqual(bar.titleLabel.styleSheet(), initialStyle)


if __name__ == '__main__':
    unittest.main()
