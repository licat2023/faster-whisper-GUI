"""跟随应用主题更新标题文字颜色。"""

from qfluentwidgets import isDarkTheme, qconfig
from qframelesswindow import StandardTitleBar


class WindowTitleBar(StandardTitleBar):
    def __init__(self, parent):
        super().__init__(parent)
        self._titleStyle = self.titleLabel.styleSheet()
        qconfig.themeChangedFinished.connect(self._updateTitleColor)
        self._updateTitleColor()

    def _updateTitleColor(self):
        color = 'white' if isDarkTheme() else 'black'
        self.titleLabel.setStyleSheet(
            self._titleStyle + f'\nQLabel {{ color: {color}; }}')
