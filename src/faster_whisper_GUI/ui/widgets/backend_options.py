"""Backend-specific editable options, kept separate from legacy Whisper controls."""
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel
from qfluentwidgets import LineEdit, ComboBox, qconfig, isDarkTheme
import math
from faster_whisper_GUI.backends.catalog import default_settings, CPP_EXECUTABLES

MODEL_FIELDS = {
 'whisper.cpp': [('model', 'ggml 模型文件'), ('executable', 'whisper-cli 程序'), ('device', '设备', ['cpu', 'vulkan', 'rocm']), ('threads', 'CPU 线程数')],
 'qwen': [('model', 'ASR 模型目录或名称'), ('aligner', '对齐模型目录或名称'), ('interpreter', 'Qwen Python 环境'),
          ('device', '设备', ['cuda', 'cpu']), ('max_new_tokens', '最大生成令牌数')],
 'sherpa': [('model', 'Zipformer 模型目录'), ('interpreter', 'sherpa Python 环境'), ('threads', 'CPU 线程数')],
}
RECOGNITION_FIELDS = {
 'whisper.cpp': [('language', '语言代码（空白自动）'), ('beam_size', '搜索宽度'), ('best_of', '候选数'), ('initial_prompt', '初始提示词')],
 'qwen': [('language', '语言代码（空白自动）'), ('max_segment_chars', '每段字幕字数'), ('max_segment_seconds', '每段字幕最长秒数')],
 'sherpa': [('language', '模型语言代码'), ('max_segment_chars', '每段字幕字数'), ('max_segment_seconds', '每段字幕最长秒数')],
}


class BackendOptions(QWidget):
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.fields = fields
        self.layout_options = QFormLayout(self)
        self.controls = {}
        self.saved = {}
        self.identity = None
        qconfig.themeChanged.connect(self._applyTheme)

    def select(self, identity):
        if self.identity:
            self.saved[self.identity] = self.values()
        while self.layout_options.count():
            item = self.layout_options.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.identity = identity
        self.controls = {}
        values = {**default_settings(identity), **self.saved.get(identity, {})}
        for field in self.fields.get(identity, []):
            key, label, *choices = field
            if choices:
                control = ComboBox(self)
                control.addItems(choices[0])
                control.setCurrentText(str(values.get(key, choices[0][0])))
            else:
                control = LineEdit(self)
                control.setText(str(values.get(key, '')))
            control.setObjectName(f'backend_{key}')
            self.controls[key] = control
            self.layout_options.addRow(label, control)
        if identity == 'qwen':
            self.layout_options.addRow(QLabel('识别后自动运行 ForcedAligner，生成可编辑、可导出的字幕。', self))
        if identity == 'sherpa':
            self.layout_options.addRow(QLabel('支持原生流式录音；语言由模型决定。字幕采用原生 token 时间，不提供词尾对齐。', self))
        if identity == 'whisper.cpp' and 'device' in self.controls:
            self.controls['device'].currentTextChanged.connect(self._cppDeviceChanged)
        self.setVisible(identity != 'faster-whisper')
        self._applyTheme()

    def _applyTheme(self, *_):
        color = '#f0f0f0' if isDarkTheme() else '#202020'
        for label in self.findChildren(QLabel):
            label.setStyleSheet(f'color: {color}; background: transparent;')

    def values(self):
        return {key: control.currentText() if isinstance(control, ComboBox) else control.text().strip()
                for key, control in self.controls.items()}

    def _cppDeviceChanged(self, device):
        executable = self.controls['executable']
        if executable.text() in CPP_EXECUTABLES.values():
            executable.setText(CPP_EXECUTABLES.get(device, CPP_EXECUTABLES['vulkan']))

    def settings(self):
        values = {**default_settings(self.identity), **self.values()}
        for key in ['threads', 'beam_size', 'best_of', 'max_new_tokens', 'max_segment_chars', 'segment_size', 'overlap']:
            if key in values:
                try:
                    values[key] = int(values[key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{key} 必须为整数") from error
                if values[key] < 1:
                    raise ValueError(f'{key} 必须大于 0')
        if 'max_segment_seconds' in values:
            try:
                values['max_segment_seconds'] = float(values['max_segment_seconds'])
            except (TypeError, ValueError) as error:
                raise ValueError('字幕时长必须为数字') from error
            if not math.isfinite(values['max_segment_seconds']) or values['max_segment_seconds'] <= 0:
                raise ValueError('字幕时长必须大于 0')
        values['language'] = values.get('language') or None
        return values

    def restore(self, saved):
        self.saved.update(saved or {})
        current = self.identity
        self.identity = None
        self.select(current)

    def snapshot(self):
        return {**self.saved, self.identity: self.values()}
