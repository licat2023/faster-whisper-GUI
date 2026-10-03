# coding:utf-8
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (
                                QHBoxLayout, 
                                QLabel, 
                                QVBoxLayout, 
                                QWidget
                            )

from qfluentwidgets import (DoubleSpinBox, PushButton, ComboBox, FluentIcon, LineEdit)
from faster_whisper_GUI.runtime.paths import CACHE_ROOT
from faster_whisper_GUI.ui.widgets.backend_options import BackendOptions

from faster_whisper_GUI.ui.widgets.navigation import NavigationBaseInterface
from faster_whisper_GUI.ui.widgets.file_list import FileNameListView
from faster_whisper_GUI.ui.widgets.output_path import OutputGroupWidget

from faster_whisper_GUI.ui.styles import StyleSheet

from faster_whisper_GUI.config import STEMS

class DemucsParamGroupWidget(QWidget):
    def __tr(self, text):
        return QCoreApplication.translate(self.__class__.__name__, text)
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.vBoxLayout = QVBoxLayout()
        self.setLayout(self.vBoxLayout)
        self.mainWidget = QWidget()
        self.mainWidget.setObjectName("mainObject")

        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)

        self.__tr_text_all_Stems = self.__tr("所有音轨")
        self.__tr_text_Vocals = self.__tr("仅人声")
        self.__tr_text_Other = self.__tr("仅其它")
        self.__tr_text_Bass = self.__tr("仅贝斯声")
        self.__tr_text_Drums = self.__tr("仅鼓点声")
        self.__tr_text_Vocals_Others = self.__tr("人声、背景声二分")

        self.__tr_text_list = [
            self.__tr_text_all_Stems,
            self.__tr_text_Vocals,
            self.__tr_text_Other,
            self.__tr_text_Bass,
            self.__tr_text_Drums,
            self.__tr_text_Vocals_Others
        ]

        self.setupUI()
    
    def addWidget(self, widget, alignment):
        self.mainLayout.addWidget(widget, alignment=alignment)
    
    def addLayout(self, layout):
        self.mainLayout.addLayout(layout)

    
    def setupUI(self):

        self.label_ = QLabel(self.__tr("Demucs 参数"))
        self.vBoxLayout.addWidget(self.label_)

        # =====================================================================

        self.vBoxLayout.addWidget(self.mainWidget)

        # =====================================================================
        self.param_hBoxLayout = QHBoxLayout()
        self.addLayout(self.param_hBoxLayout)
        self.param_hBoxLayout.setContentsMargins(10,10,10,10)
        
        self.vBoxLayout_overlap = QVBoxLayout()
        self.param_hBoxLayout.addLayout(self.vBoxLayout_overlap)

        self.spinBox_overlap = DoubleSpinBox()
        self.spinBox_overlap.setMaximum(1.0)
        self.spinBox_overlap.setMinimum(0.1)
        self.spinBox_overlap.setSingleStep(0.1)
        self.spinBox_overlap.setValue(0.10)
        self.spinBox_overlap.setDecimals(2)
        self.spinBox_overlap.setToolTip(self.tr("分段采样的段间重叠度，该值不宜过小，以免影响分段之间的分离结果的融合度"))
        
        self.label_overlap_param = QLabel(self.tr("采样重叠度"))

        self.vBoxLayout_overlap.addWidget(self.label_overlap_param)
        self.vBoxLayout_overlap.addWidget(self.spinBox_overlap)

        # =============================================================================
        self.vBoxLayout_segment = QVBoxLayout()
        self.param_hBoxLayout.addLayout(self.vBoxLayout_segment)

        self.spinBox_segment = DoubleSpinBox()
        self.spinBox_segment.setMaximum(100.0)
        self.spinBox_segment.setMinimum(1)
        self.spinBox_segment.setSingleStep(5)
        self.spinBox_segment.setValue(10)
        self.spinBox_segment.setDecimals(1)
        self.spinBox_segment.setToolTip(self.tr("Demucs 是一个极其消耗计算机内存和显存的模型，\n世界上没有任何计算机能够直接使用该模型处理长时音频数据\n因此数据将按照该值所示的秒数，分段处理，值越大将越消耗计算机资源，但效果可能会有提升"))

        self.label_segment_param = QLabel(self.tr("分段长度(s)"))

        self.vBoxLayout_segment.addWidget(self.label_segment_param)
        self.vBoxLayout_segment.addWidget(self.spinBox_segment)

        # =============================================================================
        self.vBoxLayout_stems = QVBoxLayout()
        self.param_hBoxLayout.addLayout(self.vBoxLayout_stems)

        self.comboBox_stems = ComboBox()
        for item in self.__tr_text_list:
            self.comboBox_stems.addItem(item)
        # self.comboBox_stems.addItems(STEMS)
        self.comboBox_stems.setCurrentIndex(1)

        self.label_stems_param = QLabel(self.tr("输出音轨"))
        
        self.vBoxLayout_stems.addWidget(self.label_stems_param)
        self.vBoxLayout_stems.addWidget(self.comboBox_stems)


class DemucsPageNavigation(NavigationBaseInterface):
    def __init__(self, parent=None):
        super().__init__(title="Demucs", subtitle=self.tr("使用 Demucs4.0 模型的 AVS（自动人声分离）方案"), parent=parent)
        # self.parent = parent

        self.setupUI()
        self.backend_combox = ComboBox(self)
        self.backend_combox.addItem('Demucs', userData='demucs')
        self.backend_combox.addItem('MelBand Roformer', userData='roformer')
        self.vBoxLayout.insertWidget(0, self.backend_combox)
        fields = {'roformer': [('model', '模型文件名（缓存目录）'), ('interpreter', 'Roformer Python 环境'),
                              ('segment_size', '窗口帧数'), ('overlap', '窗口重叠次数')]}
        self.roformer_options = BackendOptions(fields, self)
        self.roformer_options.saved['roformer'] = dict(model='vocals_mel_band_roformer.ckpt',
            interpreter=str(CACHE_ROOT / 'experiments/separation-env/Scripts/python.exe'), segment_size='64', overlap='2')
        self.vBoxLayout.insertWidget(1, self.roformer_options)
        self.backend_combox.currentIndexChanged.connect(self.selectBackend)
        self.selectBackend()

        StyleSheet.DENUCE_INTERFACE.apply(self.demucs_param_widget)

    def selectBackend(self, *_):
        identity = self.backend_combox.currentData()
        self.roformer_options.select(identity)
        self.roformer_options.setVisible(identity == 'roformer')
        self.demucs_param_widget.setVisible(identity == 'demucs')
        self.toolBar.titleLabel.setText('人声分离')
        self.toolBar.subtitleLabel.setText('Roformer 输出人声与伴奏两个音轨' if identity == 'roformer' else 'Demucs 四音轨分离')

    def setupUI(self):

        self.toolBar.modelStatusLabel.setVisible(False)

        self.fileListView = FileNameListView(self)
        self.addWidget(self.fileListView)

        # =============================================================================
        self.demucs_param_widget = DemucsParamGroupWidget(self)
        self.addWidget(self.demucs_param_widget)

        # =============================================================================
        self.outputGroupWidget = OutputGroupWidget(self)
        self.addWidget(self.outputGroupWidget)

        # =============================================================================
        self.process_button = PushButton()
        self.addWidget(self.process_button)
        self.process_button.setText(self.tr("提取"))
        self.process_button.setIcon(FluentIcon.IOT)

    def getParam(self) -> dict:
        param = {}
        param['backend'] = self.backend_combox.currentData()
        param['backend_options'] = self.roformer_options.snapshot()
        param["overlap"] = self.demucs_param_widget.spinBox_overlap.value()
        param["segment"]= self.demucs_param_widget.spinBox_segment.value()
        param["tracks"] = self.demucs_param_widget.comboBox_stems.currentIndex()

        return param

    def setParam(self, param: dict) -> None:
        param = {**self.getParam(), **param}
        self.roformer_options.restore(param.get('backend_options', {}))
        self.backend_combox.setCurrentIndex(max(0, self.backend_combox.findData(param.get('backend', 'demucs'))))

        self.demucs_param_widget.spinBox_overlap.setValue(param["overlap"] )
        self.demucs_param_widget.spinBox_segment.setValue(param["segment"])
        self.demucs_param_widget.comboBox_stems.setCurrentIndex(param["tracks"])
