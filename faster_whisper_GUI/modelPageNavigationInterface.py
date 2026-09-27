# coding:utf-8

from PySide6.QtCore import (QCoreApplication, Qt)
from PySide6.QtGui import QFont

from PySide6.QtWidgets import (
                                QCompleter, 
                                QGridLayout, 
                                QHBoxLayout, 
                                QLabel,
                                QStyle,
                                QVBoxLayout,
                                QSplitter
                            )

from qfluentwidgets import (
                            ComboBox, 
                            RadioButton, 
                            PushButton, 
                            ToolButton, 
                            EditableComboBox, 
                            LineEdit ,
                            SwitchButton,
                            FluentIcon,
                            PrimaryPushButton,
                            
                        )

from .navigationInterface import NavigationBaseInterface
from .paramItemWidget import ParamWidget
from .style_sheet import StyleSheet

from .config import (
                    Preciese_list
                    , Model_names
                    , LEGACY_DEVICE_LIST
                    , deviceComboItems
                )

from .util import isROCmAvailable

class ModelNavigationInterface(NavigationBaseInterface):
    def __tr(self, text):
        return QCoreApplication.translate(self.__class__.__name__, text)

    def __init__(self, parent=None):
        
        super().__init__(
                        title=self.__tr("Model")
                        , subtitle=self.__tr('加载本地模型或下载模型')
                        , parent=parent
                    )
        
        self.model_names = Model_names

        # 探测到 ROCm/HIP 运行时时才加入 "AMD ROCm" 选项
        self.device_items = deviceComboItems(include_rocm=isROCmAvailable())
        self.device_list = [text for text, _ in self.device_items]
        self.preciese_list = Preciese_list
        
        self.setObjectName('modelNavigationInterface')
        self.setupUI()

        # StyleSheet.MODELLOAD.apply(self.button_model_lodar)

        self.SignalAndSlotConnect()
    
    def setDevice(self, device):
        """
        设置下拉框选中的设备

        兼容两种配置格式：
          - 新格式：实际设备名，如 "cpu" / "cuda" / "auto"（ROCm 与 CUDA 共用 "cuda"）
          - 旧格式：下拉框索引整数，如 0 / 1 / 2（旧顺序见 config.LEGACY_DEVICE_LIST）

        若配置中的设备在当前下拉框中不存在（例如配置里选了 ROCm，但本次启动未探测到 HIP SDK），
        则回退到 "auto"，避免静默选中一个并非用户所选的设备。
        """
        device_combox = self.device_combox

        if isinstance(device, str):
            # 按实际设备值匹配
            # 注意 "cuda" 可能同时对应 "cuda" 与 "AMD ROCm (HIP)" 两项，findData 返回先出现的普通 cuda
            index = device_combox.findData(device)
            if index >= 0:
                device_combox.setCurrentIndex(index)
                return
            print(f"[Device] 配置中的设备 {device!r} 当前不可用，回退到 auto")
            fallback = device_combox.findData("auto")
            device_combox.setCurrentIndex(fallback if fallback >= 0 else 1)
            return

        # 旧格式：索引。先按旧顺序映射，映射不到再按当前列表长度夹取
        if isinstance(device, int):
            if 0 <= device < len(LEGACY_DEVICE_LIST):
                index = device_combox.findData(LEGACY_DEVICE_LIST[device])
                if index >= 0:
                    device_combox.setCurrentIndex(index)
                    return
            device_combox.setCurrentIndex(device if 0 <= device < device_combox.count() else 1)
            return

        print(f"[Device] 无法识别的 device 配置: {device!r}，回退到默认")
        device_combox.setCurrentIndex(1)

    def SignalAndSlotConnect(self):
        self.model_local_RadioButton.clicked.connect(self.setModelLocationLayout)
        self.model_online_RadioButton.clicked.connect(self.setModelLocationLayout)
        
    def setModelLocationLayout(self):
        num_widgets_layout = self.hBoxLayout_local_model.count()
            
        for i in range(num_widgets_layout):
            widget = self.hBoxLayout_local_model.itemAt(i).widget()
            widget.setEnabled(self.model_local_RadioButton.isChecked())
        
        num_widgets_layout = self.hBoxLayout_online_model.count()
            
        for i in range(num_widgets_layout):
            widget = self.hBoxLayout_online_model.itemAt(i).widget()
            widget.setEnabled(self.model_online_RadioButton.isChecked())

    def setupUI(self):
        self.layout_button_model_lodar = QVBoxLayout()
        # self.button_model_lodar = PushButton()
        self.button_model_lodar = PrimaryPushButton(self)
        
        self.button_model_lodar.setText(self.__tr("加载模型"))
        self.button_model_lodar.setFixedHeight(65)
        self.button_model_lodar.setFixedWidth(195)
        font = QFont("Segoe UI", 15)
        font.setBold(True)

        self.button_model_lodar.setFont(font)
        
        self.button_model_lodar.setIcon(FluentIcon.PLAY)
        self.button_model_lodar.setObjectName("buttonModelLodar")
        
        
        self.layout_button_model_lodar.addWidget(self.button_model_lodar)
        self.layout_button_model_lodar.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.addLayout(self.layout_button_model_lodar)


        # ==========================================================================================================

        model_local_RadioButton = RadioButton()
        model_local_RadioButton.setChecked(True)
        model_local_RadioButton.setText(self.__tr("使用本地模型"))
        model_local_RadioButton.setToolTip(self.__tr("本地模型需使用经过 CTranslate2 转换工具，从 OpenAI 模型格式转换而来的模型"))
        self.model_local_RadioButton = model_local_RadioButton
        self.addWidget(self.model_local_RadioButton)

        # ==========================================================================================================

        self.hBoxLayout_local_model = QHBoxLayout()
        self.addLayout(self.hBoxLayout_local_model)

        # 使用本地模型时添加相关控件到布局
        self.label_model_path = QLabel()
        self.label_model_path.setText(self.__tr("模型目录"))
        self.label_model_path.setObjectName("LabelModelPath")
        self.label_model_path.setStyleSheet("#LabelModelPath{ background : rgba(0, 128, 0, 120); }")
        self.lineEdit_model_path = LineEdit()
        self.lineEdit_model_path.setClearButtonEnabled(True)
        self.toolPushButton_get_model_path = ToolButton()
        self.toolPushButton_get_model_path.setIcon(self.style().standardPixmap(QStyle.StandardPixmap.SP_DirOpenIcon))

        self.hBoxLayout_local_model.addWidget(self.label_model_path)
        self.hBoxLayout_local_model.addWidget(self.lineEdit_model_path)
        self.hBoxLayout_local_model.addWidget(self.toolPushButton_get_model_path)
        
        model_online_RadioButton = RadioButton()
        model_online_RadioButton.setChecked(False)
        model_online_RadioButton.setText(self.__tr("在线下载模型"))
        model_online_RadioButton.setToolTip(self.__tr("下载可能会花费很长时间，具体取决于网络状态，\n作为参考 large-v2 模型下载量最大约 6GB"))
        self.model_online_RadioButton = model_online_RadioButton
        self.addWidget(model_online_RadioButton)

        self.hBoxLayout_online_model = QHBoxLayout()
        self.addLayout(self.hBoxLayout_online_model)

        # 添加一些控件到布局中

        # ==========================================================================================================
        self.label_online_model_name = QLabel()    
        self.label_online_model_name.setText(self.__tr("模型名称"))
        self.label_online_model_name.setObjectName("LabelOnlineModelName")
        self.label_online_model_name.setStyleSheet("#LabelOnlineModelName{ background : rgba(0, 128, 0, 120); }")

        self.combox_online_model = EditableComboBox()
        # 下拉框设置项目
        self.combox_online_model.addItems(self.model_names)
        # 下拉框设置自动完成
        completer = QCompleter(self.model_names, self)
        self.combox_online_model.setCompleter(completer)
        self.combox_online_model.setCurrentIndex(0)
        self.hBoxLayout_online_model.addWidget(self.label_online_model_name)
        self.hBoxLayout_online_model.addWidget(self.combox_online_model)

        self.setModelLocationLayout()

        GridLayout_model_param = QGridLayout()
        GridLayout_model_param.setAlignment(Qt.AlignmentFlag.AlignTop)
        GridLayout_model_param.setContentsMargins(0, 0, 0, 0)
        GridLayout_model_param.setSpacing(0)
        self.addLayout(GridLayout_model_param)
        GridLayout_model_param_widgets_list = []

        # ===================================================================================================================================================================================================
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 是否使用 v3 模型
        self.switchButton_use_v3 = SwitchButton()
        self.switchButton_use_v3.setChecked(False)
        self.switchButton_use_v3.setObjectName("SwitchButtonUseV3")
        self.switchButton_use_v3.setOnText(self.__tr("v3 模型"))
        self.switchButton_use_v3.setOffText(self.__tr("非 V3 模型"))

        self.paramItemWidget_use_v3 = ParamWidget(self.__tr("使用 v3 模型"), self.__tr("如果使用 V3 模型开启该选项，以修正梅尔滤波器组的大小，不使用 V3 模型请不要开启") ,self.switchButton_use_v3)

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_use_v3)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 设备
        device_combox  = ComboBox()
        for device_text, device_value in self.device_items:
            # 显示文本与「传给 CTranslate2 的实际设备名」分离：
            # "AMD ROCm (HIP)" 显示给用户看，实际值是 "cuda"（HIP 后端沿用 CUDA 的 API 命名）
            device_combox.addItem(self.__tr(device_text), userData=device_value)
        device_combox.setCurrentIndex(1)
        device_combox.setToolTip(self.__tr(
            "选择运行语音识别的设备。\n"
            "cpu：使用 CPU\n"
            "cuda：使用 NVIDIA CUDA\n"
            "auto：自动选择\n"
            "AMD ROCm (HIP)：使用 AMD 显卡，需要已安装 HIP SDK"))
        self.device_combox = device_combox
        
        self.paramItemWidget_device = ParamWidget(self.__tr("处理设备"), self.__tr("选择运行语音识别的设备。"), device_combox)
        GridLayout_model_param_widgets_list.append(self.paramItemWidget_device)   

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        LineEdit_device_index = LineEdit()
        LineEdit_device_index.setText("0")
        LineEdit_device_index.setToolTip(self.__tr("要使用的设备ID。也可以通过传递ID列表(例如0,1,2,3)在多GPU上加载模型。"))
        self.LineEdit_device_index = LineEdit_device_index

        self.paramItemWidget_device_index = ParamWidget(self.__tr("设备号"), self.__tr("要使用的设备ID。也可以通过传递ID列表(例如0,1,2,3)在多GPU上加载模型。"), LineEdit_device_index)

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_device_index) 

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 计算精度
        preciese_combox  = EditableComboBox()
        preciese_combox.addItems(self.preciese_list)
        preciese_combox.setCurrentIndex(5)
        preciese_combox.setCompleter(QCompleter(self.preciese_list))
        preciese_combox.setToolTip(self.__tr("要使用的计算精度，尽管某些设备不支持半精度，\n但事实上不论选择什么精度类型都可以隐式转换。\n请参阅 https://opennmt.net/CTranslate2/quantization.html。"))
        self.preciese_combox = preciese_combox

        self.paramItemWidget_preciese = ParamWidget(self.__tr("计算精度"), self.__tr("要使用的计算精度，尽管某些设备不支持半精度，\n但事实上不论选择什么精度类型都可以隐式转换。\n请参阅 https://opennmt.net/CTranslate2/quantization.html。"), preciese_combox)

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_preciese)   

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        LineEdit_cpu_threads = LineEdit()
        LineEdit_cpu_threads.setText("4")
        LineEdit_cpu_threads.setToolTip(self.__tr("在CPU上运行时使用的线程数(默认为4)。非零值会覆盖"))
        self.LineEdit_cpu_threads = LineEdit_cpu_threads

        self.paramItemWidget_cpu_threads = ParamWidget(self.__tr("线程数（CPU）"), self.__tr("在CPU上运行时使用的线程数(默认为4)。非零值会覆盖"), LineEdit_cpu_threads)  

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_cpu_threads)   
        
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        LineEdit_num_workers = LineEdit()
        LineEdit_num_workers.setText("1")
        LineEdit_num_workers.setToolTip(self.__tr("具有多个工作线程可以在运行模型时实现真正的并行性。\n这可以以增加内存使用为代价提高整体吞吐量。"))
        self.LineEdit_num_workers = LineEdit_num_workers

        self.paramItemWidget_num_workers = ParamWidget(self.__tr("并发数"), self.__tr("具有多个工作线程可以在运行模型时实现真正的并行性。\n这可以以增加内存使用为代价提高整体吞吐量。"), LineEdit_num_workers)

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_num_workers)
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        button_download_root = PushButton()
        button_download_root.setText(self.__tr("下载缓存目录"))
        
        self.LineEdit_download_root = LineEdit()
        self.LineEdit_download_root.setToolTip(self.__tr("模型下载保存的目录。如果未修改,\n则模型将保存在标准Hugging Face缓存目录中。"))
        
        self.LineEdit_download_root = self.LineEdit_download_root
        self.button_download_root = button_download_root

        self.paramItemWidget_download_root = ParamWidget(self.__tr("下载缓存目录"), self.__tr("模型下载保存的目录。如果未修改,\n则模型将保存在标准Hugging Face缓存目录中。"), self.button_download_root)
        self.paramItemWidget_download_root.addwidget(self.LineEdit_download_root)

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_download_root) 
        
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        self.switchButton_local_files_only = SwitchButton()
        self.switchButton_local_files_only.setChecked(False)
        self.switchButton_local_files_only.setOnText(self.__tr("使用"))
        self.switchButton_local_files_only.setOffText(self.__tr("不使用"))
        self.paramItemWidget_local_files_only = ParamWidget(self.__tr("是否使用本地缓存"), self.__tr("如果为True，在本地缓存的文件存在时返回其路径，不再重新下载文件。"), self.switchButton_local_files_only)

        GridLayout_model_param_widgets_list.append(self.paramItemWidget_local_files_only)  
        
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for i,item in enumerate(GridLayout_model_param_widgets_list):
            GridLayout_model_param.addWidget(item, i,0)
        
        # ==================================================================================================================================================================================================
        hBoxLayout_model_convert = QHBoxLayout()

        # TODO:暂时移除转换模型功能,相关接口需要维护
        # self.addLayout(hBoxLayout_model_convert)

        self.button_set_model_out_dir =  PushButton()
        self.button_set_model_out_dir.setText(self.__tr("模型输出目录"))
        hBoxLayout_model_convert.addWidget(self.button_set_model_out_dir)
        self.button_set_model_out_dir.setEnabled(False)

        self.LineEdit_model_out_dir = LineEdit()
        self.LineEdit_model_out_dir.setToolTip(self.__tr("转换模型的保存目录，不会自动创建子目录"))
        hBoxLayout_model_convert.addWidget(self.LineEdit_model_out_dir)
        self.LineEdit_model_out_dir.setEnabled(False)

        self.button_convert_model = PushButton()
        self.button_convert_model.setText(self.__tr("下载并转换模型"))
        self.button_convert_model.setToolTip(self.__tr("转换 OpenAi 模型到本地格式，\n必须选择在线模型"))
        hBoxLayout_model_convert.addWidget(self.button_convert_model)
        self.button_convert_model.setEnabled(False)

        # self.setViewportMargins(0, self.toolBar.height(), 0, 216)
        # self.LineEdit_download_root.setText(self.parent().download_cache_path)

        
    def setParam(self, param:dict):
        self.model_local_RadioButton.setChecked(param["localModel"])
        self.model_online_RadioButton.setChecked(param["onlineModel"])

        self.setModelLocationLayout()

        self.lineEdit_model_path.setText(param["model_path"])
        self.combox_online_model.setCurrentIndex(param["modelName"])

        self.switchButton_use_v3.setChecked(param["use_v3_model"])
        self.setDevice(param["device"])
        self.LineEdit_device_index.setText(param["deviceIndex"])
        self.preciese_combox.setCurrentIndex(param["preciese"])
        self.LineEdit_cpu_threads.setText(param["thread_num"])
        self.LineEdit_num_workers.setText(param["num_worker"])
        self.LineEdit_download_root.setText(param["download_root"])
        self.switchButton_local_files_only.setChecked(param["local_files_only"] )

    def getParam(self):
        param = {}
        param["localModel"] = self.model_local_RadioButton.isChecked()
        param["onlineModel"] = self.model_online_RadioButton.isChecked()
        param["model_path"] = self.lineEdit_model_path.text()
        param["modelName"] = self.combox_online_model.currentIndex()
        param["use_v3_model"] = self.switchButton_use_v3.isChecked()
        # 保存「实际设备名」而非下拉框索引：新增/隐藏设备选项时不会让旧配置错位
        param["device"] = self.device_combox.currentData()
        param["deviceIndex"] = self.LineEdit_device_index.text().strip()
        param["preciese"] = self.preciese_combox.currentIndex()
        param["thread_num"] = self.LineEdit_cpu_threads.text().strip()
        param["num_worker"] = self.LineEdit_num_workers.text().strip()
        param["download_root"] = self.LineEdit_download_root.text().strip()
        param["local_files_only"] = self.switchButton_local_files_only.isChecked()

        return param
        
