# coding:utf-8
default_Huggingface_user_token = ""

Language_without_space = ["ja","zh","ko","yue"]
Language_dict = {
                "en": "english",
                "zht": "Traditional Chinese",
                "zhs": "Simplified Chinese ",
                "yue": "cantonese",
                "de": "german",
                "es": "spanish",
                "ru": "russian",
                "ko": "korean",
                "fr": "french",
                "ja": "japanese",
                "pt": "portuguese",
                "tr": "turkish",
                "pl": "polish",
                "ca": "catalan",
                "nl": "dutch",
                "ar": "arabic",
                "sv": "swedish",
                "it": "italian",
                "id": "indonesian",
                "hi": "hindi",
                "fi": "finnish",
                "vi": "vietnamese",
                "he": "hebrew",
                "uk": "ukrainian",
                "el": "greek",
                "ms": "malay",
                "cs": "czech",
                "ro": "romanian",
                "da": "danish",
                "hu": "hungarian",
                "ta": "tamil",
                "no": "norwegian",
                "th": "thai",
                "ur": "urdu",
                "hr": "croatian",
                "bg": "bulgarian",
                "lt": "lithuanian",
                "la": "latin",
                "mi": "maori",
                "ml": "malayalam",
                "cy": "welsh",
                "sk": "slovak",
                "te": "telugu",
                "fa": "persian",
                "lv": "latvian",
                "bn": "bengali",
                "sr": "serbian",
                "az": "azerbaijani",
                "sl": "slovenian",
                "kn": "kannada",
                "et": "estonian",
                "mk": "macedonian",
                "br": "breton",
                "eu": "basque",
                "is": "icelandic",
                "hy": "armenian",
                "ne": "nepali",
                "mn": "mongolian",
                "bs": "bosnian",
                "kk": "kazakh",
                "sq": "albanian",
                "sw": "swahili",
                "gl": "galician",
                "mr": "marathi",
                "pa": "punjabi",
                "si": "sinhala",
                "km": "khmer",
                "sn": "shona",
                "yo": "yoruba",
                "so": "somali",
                "af": "afrikaans",
                "oc": "occitan",
                "ka": "georgian",
                "be": "belarusian",
                "tg": "tajik",
                "sd": "sindhi",
                "gu": "gujarati",
                "am": "amharic",
                "yi": "yiddish",
                "lo": "lao",
                "uz": "uzbek",
                "fo": "faroese",
                "ht": "haitian creole",
                "ps": "pashto",
                "tk": "turkmen",
                "nn": "nynorsk",
                "mt": "maltese",
                "sa": "sanskrit",
                "lb": "luxembourgish",
                "my": "myanmar",
                "bo": "tibetan",
                "tl": "tagalog",
                "mg": "malagasy",
                "as": "assamese",
                "tt": "tatar",
                "haw": "hawaiian",
                "ln": "lingala",
                "ha": "hausa",
                "ba": "bashkir",
                "jw": "javanese",
                "su": "sundanese",
            }

Preciese_list = ['int8',
                'int8_float16',
                'int8_bfloat16',
                'int16',
                'float16',
                'float32',
                'bfloat16'
            ]

Model_names = [
                "tiny", 
                "tiny.en", 
                "base", 
                "base.en", 
                "small", 
                "small.en", 
                "medium", 
                "medium.en", 
                "large-v1", 
                "large-v2",
                "large-v3",
                "large-v3-turbo",
                "distil-large-v3",
                "distil-large-v2",
                "distil-medium.en",
                "distil-small.en",
            ]

Task_list = ["transcribe" , "translate"]

# ---------------------------------------------------------------------------------------------------------------------------
# 处理设备
#
# 每项同时维护「传给 CTranslate2 的实际设备名」和「下拉框显示文本」：
#   "AMD ROCm (HIP)" 显示给用户看，实际传给 CTranslate2 的是 "cuda"。
#
# 为什么 AMD 也要传 "cuda"：
#   CTranslate2 的 HIP 后端是把 CUDA 后端源码用 HIP 重新编译得到的同一套代码，对外 API 完全不变 ——
#   设备名、get_cuda_device_count()、CUDA_VISIBLE_DEVICES 全部沿用 CUDA 的命名。ROCm 版只是换了一个
#   编译产物（ctranslate2.dll 链接 amdhip64_7.dll / libhipblas.dll 而非 cudnn），不是代码层可切换的模式。
#   所以这里必须映射成 "cuda"，传 "rocm"/"hip" 会直接被 CTranslate2 拒绝。
#
# CPU / CUDA / auto 三项保持原有顺序和位置，确保旧的数字索引配置能正确迁移。
DEVICE_LIST = [
    {"text": "cpu",  "value": "cpu"},
    {"text": "cuda", "value": "cuda"},
    {"text": "auto", "value": "auto"},
    # 仅在探测到 ROCm/HIP 运行时（util.setupROCm() 成功）才加入下拉框
    #
    # 注意这里的 value 用 "rocm" 而不是 "cuda"：
    # 两者最终传给 CTranslate2 的都是 "cuda"（见下方说明），但必须用不同的值，
    # 否则 getParam() 存下 "cuda"、setDevice() 用 findData("cuda") 又只能匹配到第一项
    # （普通的 cuda），导致用户选了 AMD ROCm 后重启界面却显示回 "cuda"。
    # 真正的映射在 mainWindows.getParam_model() 里完成：rocm -> cuda
    {"text": "AMD ROCm (HIP)", "value": "rocm", "require_rocm": True},
]

# 旧版本配置文件中 device 字段保存的是下拉框索引，此处保留旧顺序用于迁移
LEGACY_DEVICE_LIST = ["cpu", "cuda", "auto"]


def deviceComboItems(include_rocm: bool = False):
    """
    构造设备下拉框的 (显示文本, 实际值) 列表

    :param include_rocm: 是否加入 AMD ROCm 选项
    """
    items = []
    for entry in DEVICE_LIST:
        if entry.get("require_rocm") and not include_rocm:
            continue
        items.append((entry["text"], entry["value"]))
    return items


def deviceComboTexts(include_rocm: bool = False):
    """设备下拉框的显示文本列表"""
    return [text for text, _ in deviceComboItems(include_rocm)]


# 兼容旧代码中对 Device_list 的引用（不含 ROCm）
Device_list = deviceComboTexts(include_rocm=False)

STR_BOOL = {"False" : False, "True" : True}

SUBTITLE_FORMAT = ["ASS", "JSON", "LRC", "SMI", "SRT", "TXT", "VTT"]

CAPTURE_PARA = [
    {"rate": 44100
    ,"channel": 2
    ,"dType": 16
    ,"quality": "CD Quality"
    },
    {"rate": 48000
    ,"channel": 2
    ,"dType": 16
    ,"quality": "DVD Quality"
    },
    {"rate": 44100
    ,"channel": 2
    ,"dType": 24
    ,"quality": "Studio Quality"
    },
    {"rate": 48000
    ,"channel": 2
    ,"dType": 24
    ,"quality": "Studio Quality"
    }
]

STEMS = [
            "All Stems", 
            "Vocals", 
            "Other",
            "Bass", 
            "Drums", 
            "Vocals and Others dichotomy"
        ]

ENCODING_DICT = {"UTF-8":"utf8", 
                    "UTF-8 BOM":"utf_8_sig", 
                    "GBK":"gbk", 
                    "GB2312":"gb18030", 
                    "ANSI":"ansi"
                }

THEME_COLORS = [
    "#009faa",
    "#81D8CF",
    "#ff009f",
    "#84BE84",
    "#aaff00",
    "#FF9500",
    "#00CD00",
    "#DB4437",
    "#23CD5E",
    "#E61D34",
    "#00FF00",
    "#FF00FF",
    "#1ABC9C",
    "#FF3300",
    "#FFFF00",
    "#FFC019",
    "#FF6600",
    "#00FFFF",
    "#FF7A1D",
    "#E71A1B",
    "#FF8800",
    "#3388FF",
    "#F4B400",
    "#0069B7",
    "#FFCC00",
    "#0078D4",
]

tableItem_dark_warning_BackGround_color = "#50ffff00" # QColor(255,255,0, a=80)
tableItem_light_warning_BackGround_color  = "#50ff0000" # QColor(255,0,0,a=127)
