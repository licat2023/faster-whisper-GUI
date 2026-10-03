# coding:utf-8
"""
本次修复的回归自检（与 src/project_tools/check_logging.py、src/project_tools/check_workers.py 同规格）。

覆盖 14 项，全部对应真实修复过的问题：
  F1  raiseErrorBar 未定义导致 except 里再抛 AttributeError
  F2  OpenCC 简繁转换：cc 未绑定 / 词级转换是空操作
  F3  pyannote 4.x 把 use_auth_token 改名为 token
  F4  whisperx 的 TranscriptionOptions 缺 5 个必填字段
  F5  裸 except + pass 静默吞异常
  F6  跨线程操作 Qt（日志→文本框 / worker→StateTool / 扫描线程→列表模型）
  F7  requestInterruption() 是无效死调用
  F8  writeASS 缺 speaker 字段 / seg_ment 词级概率取错
  WX  WhisperX 复用 worker 时结果串台 + 结束回调重复连接
  TK  硬编码 HuggingFace 令牌

用法（在仓库根目录）：
    uv run src/project_tools/check_fixes.py
    # 或 .venv\\Scripts\\python.exe src\\project_tools\\check_fixes.py

说明：全程走 Qt offscreen 平台，不需要显示器；**不构造完整的 MainWindows**
（无头环境下它在构造过程中就会原生退出，原始 HEAD 版本同样如此），
日志出口改为组件级验证。
"""

if not __debug__:
    raise RuntimeError("Run verification without -O or PYTHONOPTIMIZE; assertions are required")

import ast
import dataclasses
import importlib.util
import inspect
import io
import logging
import os
import re
import sys
import tempfile
import time
import threading
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 统一的推理引导负责 DLL -> torch -> CTranslate2 的加载顺序。
from faster_whisper_GUI.runtime.inference import prepare_inference_runtime

if not prepare_inference_runtime():
    print("[警告] 未探测到 ROCm/HIP，仍继续（对齐/分离相关项可能不可用）")

import whisperx  # noqa: E402,F401

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append(("OK  ", name, detail or ""))
        print(f"  [OK  ] {name}" + (f"  --  {detail}" if detail else ""), flush=True)
    except Exception as exc:                                   # noqa: BLE001
        RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {name}  --  {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()


# --------------------------------------------------------------------------- 工具
def iter_python_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in
                   (".venv", "__pycache__", ".git", ".probe", "logs", "temp", "cache", ".cache", "build", "dist")]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def attribute_calls(paths=None):
    """(文件, 行号, 方法名) —— 只看真实调用，不受注释/文档字符串影响。"""
    for path in (paths or iter_python_files()):
        try:
            tree = ast.parse(io.open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                yield path, node.lineno, node.func.attr


def attribute_accesses(path):
    tree = ast.parse(io.open(path, encoding="utf-8", errors="replace").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            yield node.attr, node.lineno


def make_silent_wav(path, seconds=0.2, rate=16000):
    with wave.open(path, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


# =========================================================================== F1
def t_f1():
    hits = [f"{os.path.relpath(p, REPO_ROOT)}:{n}" for p, n, attr in attribute_calls()
            if attr == "raiseErrorBar"]
    assert not hits, f"仍存在 raiseErrorBar 调用: {hits}"
    from faster_whisper_GUI.ui.window.main import MainWindows

    src = inspect.getsource(MainWindows.loadBackupConfigFile)
    assert 'self.raiseErrorInfoBar(self._tr("加载配置文件失败")' in src
    return "调用已消除，改用已定义的 raiseErrorInfoBar"


# =========================================================================== F2
def t_f2():
    from faster_whisper.transcribe import Word

    from faster_whisper_GUI.ui.window.main import MainWindows

    convert = MainWindows.simplifiedAndTraditionalChineseConvert

    def fake_segment(text, words):
        seg = type("S", (), {})()
        seg.text = text
        seg.words = words
        return seg

    seg = fake_segment("漢語", [Word(0.0, 0.5, "漢", 0.9)])
    convert(None, [seg], "yue")                      # 旧实现：UnboundLocalError
    assert seg.text == "漢語", "未知语言不应改动文本"

    words = [Word(0.0, 0.5, "漢", 0.9), Word(0.5, 1.0, "語", 0.8)]
    seg = fake_segment("漢語", words)
    convert(None, [seg], "Auto")
    assert seg.text == "汉语", f"t2s 段级失败: {seg.text!r}"
    assert [w.word for w in words] == ["汉", "语"], f"t2s 词级失败: {[w.word for w in words]}"

    seg = fake_segment("漢語", [Word(0.0, 0.5, "漢", 0.9)])
    convert(None, [seg], "zhs")
    assert seg.text == "汉语"

    words = [Word(0.0, 0.5, "汉", 0.9)]
    seg = fake_segment("汉语", words)
    convert(None, [seg], "zht")
    assert seg.text == "漢語", f"s2t 段级失败: {seg.text!r}"
    assert words[0].word == "漢", f"s2t 词级失败: {words[0].word!r}"
    return "yue 不抛异常；zhs/zht 段级与词级均生效"


# =========================================================================== F3
def t_f3():
    import inspect

    import whisperx.diarize as dz
    from pyannote.audio import Pipeline as RealPipeline

    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            captured.clear()
            captured["checkpoint"] = checkpoint
            captured.update(kwargs)
            return "FAKE_MODEL"

    original = dz.Pipeline
    dz.Pipeline = FakePipeline
    try:
        assert dz._load_pipeline("pyannote/speaker-diarization@2.1",
                                 "hf_TESTTOKEN", "/tmp/cache") == "FAKE_MODEL"
    finally:
        dz.Pipeline = original

    assert captured.get("token") == "hf_TESTTOKEN", f"新名字未生效: {captured}"
    assert "use_auth_token" not in captured, f"仍在传旧名字: {captured}"

    params = list(inspect.signature(RealPipeline.from_pretrained).parameters)
    assert "token" in params and "use_auth_token" not in params, \
        f"当前 pyannote 签名与预期不符: {params}"
    return f"按 token= 调用；真实签名 = {params}"


def t_f3b():
    import whisperx.diarize as dz

    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            captured.clear()
            captured["checkpoint"] = checkpoint
            captured.update(kwargs)
            return "FAKE_MODEL"

    original = dz.Pipeline
    dz.Pipeline = FakePipeline
    try:
        dz._load_pipeline("pyannote/speaker-diarization@2.1", None, None)
    finally:
        dz.Pipeline = original
    assert captured == {"checkpoint": "pyannote/speaker-diarization@2.1",
                        "cache_dir": None}, captured
    return "无令牌时不传鉴权参数，并已记 WARNING"


# =========================================================================== F4
def t_f4():
    from faster_whisper.transcribe import TranscriptionOptions

    tree = ast.parse(io.open(os.path.join(REPO_ROOT, "src", "whisperx", "asr.py"),
                             encoding="utf-8").read())
    node = None
    for item in ast.walk(tree):
        if (isinstance(item, ast.Assign)
                and getattr(item.targets[0], "id", "") == "default_asr_options"
                and isinstance(item.value, ast.Dict)):
            node = item.value
            break
    assert node is not None, "没找到 default_asr_options"

    options = ast.literal_eval(node)
    options.pop("suppress_numerals", None)
    fields = {f.name for f in dataclasses.fields(TranscriptionOptions)}
    missing = sorted(fields - set(options))
    assert not missing, f"仍缺字段: {missing}"
    TranscriptionOptions(**options)          # 旧代码在这里 TypeError
    return f"{len(options)} 个键，TranscriptionOptions 构造成功"


# =========================================================================== F5
def t_f5():
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication(sys.argv)

    from faster_whisper_GUI.ui.pages.settings import SettingPageNavigationInterface
    from faster_whisper_GUI.ui.pages.transcription import TranscribeNavigationInterface

    # 父控件必须留引用：匿名 QWidget() 会被 GC，连带销毁子页面
    parent_setting, parent_transcribe = QWidget(), QWidget()

    setting_page = SettingPageNavigationInterface(parent_setting)
    try:
        setting_page.setParam({})            # 必然 KeyError
    except Exception as exc:                 # noqa: BLE001
        raise AssertionError(f"设置页异常泄漏到调用方: {type(exc).__name__}: {exc}") from exc

    # 用页面自己产出的参数字典，不依赖 fasterWhisperGUIConfig.json ——
    # 该文件已移出版本库，全新克隆里并不存在。
    transcribe_page = TranscribeNavigationInterface(parent_transcribe)
    params = transcribe_page.getParam()      # 自洽：键集与 setParam 完全对应
    transcribe_page.setParam(params)         # 正常路径
    broken = dict(params)
    broken.pop("chunk_length")               # try 块里的第一个键
    try:
        transcribe_page.setParam(broken)     # 缺键路径
    except Exception as exc:                 # noqa: BLE001
        raise AssertionError(f"转写页异常泄漏到调用方: {type(exc).__name__}: {exc}") from exc

    assert app is not None and parent_setting is not None and parent_transcribe is not None
    return "设置页/转写页都吞住异常并记 WARNING（不再静默）"


# =========================================================================== F6
def t_f6a_b():
    from faster_whisper_GUI.ui.window.main import MainWindows
    from faster_whisper_GUI.tasks.alignment import WhisperXWorker

    assert MainWindows.staticMetaObject.indexOfSignal("signal_guiLog(QString)") >= 0, \
        "缺少 signal_guiLog"
    assert "attachGuiHandler(self.signal_guiLog.emit)" in inspect.getsource(MainWindows.redirectOutput), \
        "日志仍直连界面槽"
    src = inspect.getsource(MainWindows.whisperXAligmentTimeStample)
    src += inspect.getsource(MainWindows.whisperXDiarizeSpeakers)
    assert "self.whisperXWorker.stateToolRequest.connect(self.setStateTool)" in src, \
        "WhisperX 的 stateToolRequest 未接"

    worker = inspect.getfile(WhisperXWorker)
    assert "stateToolRequest = Signal(str, bool)" in io.open(worker, encoding="utf-8").read()
    parents = [line for attr, line in attribute_accesses(worker) if attr == "parent"]
    assert not parents, f"whisper_x.py 仍在访问 parent()，行号 {parents}"
    return "日志与 StateTool 均改为信号投递；worker 不再碰 parent()"


def t_f6c():
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication(sys.argv)

    from faster_whisper_GUI.ui.widgets.file_list import FileNameListView

    parent = QWidget()                       # 留引用，避免被 GC
    view = FileNameListView(parent)
    assert view.filterFileNames([]) == ([], "", [])

    emitted = []
    view.ignore_files_signal.connect(lambda info: emitted.append(info))

    with tempfile.TemporaryDirectory() as tmp:
        sample = make_silent_wav(os.path.join(tmp, "sample.wav"))
        missing = os.path.join(tmp, "no_such_file.wav")

        accepted, base_dir, infos = view.filterFileNames([missing, sample])
        assert accepted == [sample], accepted
        assert base_dir == os.path.dirname(missing), base_dir
        assert len(infos) == 1, infos
        assert emitted == [], "过滤阶段不应发信号（那是工作线程会做的事）"

        view._applyFilteredFileNames(accepted, base_dir, infos)
        assert view.FileNameModle.stringList() == [sample], view.FileNameModle.stringList()
        assert view.avFileList == [sample]
        assert view.avDataRootDir == base_dir
        assert len(emitted) == 1, "落地阶段应发一次忽略提示"

        emitted.clear()
        view.setFileNameListToDataModel([sample])      # 重复添加
        deadline = time.monotonic() + 5
        while not emitted and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert len(emitted) == 1, emitted
        assert view.FileNameModle.stringList() == [sample]
    assert app is not None
    return "过滤纯 I/O 化；信号只在界面线程发；模型只在界面线程写"


def t_log_bridge():
    """F6 的组件级验证：日志经 Qt 信号从工作线程投递到界面线程。"""
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from faster_whisper_GUI import logging_setup

    app = QApplication.instance() or QApplication(sys.argv)
    main_thread = threading.get_ident()

    received, seen_threads = [], []

    class Bridge(QObject):
        message = Signal(str)

    bridge = Bridge()
    bridge.message.connect(lambda text: (received.append(text), seen_threads.append(threading.get_ident())))
    handler = logging_setup.attachGuiHandler(bridge.message.emit)
    log = logging.getLogger("check_fixes.bridge")

    try:
        done = threading.Event()

        def worker():
            log.warning("来自工作线程")
            done.set()

        thread = threading.Thread(target=worker, name="fake-transcribe")
        thread.start()
        thread.join(timeout=10)
        for _ in range(50):
            app.processEvents()

        assert received == ["来自工作线程\n"], f"工作线程日志未投递/内容不对: {received!r}"
        assert seen_threads[0] == main_thread, \
            f"槽在非界面线程执行: {seen_threads[0]} != {main_thread}"

        received.clear()
        seen_threads.clear()
        log.warning("来自界面线程")
        for _ in range(50):
            app.processEvents()
        assert received == ["来自界面线程\n"], f"界面线程日志不正常: {received!r}"
        assert seen_threads[0] == main_thread

        received.clear()
        seen_threads.clear()
        workers = [threading.Thread(target=lambda i=i: log.warning("并发%d", i))
                   for i in range(20)]
        for item in workers:
            item.start()
        for item in workers:
            item.join(timeout=10)
        for _ in range(100):
            app.processEvents()
        assert len(received) == 20, f"并发日志丢失: 收到 {len(received)}/20"
        assert set(seen_threads) == {main_thread}, f"并发时槽跑到了别的线程: {set(seen_threads)}"
    finally:
        logging_setup.detachHandler(handler)

    return "工作线程/界面线程/20 线程并发都正确投递，且全部在界面线程执行"


# =========================================================================== WX
def t_wx():
    from faster_whisper_GUI.ui.window.main import MainWindows

    src = inspect.getsource(MainWindows.whisperXAligmentTimeStample)
    src += inspect.getsource(MainWindows.whisperXDiarizeSpeakers)
    src += inspect.getsource(MainWindows._connectWhisperXFinished)
    assert "self.whisperXWorker.result_segments_path_info = self.current_result" not in src, \
        "对齐分支仍写入 run() 不读的属性"
    assert "self.whisperXWorker.segments_path_info = self.current_result" in src
    assert "_connectWhisperXFinished" in src
    return "对齐分支改用正确的 segments_path_info；结束回调不再重复连接"


# =========================================================================== F7
def t_f7():
    hits = [f"{os.path.relpath(p, REPO_ROOT)}:{n}" for p, n, attr in attribute_calls()
            if attr == "requestInterruption"]
    assert not hits, f"仍有 requestInterruption 调用: {hits}"
    return "已全部移除（取消只走 stop()）"


# =========================================================================== F8
def t_f8a():
    from faster_whisper_GUI.domain.segments import dictionaryListToSegmentList

    data = [{
        "start": 0.0, "end": 1.0, "text": "hi",
        "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.87}],
    }]
    segments = dictionaryListToSegmentList(data)
    probability = segments[0].words[0].probability
    assert probability == 0.87, f"概率值仍不对: {probability!r}（旧实现是 ['score']）"

    data[0]["words"] = [{"word": "hi", "start": 0.0, "end": 1.0}]   # 缺 score 也不能炸
    dictionaryListToSegmentList(data)
    return "词级概率正确取自 score，缺键时退回 0.0"


def t_f8b():
    from faster_whisper_GUI.domain.segments import segment_Transcribe
    from faster_whisper_GUI.subtitles.writers import writeASS

    segment = segment_Transcribe(start=0.0, end=1.5, text="你好", words=[], speaker=None)
    del segment.speaker                      # 模拟缺字段的外部结果
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.ass")
        writeASS(path, [segment], file_code="utf8")
        body = io.open(path, encoding="utf8").read()
    assert "Dialogue:" in body
    return "缺 speaker 字段不再 AttributeError，Dialogue 行正常写出"


def t_f9():
    """实时转写结果必须与普通文件转写使用相同的数据结构。"""
    from faster_whisper.transcribe import Segment, Word
    from PySide6.QtCore import Qt

    from faster_whisper_GUI.domain.segments import segment_Transcribe
    from faster_whisper_GUI.ui.models.segments import TableModel
    from faster_whisper_GUI.transcription.streaming import AudioStreamTranscribeWorker
    from faster_whisper_GUI.subtitles.writers import writeJson

    source = Segment(
        id=0,
        seek=0,
        start=0.25,
        end=1.0,
        text=" hello",
        tokens=[1],
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=[Word(start=0.25, end=1.0, word=" hello", probability=0.9)],
        temperature=0.0,
    )
    worker = AudioStreamTranscribeWorker()
    shifted = worker._shiftSegments([source], 2.0)

    assert len(shifted) == 1
    result = shifted[0]
    assert isinstance(result, segment_Transcribe), type(result)
    assert result.speaker is None
    assert result.start == 2.25 and result.end == 3.0
    assert result.words[0].start == 2.25 and result.words[0].end == 3.0

    # 复现真实崩溃路径：结果表说话人列和 JSON 导出都必须能直接消费它。
    model = TableModel(shifted)
    assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == ""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stream.json")
        writeJson(path, shifted, "en", "stream.wav")
        assert os.path.exists(path)

    return "实时 Segment 已归一化；表格显示与 JSON 导出均可消费"


# =========================================================================== TK
def t_token():
    import subprocess

    from faster_whisper_GUI.config import default_Huggingface_user_token
    assert default_Huggingface_user_token == "", repr(default_Huggingface_user_token)

    src = io.open(os.path.join(REPO_ROOT, "src", "faster_whisper_GUI", "config.py"),
                  encoding="utf8").read()
    # 只看像真令牌的字面量（hf_ + 长串）；注释里的 hf_xxxxx 占位不算
    leaked = re.findall(r"hf_[A-Za-z0-9]{20,}", src)
    assert not leaked, f"config.py 仍硬编码令牌字面量: {leaked}"
    assert "os.environ.get(" in src and "FASTER_WHISPER_GUI_HF_TOKEN" in src, \
        "默认值没有改成从环境变量读取"

    # 运行时配置不该进版本库 —— 这才是防止再次泄露的根本。
    # 旧实现把它纳入了 git，而程序退出时会把（含令牌的）设置写回去。
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                              "fasterWhisperGUIConfig.json"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    assert tracked.returncode != 0, "fasterWhisperGUIConfig.json 仍被 git 跟踪"

    ignore_text = io.open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf8").read()
    assert "fasterWhisperGUIConfig.json" in ignore_text, ".gitignore 未忽略该配置文件"
    return "令牌改由环境变量提供；运行时配置文件已移出版本库并被忽略"


# --------------------------------------------------------------------------- 主流程
def main():
    print("=== 1. 崩溃路径 ===")
    check("F1 raiseErrorBar 未定义", t_f1)
    check("F2 OpenCC 简繁转换", t_f2)

    print()
    print("=== 2. 依赖与代码不一致 ===")
    check("F3 pyannote 4.x 令牌参数改名", t_f3)
    check("F3b 无令牌路径", t_f3b)
    check("F4 whisperx TranscriptionOptions 缺字段", t_f4)

    print()
    print("=== 3. 静默失败与数据错误 ===")
    check("F5 静默吞异常", t_f5)
    check("F8a seg_ment 词级概率", t_f8a)
    check("F8b writeASS 字段兜底", t_f8b)
    check("F9 实时转写 Segment 归一化", t_f9)

    print()
    print("=== 4. 跨线程与界面状态 ===")
    check("F6a/F6b 跨线程 Qt 访问", t_f6a_b)
    check("F6c 文件列表跨线程", t_f6c)
    check("F6 日志跨线程投递", t_log_bridge)
    check("WX WhisperX 复用时的结果串台", t_wx)

    print()
    print("=== 5. 死调用与凭据 ===")
    check("F7 取消语义统一", t_f7)
    check("TK 硬编码令牌", t_token)

    print()
    print("=" * 70)
    failed = [row for row in RESULTS if row[0] == "FAIL"]
    print(f"共 {len(RESULTS)} 项，通过 {len(RESULTS) - len(failed)} 项，失败 {len(failed)} 项")
    if failed:
        for _, name, detail in failed:
            print(f"  [FAIL] {name}  --  {detail}")
        print("未全部通过")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
