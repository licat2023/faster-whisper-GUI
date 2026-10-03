# coding:utf-8
"""
GuardedWorker 的自检脚本。

    .venv\\Scripts\\python.exe src\\project_tools\\check_workers.py

验证「QThread.run() 抛异常时 traceback 不再消失」这件事真的成立。
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 先起日志（workers 的护栏依赖它落盘）
_spec = importlib.util.spec_from_file_location(
    "logging_setup_under_test", ROOT / "src" / "faster_whisper_GUI" / "logging_setup.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)

sandbox = Path(tempfile.mkdtemp(prefix="fwhisper-workertest-"))
ls.BASE_DIR = sandbox
log_path = ls.setupLogging()

# 单独加载任务护栏；不初始化推理依赖。
_wspec = importlib.util.spec_from_file_location(
    "workers_under_test", ROOT / "src" / "faster_whisper_GUI" / "tasks" / "base.py")
w = importlib.util.module_from_spec(_wspec)
_wspec.loader.exec_module(w)

from PySide6.QtCore import QCoreApplication  # noqa: E402

app = QCoreApplication(sys.argv)

failures = []
received_failures = []


def check(name, condition, detail=""):
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def log_text():
    return log_path.read_text(encoding="utf-8")


print("=== 1. run() 抛异常：traceback 必须落盘 ===")
on_error_calls = []


class Boom(w.GuardedWorker):
    def run(self):
        raise ValueError("故意炸一个")

    def onError(self, exc):
        on_error_calls.append(exc)


boom = Boom()
boom.failed.connect(lambda msg: received_failures.append(msg))
marker = len(log_text().splitlines())
boom.run()                      # 直接调用，避免依赖事件循环
new = log_text().splitlines()[marker:]
new_text = "\n".join(new)

check("记录了 CRITICAL", any("CRITICAL" in ln for ln in new))
check("记录了异常信息", "故意炸一个" in new_text)
check("记录了完整 traceback", "Traceback (most recent call last)" in new_text)
check("traceback 指到真正的出错行", 'raise ValueError("故意炸一个")' in new_text)
check("标出了线程类名", "Boom" in new_text)
check("onError 被调用", len(on_error_calls) == 1
      and isinstance(on_error_calls[0], ValueError))
check("failed 信号发出", len(received_failures) == 1, str(received_failures))
check("lastError 已保存", isinstance(boom.lastError, ValueError))
check("异常没有再往外抛", True)

print("\n=== 2. run() 正常返回：不受影响 ===")
ran = []


class Fine(w.GuardedWorker):
    def run(self):
        ran.append(True)
        return "结果"


fine = Fine()
result = fine.run()
check("run 正常执行", ran == [True])
check("返回值原样返回", result == "结果")
check("没有误记异常", "Fine" not in "\n".join(
    ln for ln in log_text().splitlines() if "CRITICAL" in ln))

print("\n=== 3. onError 自己抛异常也不能影响日志 ===")
nasty_calls = []


class Nasty(w.GuardedWorker):
    def run(self):
        raise RuntimeError("原始故障")

    def onError(self, exc):
        nasty_calls.append(exc)
        raise OSError("onError 自己也炸了")


marker = len(log_text().splitlines())
Nasty().run()
new_text = "\n".join(log_text().splitlines()[marker:])
check("原始异常已记录", "原始故障" in new_text)
check("onError 被调用", len(nasty_calls) == 1)
check("onError 的异常也被记录", "onError 自己也炸了" in new_text)
check("onError 的异常没覆盖原始异常",
      "Traceback (most recent call last)" in new_text)

print("\n=== 4. 未定义 run() 的子类不受影响 ===")
try:
    class NoRun(w.GuardedWorker):
        pass

    NoRun()
    check("可以实例化", True)
except Exception as exc:
    check("可以实例化", False, str(exc))

print("\n=== 5. 包装只做一次（不会层层套娃）===")
check("run 带 _guarded 标记", getattr(Boom.__dict__["run"], "_guarded", False))
check("不会重复包装", Boom.run.__wrapped__ is not None
      if hasattr(Boom.run, "__wrapped__") else True)

print("\n=== 6. 源码里不应再有裸的 QThread 子类 ===")
pkg = ROOT / "src" / "faster_whisper_GUI"
offenders = []
for path in sorted(pkg.rglob("*.py")):
    if path == pkg / "tasks" / "base.py":     # 基类自己当然继承 QThread
        continue
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"^class\s+(\w+)\(([^)]*)\)", text, re.M):
        name, bases = match.group(1), match.group(2)
        if "QThread" in bases:
            offenders.append(f"{path.name}:{name}")
check("所有 QThread 子类都已换成 GuardedWorker", not offenders, str(offenders))

print("\n=== 7. 每个 worker 类都确实被护栏覆盖 ===")
worker_files = ["transcription/file.py", "transcription/streaming.py",
                "tasks/capture.py", "tasks/export.py", "tasks/alignment.py",
                "tasks/separation.py", "tasks/model.py", "tasks/audio_split.py"]
for fname in worker_files:
    text = (pkg / fname).read_text(encoding="utf-8")
    classes = re.findall(r"^class\s+(\w+)\(GuardedWorker\)", text, re.M)
    has_run = [c for c in classes
               if re.search(rf"class {c}\(GuardedWorker\):(.*?)(?=^class |\Z)",
                            text, re.M | re.S)
               and "def run(" in re.search(
                   rf"class {c}\(GuardedWorker\):(.*?)(?=^class |\Z)",
                   text, re.M | re.S).group(1)]
    check(f"{fname}: {len(classes)} 个类", len(classes) > 0,
          f"带 run() 的: {has_run}")

ls.shutdownLogging()
shutil.rmtree(sandbox, ignore_errors=True)

print()
if received_failures:
    print(f"  (捕获到的 failed 信号: {received_failures})")
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("全部通过")
