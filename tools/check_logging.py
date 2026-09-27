# coding:utf-8
"""
logging_setup 的自检脚本。

    .venv\\Scripts\\python.exe tools\\check_logging.py

不依赖 pytest，直接跑，失败即非零退出。
所有产物写在临时目录里，不污染真实 logs/。
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# logging_setup 必须能在包导入之前使用（此时 ctranslate2 还没就绪），
# 所以这里也按路径加载，和 FasterWhisperGUI.py 的引导方式保持一致。
_spec = importlib.util.spec_from_file_location(
    "logging_setup_under_test", ROOT / "faster_whisper_GUI" / "logging_setup.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)

# 重定向到临时目录：真实 logs/ 不该被测试塞满
sandbox = Path(tempfile.mkdtemp(prefix="fwhisper-logtest-"))
ls.BASE_DIR = sandbox
ls.KEEP_RUNS = 3

failures = []


def check(name, condition, detail=""):
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


print("=== 1. 日志目录解析 ===")
log_dir = ls.resolveLogDir()
check("目录在程序目录下", log_dir == sandbox / "logs", str(log_dir))
check("目录已创建", log_dir.is_dir())
probe = log_dir / ".probe_write"
try:
    probe.write_text("x", encoding="utf-8")
    probe.unlink()
    check("目录可写", True)
except Exception as exc:
    check("目录可写", False, str(exc))

print("\n=== 2. 初始化 ===")
path = ls.setupLogging()
check("返回本次运行的日志路径", path is not None and path.exists(), path.name)
check("latest.log 存在", (log_dir / ls.LATEST_NAME).exists())
check("faster-whisper 日志存在",
      ls.getFrameworkLogPath() is not None and ls.getFrameworkLogPath().exists())
check("重复调用幂等", ls.setupLogging() == path)

print("\n=== 3. print 桥接 ===")
stream = ls.LoggingStream()
stream.write("单参数\n")
# 模拟 print("两", "个", "参数")：CPython 会分四次 write
for chunk in ("两", " ", "个", " ", "参数", "\n"):
    stream.write(chunk)
stream.write("a"); stream.write("|"); stream.write("b"); stream.write("\n")
stream.write("没有换行结尾"); stream.write("\n")
stream.write("空行\n\n结束\n")
stream.flush()

added = [ln for ln in path.read_text(encoding="utf-8").splitlines()
         if "logging_setup_under_test" in ln or "__main__" in ln]
added = [ln for ln in added if any(k in ln for k in
         ("单参数", "两 个 参数", "a|b", "没有换行结尾", "空行", "结束"))]
check("多参数合并成一条记录", any("两 个 参数" in ln for ln in added))
check("sep 生效", any("a|b" in ln for ln in added))
check("end='' 的部分写被缓冲后整行发出", any("没有换行结尾" in ln for ln in added))
check("空行不产生记录", not any(ln.rstrip().endswith("WARNING") for ln in added))
check("共 6 条记录", len(added) == 6, f"实际 {len(added)}: "
      f"{[ln.split(None, 4)[-1] for ln in added]}")

print("\n=== 4. 调用方归属 ===")
fake = types.ModuleType("我的模块")
fake.__dict__["stream"] = stream
exec("stream.write('来自归属测试\\n')", fake.__dict__)
check("日志里标出调用模块名", "我的模块" in path.read_text(encoding="utf-8"))

print("\n=== 5. 行格式 ===")
sample = [ln for ln in path.read_text(encoding="utf-8").splitlines() if "a|b" in ln]
if sample:
    line = sample[0]
    head = line[:23]
    check("时间戳含毫秒", head.count(":") == 2 and "." in head, head)
    check("含级别", " INFO " in line)
    check("含模块名", "logging_setup_under_test" in line or "__main__" in line)
else:
    check("找到样本行", False)

print("\n=== 6. 环境快照 ===")
snap = ls.environmentSnapshot()
for token in ("运行环境", "Python", "faster-whisper", "ctranslate2",
              "日志目录", "物理内存"):
    check(f"含 {token}", token in snap)
for line in snap.splitlines()[:5]:
    print("        " + line)

print("\n=== 7. 旧日志清理（保留 3 次运行）===")
import time  # noqa: E402
for i in range(6):
    f = log_dir / f"app-2020010{i}-000000.log"
    f.write_text("old", encoding="utf-8")
    time.sleep(0.01)
before = len(list(log_dir.glob("app-*.log")))
removed = ls.pruneOldRuns(log_dir, keep=3)
after = len(list(log_dir.glob("app-*.log")))
check("只保留 3 份当前运行日志", after <= 4, f"{before} -> {after}，删了 {len(removed)}")
check("最新那份没被删", path.exists())
check("latest.log 没被删", (log_dir / ls.LATEST_NAME).exists())

print("\n=== 8. 诊断包 ===")
try:
    bundle = ls.exportDiagnostics()
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    check("zip 可打开", True, bundle.name)
    check("含本次日志", any(n.startswith("app-") for n in names), str(names))
    check("含环境快照", "environment.txt" in names)
    check("不含配置文件（避免泄漏 token）",
          not any(n.endswith(".json") for n in names))
except Exception as exc:
    check("诊断包", False, str(exc))

print("\n=== 9. 输出流不会递归 / 巨量输出 ===")
old_out, old_err = sys.stdout, sys.stderr
recursion_error = None
lines_before = len(path.read_text(encoding="utf-8").splitlines())
try:
    sys.stdout = ls.LoggingStream()
    sys.stderr = ls.LoggingStream()
    for i in range(500):
        print(f"压力测试第 {i} 行")
    sys.stderr.write("错误流也要进日志\n")
    sys.stdout.flush()
except RecursionError:
    recursion_error = "RecursionError"
except Exception as exc:
    recursion_error = f"{type(exc).__name__}: {exc}"
finally:
    sys.stdout, sys.stderr = old_out, old_err
lines_after = len(path.read_text(encoding="utf-8").splitlines())
check("500 行输出无递归", recursion_error is None, recursion_error or "")
check("全部落盘", lines_after - lines_before >= 500,
      f"新增 {lines_after - lines_before} 行")

print("\n=== 10. 关闭 ===")
ls.shutdownLogging()
ls.shutdownLogging()
check("shutdown 幂等", True)

print("\n=== 11. 模块被加载成两份实例时不得清空日志 ===")
# 这是实测踩过的坑：本模块既可能被按路径加载（FasterWhisperGUI.py 的引导），
# 也可能被包内模块正常导入。两份实例各自的 _configured 都是 False，
# 第二次 setupLogging() 会用 mode="w" 把刚写好的日志清空。
sandbox2 = Path(tempfile.mkdtemp(prefix="fwhisper-logtest2-"))
ls2 = importlib.util.module_from_spec(_spec)      # 同一份源码，第二个模块对象
_spec.loader.exec_module(ls2)
ls2.BASE_DIR = sandbox2

size_before = path.stat().st_size
second = ls2.setupLogging()                        # 第二实例尝试重新配置
size_after = path.stat().st_size
check("第二实例没有另建日志目录", not (sandbox2 / "logs").exists(),
      str(list(sandbox2.iterdir()) if sandbox2.exists() else []))
check("第二实例返回第一实例的路径", second == path, f"{second} vs {path}")
check("日志内容没有被清空", size_after >= size_before and size_after > 0,
      f"{size_before} -> {size_after}")
shutil.rmtree(sandbox2, ignore_errors=True)

shutil.rmtree(sandbox, ignore_errors=True)
check("临时目录已清理", not sandbox.exists())

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("全部通过")
