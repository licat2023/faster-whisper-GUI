# coding:utf-8
"""
把 print() 调用改写成带级别的 logging 调用。

    .venv\\Scripts\\python.exe tools\\convert_prints.py            # 预览（不改文件）
    .venv\\Scripts\\python.exe tools\\convert_prints.py --apply    # 实际改写

为什么用 AST 而不是正则
-----------------------
print 的参数可能跨很多行、含嵌套括号和 f-string，正则分不清 `print(a, b)` 里的逗号
是参数分隔还是表达式内部。AST 能给出每个参数对象的精确源码区间（get_source_segment），
所以能原样保留参数文本。

为什么统一写成 log.info("%s %s", a, b) 而不是 log.info(a)
---------------------------------------------------------
logging 会把第一个参数当格式串。若 a 本身含 `%`（转义序列、百分号、URL 编码），
`log.info(a)` 会抛 "not enough arguments for format string"。加一层常量格式串
`"%s"` 就把用户数据全部降级为参数，任何内容都安全。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "faster_whisper_GUI"
TARGETS = sorted(PACKAGE.glob("*.py")) + [
    ROOT / "FasterWhisperGUI.py",
    ROOT / "whisperx" / "transcribe.py",
    ROOT / "whisperx" / "asr.py",
    ROOT / "whisperx" / "utils.py",
]

ERROR_WORDS = ("错误", "失败", "异常", "出错", "error", "Error", "ERROR",
               "fail", "Fail", "FAIL", "无法", "不能", "invalid", "Invalid")
WARN_WORDS = ("警告", "注意", "warn", "Warn", "WARNING", "已忽略", "跳过",
              "ignore", "Ignore", "未找到", "不支持")

LOGGER_LINE = "log = logging.getLogger(__name__)"
_LOGGER_RE = re.compile(r"^log\s*=\s*logging\.getLogger\(", re.M)


class Collector(ast.NodeVisitor):
    """收集所有 print() 调用，并记录它是否位于 except 块内。"""

    def __init__(self, source: str):
        self.source = source
        self.found: List[Tuple[ast.Call, bool]] = []
        self._in_except = 0

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self._in_except += 1
        self.generic_visit(node)
        self._in_except -= 1

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            self.found.append((node, self._in_except > 0))
        self.generic_visit(node)


class SourceMap:
    """
    把 AST 的 (行号, 列) 换算成源码字符串下标。

    坑：Python 的 col_offset / end_col_offset 是 **UTF-8 字节偏移**，不是字符偏移
    （CPython 文档：col_offset is the UTF-8 byte offset of the first token）。
    源码里有中文时两者不等长（1 个汉字 = 3 字节）。直接用会把替换区间算歪 ——
    实测就是把一个 print 的范围多算了 7 个字符，切进了下一行的 `new_line = ...`，
    结果生成出 `log.info(...))ew_line = "..."` 这种语法错误。
    """

    def __init__(self, source: str):
        self.lines = source.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, byte_col: int) -> int:
        line = self.lines[lineno - 1]
        # 取前 byte_col 个字节再解码，末尾被截断的半个字符会被丢弃
        char_col = len(line.encode("utf-8")[:byte_col].decode("utf-8", "ignore"))
        return self.starts[lineno - 1] + char_col


def pick_level(node: ast.Call, in_except: bool) -> str:
    """按上下文和文本内容选级别。宁可保守：不确定就 INFO。"""
    text = ""
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            text += arg.value + " "
    if in_except:
        # except 块里的 print 基本都是在报错
        return "error"
    if any(word in text for word in ERROR_WORDS):
        return "error"
    if any(word in text for word in WARN_WORDS):
        return "warning"
    return "info"


def convert_source(source: str, path: Path) -> Tuple[str, int, int, List[str]]:
    """返回 (新源码, 改写数, 跳过数, 跳过原因)。"""
    tree = ast.parse(source)
    collector = Collector(source)
    collector.visit(tree)

    offsets = SourceMap(source)
    edits: List[Tuple[int, int, str]] = []
    skipped: List[str] = []

    for node, in_except in collector.found:
        if node.keywords:
            # sep/end/file/flush 没有对应的 logging 语义。
            # 不强行改写 —— 这些输出仍会被 LoggingStream 捕获，只是没有级别。
            skipped.append(f"{path.name}:{node.lineno} 含关键字参数")
            continue

        level = pick_level(node, in_except)
        start = offsets.offset(node.lineno, node.col_offset)
        end = offsets.offset(node.end_lineno, node.end_col_offset)

        if not node.args:
            replacement = f'log.{level}("")'
        else:
            segments = []
            for arg in node.args:
                segment = ast.get_source_segment(source, arg)
                if segment is None:
                    break
                segments.append(segment)
            else:
                fmt = " ".join(["%s"] * len(node.args))
                replacement = (f'log.{level}("{fmt}", '
                               + ", ".join(segments) + ")")
                edits.append((start, end, replacement))
                continue
            skipped.append(f"{path.name}:{node.lineno} 无法取到参数源码")
            continue

        edits.append((start, end, replacement))

    # 从后往前替换，避免前面的偏移量失效
    new_source = source
    for start, end, replacement in sorted(edits, reverse=True):
        new_source = new_source[:start] + replacement + new_source[end:]

    return new_source, len(edits), len(skipped), skipped


def ensure_logger(source: str) -> str:
    """确保文件里有 `log = logging.getLogger(__name__)`（以及 import logging）。"""
    if _LOGGER_RE.search(source):
        return source

    if not re.search(r"^import logging\b", source, re.M):
        # 插到第一个 import 之前，保证顺序合理
        match = re.search(r"^(?:#.*\n)*?(import |from )", source, re.M)
        if match:
            source = (source[:match.start()]
                      + "import logging\n"
                      + source[match.start():])
        else:
            lines = source.splitlines(keepends=True)
            insert_at = 1 if lines and lines[0].startswith("#") else 0
            source = ("".join(lines[:insert_at]) + "import logging\n"
                      + "".join(lines[insert_at:]))

    # 找最后一个顶层 import 语句，把 logger 定义插在它后面
    tree = ast.parse(source)
    last_import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = node.end_lineno
    if not last_import_end:
        last_import_end = 0

    lines = source.splitlines(keepends=True)
    block = f"\n{LOGGER_LINE}\n"
    return ("".join(lines[:last_import_end]) + block
            + "".join(lines[last_import_end:]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真正写回文件")
    args = parser.parse_args()

    total_changed = 0
    total_skipped = 0
    touched: List[str] = []

    for path in TARGETS:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if "print(" not in source:
            continue

        try:
            new_source, changed, skipped_n, skipped_reasons = convert_source(
                source, path)
        except SyntaxError as exc:
            print(f"  [SKIP] {path.name}: 语法错误 {exc}")
            continue

        if changed == 0:
            if skipped_n:
                print(f"  [----] {path.name}: {skipped_n} 处全部跳过")
            continue

        new_source = ensure_logger(new_source)
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            print(f"  [FAIL] {path.name}: 改写后语法错误，已放弃 -> {exc}")
            return 1

        total_changed += changed
        total_skipped += skipped_n
        touched.append(path.name)
        print(f"  [OK  ] {path.name:<40} 改写 {changed:3d} 处"
              + (f"，跳过 {skipped_n}" if skipped_n else ""))

        if args.apply:
            path.write_text(new_source, encoding="utf-8")

    print(f"\n合计改写 {total_changed} 处，跳过 {total_skipped} 处，"
          f"涉及 {len(touched)} 个文件")
    if not args.apply:
        print("（预览模式，未写入。加 --apply 生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
