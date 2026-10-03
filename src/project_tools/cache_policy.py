"""为项目虚拟环境配置集中字节码缓存；尊重显式的缓存前缀。"""

from pathlib import Path
import sys
import sysconfig


def main():
    root = Path(__file__).resolve().parents[2]
    environment = (root / ".venv").resolve()
    purelib = sysconfig.get_path("purelib")
    if not purelib:
        raise RuntimeError("解释器未提供 site-packages 路径")
    site_packages = Path(purelib).resolve()
    if sys.prefix == sys.base_prefix or not site_packages.is_relative_to(environment):
        raise RuntimeError("请使用本项目 .venv 的 Python 配置缓存")
    prefix = str(root / ".cache" / "pycache")
    hook = site_packages / "_project_bytecode_cache.pth"
    temporary = hook.with_suffix(".pth.tmp")
    temporary.write_text(
        f"import sys; sys.pycache_prefix = sys.pycache_prefix or {prefix!r}\n",
        encoding="utf-8",
    )
    temporary.replace(hook)
    print(f"字节码缓存：{prefix}")


if __name__ == "__main__":
    main()
