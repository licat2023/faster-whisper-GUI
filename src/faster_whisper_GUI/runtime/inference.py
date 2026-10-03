"""推理依赖的唯一引导顺序：DLL 探测 -> torch -> CTranslate2 调用方。"""

from functools import lru_cache

from .rocm import setupROCm


@lru_cache(maxsize=1)
def prepare_inference_runtime() -> bool:
    """幂等注册 DLL，再预加载 torch；调用方随后可安全导入推理库。"""
    ready = setupROCm()
    import torch  # noqa: F401: torch 必须早于 ROCm 版 CTranslate2 加载。
    return ready
