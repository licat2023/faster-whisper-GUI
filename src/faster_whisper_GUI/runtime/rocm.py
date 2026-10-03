"""AMD ROCm/HIP 探测和进程级 DLL 初始化，不导入 Qt、torch 或 CTranslate2。"""

import glob
import logging
import os
import re
from typing import List, Union

from .paths import PROJECT_ROOT

log = logging.getLogger(__name__)

# AMD ROCm / HIP 支持
#
# 必须在 import ctranslate2 之前调用 setupROCm()，原因：
#   1. ROCBLAS_USE_HIPBLASLT 是 rocBLAS 在「首次初始化」时读取的环境变量，之后再设无效
#   2. ctranslate2.dll 会直接链接 amdhip64_7.dll / libhipblas.dll，加载时就需要能解析到这些 DLL，
#      因此必须先用 os.add_dll_directory() 注册目录
#
# 背景（本机实测，AMD Radeon 780M / gfx1103 / HIP SDK 7.1）：
#   AMD 的 HIP SDK 给 hipBLASLt 编了 gfx1103 内核，却没给 rocBLAS 编，导致 CTranslate2 走 hipBLAS
#   （底层 rocBLAS）时报 "Cannot read .../rocblas/library/TensileLibrary.dat ... for GPU arch : gfx1103"。
#   ROCBLAS_USE_HIPBLASLT=1 让 rocBLAS 改走 hipBLASLt，从而绕开这个内核缺口。
#   该开关是从 rocblas.dll 的字符串表中发现的，并非公开文档化的接口，HIP SDK 升级后可能失效。

# 是否探测到可用的 ROCm 运行时
ROCM_AVAILABLE: bool = False
# 探测到的 ROCm 根目录
ROCM_ROOT: Union[str, None] = None
# 已注册到本进程的 DLL 目录
ROCM_DLL_DIRECTORIES: List[str] = []
_DLL_HANDLES = []  # Keep AddDllDirectory registrations alive for this process.

# 进程级状态标记
#
# 正常启动和 util 兼容层共用本模块的状态。保留进程级标记，兼容旧脚本
# 按文件路径加载运行时模块的情况，保证设备下拉框能读取初始化结果。
ROCM_ENV_FLAG = "FASTER_WHISPER_GUI_ROCM_AVAILABLE"
ROCM_PROCESS_FLAG = "FASTER_WHISPER_GUI_ROCM_PID"

# rocBLAS 在 gfx1103 上缺少内核，需要改走 hipBLASLt（详见上方说明）
ROCM_FORCE_HIPBLASLT: bool = True
# 把 gfx 架构上报覆盖为 gfx1100：gfx1100 与 gfx1103 同属 RDNA3、同为 wave32，
# 可复用 rocBLAS 里已有的 gfx1100 内核。本机实测这一步还能再快约 1.5 倍。
# 若你使用的是官方已支持、rocBLAS 内核齐全的独显（如 RX 7900 XTX / gfx1100），
# 把这里改成 False 可以避免不必要的架构上报覆盖。
ROCM_OVERRIDE_GFX_VERSION: Union[str, None] = "11.0.0"


def findROCmRoot() -> Union[str, None]:
    """
    查找 ROCm / HIP SDK 根目录

    查找顺序：
      1. 程序自带目录下的 "bin"（若随程序分发了 ROCm 运行时 DLL）
      2. 环境变量 ROCM_PATH / HIP_PATH
      3. 标准安装位置 C:/Program Files/AMD/ROCm/<version>

    :return: 找到的根目录；未找到返回 None
    """
    candidates: List[str] = []

    # 1. 程序自带目录（打包分发时可以在这里放一份 ROCm 运行时 DLL）
    base_dir = str(PROJECT_ROOT)
    candidates.append(base_dir)

    # 2. 环境变量
    for env_name in ("ROCM_PATH", "HIP_PATH"):
        env_value = os.environ.get(env_name, "")
        if env_value:
            candidates.append(env_value)

    # 3. 标准安装位置
    amd_root = r"C:\Program Files\AMD\ROCm"
    if os.path.isdir(amd_root):
        try:
            versions = []
            for name in os.listdir(amd_root):
                # 只认形如 "7.1" 的版本目录，跳过 bin / include / lib 等
                if re.fullmatch(r"\d+(\.\d+)*", name):
                    version_key = tuple(int(part) for part in name.split("."))
                    versions.append((version_key, name))
            versions.sort(reverse=True)          # 优先使用最高版本
            for _, name in versions:
                candidates.append(os.path.join(amd_root, name))
        except OSError:
            pass

    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(os.path.join(candidate, "bin", "amdhip64_7.dll")):
            return candidate

    return None


def findDependentDllDirectories(root: str) -> List[str]:
    """
    查找 ctranslate2.dll 的额外 DLL 依赖目录

    ROCm 版 ctranslate2.dll 在 WITH_DNNL=ON 下构建，会直接链接 dnnl.dll；而 oneAPI 的 dnnl.dll
    又依赖 sycl8.dll / tbb12.dll 等，所以必须把 oneAPI 相关目录一并注册，否则加载 ctranslate2
    时会报 "Could not find module ... (or one of its dependencies)"。

    查找顺序：
      1. 程序自带目录下的 "rocm"（随程序分发依赖时使用）
      2. 环境变量 ONEAPI_ROOT
      3. Intel oneAPI 标准安装位置

    :param root: ROCm 根目录
    :return: 需要注册的目录列表
    """
    candidates: List[str] = []

    # 1. 程序自带依赖目录
    base_dir = str(PROJECT_ROOT)
    candidates.append(os.path.join(base_dir, "bin", "rocm"))

    # 2/3. Intel oneAPI
    oneapi_roots: List[str] = []
    if os.environ.get("ONEAPI_ROOT"):
        oneapi_roots.append(os.environ["ONEAPI_ROOT"])
    for candidate in (r"C:\Program Files (x86)\Intel\oneAPI", r"C:\Program Files\Intel\oneAPI"):
        if os.path.isdir(candidate):
            oneapi_roots.append(candidate)

    for oneapi_root in oneapi_roots:
        if not os.path.isdir(oneapi_root):
            continue
        # dnnl.dll 所在目录（含具体版本目录，如 .../dnnl/2025.3/bin）
        for pattern in ("dnnl/*/bin", "dnnl/latest/bin", "*/bin", "*/lib", "bin"):
            # oneAPI 里同时存在 dnnl.dll 与 sycl8.dll/tbb12.dll，注册同级目录即可
            for path in glob.glob(os.path.join(oneapi_root, pattern)):
                if os.path.isdir(path):
                    candidates.append(path)

    # 去重并保持顺序
    seen = set()
    result: List[str] = []
    for path in candidates:
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isdir(path):
            result.append(path)
    return result


def setupROCm() -> bool:
    """
    探测并启用 AMD ROCm / HIP 运行时

    必须在导入 ctranslate2 之前调用，否则环境变量不生效。

    :return: 是否成功启用
    """
    global ROCM_AVAILABLE, ROCM_ROOT

    # 其它模块实例可能已经完成过初始化，直接复用
    if ROCM_AVAILABLE or (os.environ.get(ROCM_ENV_FLAG) == "1"
                          and os.environ.get(ROCM_PROCESS_FLAG) == str(os.getpid())):
        ROCM_AVAILABLE = True
        return True

    os.environ.pop(ROCM_ENV_FLAG, None)
    os.environ.pop(ROCM_PROCESS_FLAG, None)
    root = findROCmRoot()
    if root is None:
        log.info("%s", "[ROCm] 未检测到 HIP SDK，AMD ROCm 选项不可用")
        return False

    bin_dir = os.path.join(root, "bin")
    ROCM_ROOT = root

    # 需要注册的目录：ROCm 运行时 + 依赖库（oneAPI dnnl 等）
    directories = [bin_dir] + findDependentDllDirectories(root)

    # 注册 DLL 目录，使 ctranslate2.dll 能解析到 amdhip64_7.dll / libhipblas.dll /
    # rocblas.dll / dnnl.dll 等
    if hasattr(os, "add_dll_directory"):
        for directory in directories:
            try:
                _DLL_HANDLES.append(os.add_dll_directory(directory))
                ROCM_DLL_DIRECTORIES.append(directory)
            except OSError as error:
                log.error("%s", f"[ROCm] 注册 DLL 目录失败: {directory} -> {error}")
    else:
        ROCM_DLL_DIRECTORIES.extend(directories)

    if not ROCM_DLL_DIRECTORIES:
        log.info("%s", "[ROCm] 没有可用的 DLL 目录")
        return False

    # 同时追加到 PATH，兼容未使用 add_dll_directory 的旧式 DLL 解析
    os.environ["PATH"] = os.pathsep.join(ROCM_DLL_DIRECTORIES) + os.pathsep + os.environ.get("PATH", "")

    # rocBLAS 在 gfx1103 上缺少内核，改走 hipBLASLt 绕开（必须在 rocBLAS 初始化前设置）
    if ROCM_FORCE_HIPBLASLT:
        os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"

    # 覆盖上报的 gfx 架构，以复用 gfx1100 的内核
    if ROCM_OVERRIDE_GFX_VERSION:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", ROCM_OVERRIDE_GFX_VERSION)

    # 供 CTranslate2 / hipBLAS 定位 ROCm
    os.environ.setdefault("ROCM_PATH", root)
    os.environ.setdefault("HIP_PATH", root)

    ROCM_AVAILABLE = True
    # 写入进程级标记，使其它模块实例（见 ROCM_ENV_FLAG 的说明）也能读到本状态
    os.environ[ROCM_ENV_FLAG] = "1"
    os.environ[ROCM_PROCESS_FLAG] = str(os.getpid())
    log.info("%s", f"[ROCm] 已启用 HIP 运行时: {root}")
    log.info("%s", f"[ROCm]   已注册 DLL 目录 {len(ROCM_DLL_DIRECTORIES)} 个:")
    for directory in ROCM_DLL_DIRECTORIES:
        log.info("%s", f"[ROCm]     {directory}")
    log.info("%s", f"[ROCm]   ROCBLAS_USE_HIPBLASLT = {os.environ.get('ROCBLAS_USE_HIPBLASLT')}")
    log.info("%s", f"[ROCm]   HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    return True


def isROCmAvailable() -> bool:
    """
    ROCm / HIP 运行时是否可用

    同时检查模块级变量与进程级环境标记，兼容旧脚本按路径加载的模块实例。
    """
    return ROCM_AVAILABLE or (os.environ.get(ROCM_ENV_FLAG) == "1" and os.environ.get(ROCM_PROCESS_FLAG) == str(os.getpid()))
