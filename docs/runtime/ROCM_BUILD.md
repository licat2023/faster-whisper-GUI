> 本文保留历史构建与性能验证记录。原 `.probe` 实验脚本已归档到工作区外，位置见 [清理说明](../CLEANUP.md)。当前运行时初始化代码位于 `src/faster_whisper_GUI/runtime/rocm.py` 与 `inference.py`，启动入口为 `app.py`。

# AMD ROCm 支持可行性验证报告（v3 · 实证完成）

**验证机器**：Ryzen 7 7840H + Radeon 780M (gfx1103) / 13.7 GB / Windows 11
**HIP SDK**：`C:\Program Files\AMD\ROCm\7.1`（7.1.51803）

> **v3 修订**：v1 说"封死"，v2 说"很贵"。**v3 是实测结果：已经跑通了。**
> 成功构建 ROCm 版 CTranslate2，在 780M 上完成真实转写，**比 CPU 快 3.21 倍，识别质量一致**。

---

## 1. 结论

| | 状态 |
|---|---|
| 构建 ROCm 版 CTranslate2 | ✅ **成功**（含 gfx1103 HIP 内核） |
| 780M GPU 转写 | ✅ **成功**（`large-v3-turbo` 15.33x 实时） |
| 相对 CPU 加速 | **3.21×** |
| 识别质量 | ✅ **与 CPU 一致** |
| 构建耗时 | **4.8 分钟**（首次配置调试约 2 小时） |
| 代价 | 运行需 254 MB 外部 DLL + 系统级 HIP SDK；一个未文档化的环境变量绕行 |

---

## 2. 实测性能（58.6 秒真实语音，large-v3-turbo）

| 配置 | 加载 | 解码 | RTF | 实时倍率 |
|---|---|---|---|---|
| CPU int8 16线程（GUI 当前用 4） | 10.9 s | 12.3 s | 0.209 | 4.78x |
| **GPU float16** | 6.4 s | **4.0 s** | **0.068** | **14.62x** |
| **GPU int8_float16** | 7.7 s | **3.8 s** | **0.065** | **15.33x** |

**GPU / CPU = 3.21×**（int8_float16）

识别质量三者完全一致：

```
CPU / GPU float16 / GPU int8_float16 均输出:
"The quick brown fox jumps over the lazy dog. This is a test of the speech
 recognition system running on an AMD Ryzen processor. ..."
```

### 小模型会误导判断

同一台机器上 `tiny` 模型 GPU 达 30x 实时、CPU 达 56x —— GPU 反而更慢。
**只有在 `medium` 以上的真实模型上 GPU 才有优势**，因为小模型的固定开销（权重传输、启动）压倒了算力收益。

---

## 3. 构建配方（可复现）

### 3.1 前置件（本机均已具备）

| 组件 | 本机路径 |
|---|---|
| ROCm HIP SDK 7.1 | `C:\Program Files\AMD\ROCm\7.1` |
| MSVC 工具集 | VS 18 Community（`D:\Program Files\DevolopPorgam\...`） |
| Windows SDK | `D:\Windows Kits\10` |
| Intel oneAPI（MKL + oneDNN） | `C:\Program Files (x86)\Intel\oneAPI` |
| CMake / Ninja | `D:\Environment\w64devkit\bin` |

### 3.2 四个必须的修正（踩坑记录）

| # | 问题 | 现象 | 修正 |
|---|---|---|---|
| 1 | **MSVC 版本过高** | `__device__ function 'isgreater' cannot overload __host__ __device__` | 用 `-vcvars_ver=14.44.35207`（14.51 冲突） |
| 2 | **空格路径被截断** | `clang++: no such file or directory: 'Files/AMD/ROCm/7.1'` | 用 8.3 短路径 `C:\PROGRA~1\AMD\ROCm\7.1` |
| 3 | **设备库找不到** | `cannot find ROCm device library` | `--rocm-device-lib-path=...\amdgcn\bitcode` |
| 4 | **OpenMP runtime 名** | `Invalid OpenMP runtime COMPILER` | 正确值是 `COMP`（合法：INTEL/COMP/NONE） |

### 3.3 构建命令

```powershell
# 环境
cmd /c '"...\vcvars64.bat" -vcvars_ver=14.44.35207 >nul 2>&1 && set' | % { ... }   # 关键：14.44
$env:ROCM_PATH = "C:\PROGRA~1\AMD\ROCm\7.1"          # 关键：短路径
$env:GPU_RUNTIME = "HIP"; $env:HIP_PLATFORM = "amd"

cmake -GNinja -S ct2-src -B ct2-src/build `
  -DCMAKE_BUILD_TYPE=Release `
  -DWITH_HIP=ON -DWITH_DNNL=ON -DWITH_MKL=OFF -DWITH_CUDA=OFF `
  -DOPENMP_RUNTIME=COMP `
  -DCMAKE_HIP_ARCHITECTURES=gfx1103 `
  -DCMAKE_HIP_FLAGS="--rocm-path=$env:ROCM_PATH --rocm-device-lib-path=$env:ROCM_PATH\amdgcn\bitcode" `
  -DCMAKE_PREFIX_PATH="...oneAPI\dnnl\latest;...oneAPI\mkl\latest"

cmake --build ct2-src/build --parallel 8      # 实测 4.8 分钟
```

**省掉的一步**：PR 作者脚本会源码编译 oneDNN（拉 Intel oneAPI 安装包 + 构建），
但 oneAPI 自带的 `dnnl.lib` + `dnnl-config.cmake` **直接可用**，这一步可以完全跳过。

### 3.4 产物

| 文件 | 大小 |
|---|---|
| `ctranslate2.dll` | 50.8 MB |
| `ctranslate2.lib` | 46.0 MB |

**后端验证**（对比官方 PyPI wheel）：

| 指标 | 官方 PyPI wheel | **自建** |
|---|---|---|
| 导入的 GPU 库 | `cudnn64_9.dll` | **`libhipblas.dll` + `amdhip64_7.dll`** |
| `hip` 符号 | — | **245,253 次** |
| `gfx1103` 架构串 | — | **40 次** |
| `cublas` / `cudnn` | 35 / 1 | **2 / 0** |

---

## 4. 关键坑：rocBLAS 缺 gfx1103 内核

### 4.1 问题

构建成功后加载模型即报：

```
rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat:
               No such file or directory for GPU arch : gfx1103
rocBLAS error: Could not initialize Tensile host:
regex_error(error_backref): The expression contained an invalid back reference.
```

### 4.2 根因：AMD 在 HIP SDK 里漏了 gfx1103 的 rocBLAS 内核

全盘统计内核文件分布：

| 架构 | `bin\rocblas\library` | `bin\hipblaslt\library` |
|---|---|---|
| gfx1100 | ✅ 96 个 | ✅ 95 个 |
| gfx1101 / gfx1102 | ✅ | ✅ |
| gfx1150 / gfx1151 | ✅ | ✅ |
| gfx1200 / gfx1201 | ✅ | ✅ |
| **gfx1103** | ❌ **0 个** | ✅ **95 个** |

**AMD 给 hipBLASLt 编了 gfx1103 内核，但没给 rocBLAS 编。** 而 CTranslate2 链接的是
hipBLAS（底层走 rocBLAS），于是卡死在这个不对称上。

### 4.3 解法：让 rocBLAS 改用 hipBLASLt 路径

rocBLAS 内置了 `ROCBLAS_USE_HIPBLASLT` 开关（从 `rocblas.dll` 字符串表发现，未在常规文档中强调）：

```python
import os
os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"      # 必需
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"  # 可选，再快约 1.5x
import ctranslate2
```

### 4.4 绕过方案实测对比

| 方案 | 结果 |
|---|---|
| A: 基线（无绕过） | ❌ rocBLAS 报错 |
| B: `HSA_OVERRIDE_GFX_VERSION=11.0.0` | ❌ 仍失败（rocBLAS 不认 gfx1103） |
| C: `ROCBLAS_TENSILE_LIBPATH` | ❌ 仍失败 |
| **D: `ROCBLAS_USE_HIPBLASLT=1`** | ✅ 2.97s，19.75x |
| **E: D + `HSA_OVERRIDE_GFX_VERSION`** | ✅ **1.95s，30.08x**（最优） |

> A/B/C 的失败说明这不是路径问题，而是 **rocBLAS 架构分派层面的缺失**。

---

## 5. 运行部署清单

### 5.1 需要的 DLL（约 254 MB）

| 来源 | 文件 | 大小 |
|---|---|---|
| ROCm | `amdhip64_7.dll` | 17.3 MB |
| ROCm | `rocblas.dll` | 39.2 MB |
| ROCm | `amd_comgr0701.dll` | 109.9 MB |
| ROCm | `libhipblaslt.dll` | 5.7 MB |
| ROCm | `libhipblas.dll` | 0.7 MB |
| ROCm | `hiprtc0701.dll` + `hiprtc-builtins0701.dll` | 2.9 MB |
| Intel | `dnnl.dll` | 72.6 MB |
| Intel | `tbb12.dll` | 0.3 MB |
| Intel | `sycl8.dll` | 小 |

### 5.2 加载顺序要求

`ROCBLAS_USE_HIPBLASLT` **必须在 `import ctranslate2` 之前设置**。对本仓库意味着：

- 不能只改模型加载模块，必须在**进程启动早期**通过 `runtime/rocm.py` 设置环境变量
- 需要把上述 254 MB DLL 放进 `ctranslate2` 包目录或加入 `os.add_dll_directory()`

### 5.3 硬性前置条件

- **系统级安装 AMD HIP SDK**（`C:\Program Files\AMD\ROCm\7.1`）。Windows ROCm 源码不完全开源，
  无法做成自包含 wheel —— 这是 AMD 官方文档承认的限制。
- 因此**不能像 CUDA 版那样直接打包分发**，用户必须先装 HIP SDK。

---

## 6. 完整代价清单

| 项目 | 代价 |
|---|---|
| 前置件 | VS BuildTools + Windows SDK + Intel oneAPI（**用户已装好**） |
| HIP SDK | **用户已装** |
| 源码构建 | 4.8 分钟（首次环境调试约 2 小时） |
| 磁盘 | 构建目录 360 MB + 运行 DLL 254 MB |
| 唯一绕行 | `ROCBLAS_USE_HIPBLASLT=1`（未文档化，且依赖 AMD 未来是否补 rocBLAS gfx1103 内核） |
| 维护风险 | HIP SDK 升级可能改变行为；该开关非公开 API |

---

## 7. 对本仓库的改造建议

### 已完成可复用的部分（本次验证）
- `device="cuda"` 在 HIP 构建下即指向 AMD GPU —— **代码层无需区分 CUDA/ROCm 设备字符串**
- 现有 `de_mucs.py:49`、`whisper_x.py:58` 的 `torch.cuda.is_available()` 判断可原样复用

### 仍待完成
| 优先级 | 事项 | 成本 |
|---|---|---|
| P0 | **修 `device="cuda"` 硬报错** —— 无 GPU 时降级 CPU（现存缺陷，与 ROCm 无关） | 0.5 天 |
| P0 | `cpu_threads` 默认改 `min(逻辑核数,16)` —— 实测 **+54%** | 0.5 天 |
| P1 | 启动时探测 ROCm：设置环境变量 + `os.add_dll_directory()` | 1 天 |
| P1 | 设备扫描填入实际后端名（"AMD Radeon 780M (ROCm)"），无 GPU 时置灰 | 1 天 |
| P2 | WhisperX 对齐 / Demucs / VAD 的 GPU 化 | 需 AMD nightly `torch[device-gfx1103]`（见 v2 报告第 5 节），**本次未验证** |
| P2 | 打包：把 254 MB DLL 随程序分发 + HIP SDK 前置检测 | 2–3 天 |

### 一句实话

**技术验证已经完成，但产品化仍不便宜**：需要让每个用户先装 HIP SDK（约 2 GB），
程序再带上 254 MB DLL，且依赖一个未文档化的环境变量绕行 AMD 的内核缺失。
对**个人自用**这是完全可行的；对**对外分发**则取决于能否接受上述前置条件。

---

## 8. 复现步骤

```powershell
# 1. 前置检查
& "C:\Program Files\AMD\ROCm\7.1\bin\hipInfo.exe"        # 应显示 gcnArchName: gfx1103

# 2. 克隆源码（必须带子模块）
git clone --depth 1 --branch v4.8.2 --recursive https://github.com/OpenNMT/CTranslate2.git

# 3. 按 3.3 节配置 + 构建（4.8 分钟）

# 4. 部署 DLL 并替换
copy <build>\ctranslate2.dll  <venv>\Lib\site-packages\ctranslate2\

# 5. 运行（见 .hip-verify\bench_final.py）
```

验证脚本位于 `.hip-verify/`：

| 脚本 | 用途 |
|---|---|
| `check_hip.py` | 验证 DLL 里是否有 HIP 后端（符号/导入表对比） |
| `diag_deps.py` / `diag_dnnl.py` | 依赖链诊断 |
| `test_workarounds2.py` | 五种 rocBLAS 绕过方案对比 |
| `bench_final.py` | **GPU vs CPU 最终性能对比** |

构建脚本位于 `D:\rocm-ct2-build\`：`configure_final.ps1`、`build_hip.ps1`。

---

## 附：信息来源

- [OpenNMT/CTranslate2#1989](https://github.com/OpenNMT/CTranslate2/pull/1989) —— HIP 支持 PR（已 merge，含 Windows 构建脚本）
- [ROCm/TheRock SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) —— gfx1103 Windows 状态为 Release Ready
- [ROCm HIP SDK Windows 发布策略](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/release-versioning.html) —— 源码不完全开源
- [pytorch/pytorch#159520](https://github.com/pytorch/pytorch/issues/159520) —— Windows ROCm 的 PyTorch 侧进展
- [ROCm/TheRock#2011](https://github.com/ROCm/TheRock/issues/2011) / [#2164](https://github.com/ROCm/TheRock/issues/2164) —— gfx1103 在 TheRock 中的反复
