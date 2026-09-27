# ===================================================================
#  faster-whisper-GUI 源码运行环境搭建（含 AMD ROCm 支持）
#
#  用法（在仓库根目录）：
#      powershell -ExecutionPolicy Bypass -File setup.ps1
#
#  作用：
#     1. 用 uv 按 pyproject.toml / uv.lock 创建并同步虚拟环境
#     2. 把自建的 ROCm 版 ctranslate2.dll 放进 site-packages
#     3. 校验 ROCm 运行时能否被探测到
#
#  依赖由 pyproject.toml + uv.lock 管理（旧的 requirements.txt 已删除）。
#  改依赖请改 pyproject.toml，然后跑 `uv lock`。
#
#  历史：本脚本原先位于 .venv\setup.ps1，而 .venv 是被 .gitignore 忽略的，
#  等于安装脚本从未纳入版本控制。现已移到仓库根目录。
# ===================================================================
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$VENV       = ".venv"
$PY         = "$VENV\Scripts\python.exe"
$BUILT_DLL  = "D:\rocm-ct2-build\ct2-src\build\ctranslate2.dll"
$CT2_SRC    = "D:\rocm-ct2-build\ct2-src"
$CT2_DIR    = "$VENV\Lib\site-packages\ctranslate2"
$DEST_DLL   = Join-Path $CT2_DIR "ctranslate2.dll"
$BACKUP_DLL = "$DEST_DLL.official-backup"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash
}
function Get-ShortHash([string]$Path) {
    return (Get-Sha256 $Path).Substring(0, 16)
}

Write-Host "=== faster-whisper-GUI 环境搭建 ===" -ForegroundColor Cyan
Write-Host ""

# ---------- 0. 前置检查 ----------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[错误] 未找到 uv。请先安装：" -ForegroundColor Red
    Write-Host '       powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}
if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "uv.lock")) {
    Write-Host "[错误] 缺少 pyproject.toml 或 uv.lock" -ForegroundColor Red
    exit 1
}

# ---------- 1. 同步环境 ----------
# uv 会读取 .python-version（当前 3.13）自动准备解释器。
# --extra rocm 装上 AMD 的 ROCm 版 torch 与 gfx1103 设备内核。
Write-Host "[1/4] 用 uv 同步环境（首次约需数分钟）" -ForegroundColor Cyan
Write-Host "      依赖来源：pyproject.toml + uv.lock"
uv sync --extra rocm
if ($LASTEXITCODE -ne 0) {
    throw "uv sync 失败（退出码 $LASTEXITCODE）"
}

# ---------- 2. 放置 ROCm 版 ctranslate2.dll ----------
Write-Host ""
Write-Host "[2/4] 放置 ROCm 版 ctranslate2.dll" -ForegroundColor Cyan

if (-not (Test-Path $CT2_DIR)) {
    throw "未找到 ctranslate2 包目录：$CT2_DIR（依赖可能未装成功）"
}

if (-not (Test-Path $BUILT_DLL)) {
    Write-Warning "未找到自建 DLL：$BUILT_DLL"
    Write-Warning "请先按 .probe\FEASIBILITY.md 第 3 节构建，或手动放置 ROCm 版 ctranslate2.dll"
}
else {
    $hipHash = Get-Sha256 $BUILT_DLL
    $hipMB   = [math]::Round((Get-Item $BUILT_DLL).Length / 1MB, 1)
    $hipTag  = if (Test-Path $CT2_SRC) { (git -C $CT2_SRC describe --tags 2>$null) } else { "" }

    # 版本配对检查：自建 DLL 必须与 wheel 里的 _ext.*.pyd 同版本
    $extFile = Get-ChildItem $CT2_DIR -Filter "_ext.*.pyd" -ErrorAction SilentlyContinue |
               Select-Object -First 1
    $wheelVerLine = (Get-Content (Join-Path $CT2_DIR "version.py") -ErrorAction SilentlyContinue |
                     Select-String '__version__').Line
    $wheelVer = if ($wheelVerLine) { ($wheelVerLine -replace '.*=\s*"','' -replace '".*','') } else { "" }

    Write-Host "  自建 DLL   : $hipTag  ($hipMB MB)"
    if ($wheelVer)   { Write-Host "  wheel 版本 : $wheelVer" }
    if ($extFile)    { Write-Host "  Python 绑定: $($extFile.Name)" }
    if ($hipTag -and $wheelVer -and ($hipTag -replace '^v','') -ne $wheelVer) {
        Write-Warning "  自建 DLL 的版本标签（$hipTag）与 wheel 版本（$wheelVer）不一致，可能出现 ABI 不匹配！"
    }

    # ---- 备份：只备份"确认不是自建 HIP 版"的文件 ----
    # 旧实现只要备份文件不存在，就无条件把当前 DLL 存成 .official-backup。
    # 后果：若脚本第一次运行时 DLL 已经是 HIP 构建，就把 HIP DLL 备份成了
    # "官方备份"，依赖该名字回滚等于没回滚（本仓库实际踩过这个坑：
    # .official-backup 与自建 DLL 哈希完全相同）。
    $destIsHip = (Test-Path $DEST_DLL) -and ((Get-Sha256 $DEST_DLL) -eq $hipHash)

    if ($destIsHip) {
        Write-Host "  目标当前已是自建 HIP 版（哈希一致），无需备份"
    }
    elseif (Test-Path $DEST_DLL) {
        if (-not (Test-Path $BACKUP_DLL)) {
            Copy-Item $DEST_DLL $BACKUP_DLL -Force
            Write-Host "  已备份原 DLL -> $(Split-Path $BACKUP_DLL -Leaf)  (sha256:$(Get-ShortHash $DEST_DLL))"
        }
        else {
            $bakHash = Get-Sha256 $BACKUP_DLL
            if ($bakHash -eq $hipHash) {
                # 备份其实就是 HIP 版的副本，回滚会拿到同一个文件
                Write-Warning "  现有备份与自建 DLL 哈希相同（坏备份），用当前原 DLL 重建"
                Copy-Item $DEST_DLL $BACKUP_DLL -Force
                Write-Host "  已重建备份 (sha256:$(Get-ShortHash $DEST_DLL))"
            }
            elseif ($bakHash -eq (Get-Sha256 $DEST_DLL)) {
                Write-Host "  备份已存在且有效，保留"
            }
            else {
                # 与当前 DLL 是不同版本：留档而不是直接丢弃
                $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
                Move-Item $BACKUP_DLL "$BACKUP_DLL.$stamp" -Force
                Copy-Item $DEST_DLL $BACKUP_DLL -Force
                Write-Host "  旧备份版本不同，已留档为 $(Split-Path $BACKUP_DLL -Leaf).$stamp"
                Write-Host "  并重新备份当前版本 (sha256:$(Get-ShortHash $DEST_DLL))"
            }
        }
    }

    # ---- 修复已存在的坏备份 ----
    # 目标已经是 HIP 版时没有"原 DLL"可备份，但历史上可能留下过一个坏备份。
    # 这里在同目录里找一份真正的官方 DLL 来修复它。
    if (Test-Path $BACKUP_DLL) {
        $bakHash = Get-Sha256 $BACKUP_DLL
        if ($bakHash -eq $hipHash) {
            Write-Warning "  现有 .official-backup 与自建 DLL 内容相同 —— 它不是可用的官方版备份"
            $cand = Get-ChildItem $CT2_DIR -Filter "ctranslate2.dll.*" -File -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -ne $BACKUP_DLL } |
                    Where-Object { (Get-Sha256 $_.FullName) -ne $hipHash } |
                    Select-Object -First 1
            if ($cand) {
                Copy-Item $cand.FullName $BACKUP_DLL -Force
                Write-Host "  已用 $($cand.Name) 重建有效备份 (sha256:$(Get-ShortHash $BACKUP_DLL))" -ForegroundColor Green
            }
            else {
                Write-Warning "  同目录下没有其它版本 DLL，无法重建。下次重装 ctranslate2 时会产生官方 DLL。"
            }
        }
    }

    Copy-Item $BUILT_DLL $DEST_DLL -Force

    # 落盘校验：确认真的换成功了，而不是"以为换成功了"
    $afterHash = Get-Sha256 $DEST_DLL
    if ($afterHash -ne $hipHash) {
        throw "替换后哈希不一致！期望 $hipHash，实际 $afterHash"
    }
    Write-Host "  已放置: ctranslate2.dll ($hipMB MB, sha256:$($afterHash.Substring(0,16)))" -ForegroundColor Green

    # 记录清单，方便日后核对（uv sync 重装 ctranslate2 会覆盖此 DLL）
    @{
        dllSha256 = $afterHash
        ct2Tag    = $hipTag
        placedAt  = (Get-Date -Format 's')
        sourceDll = $BUILT_DLL
    } | ConvertTo-Json | Set-Content (Join-Path $CT2_DIR "rocm-dll.json") -Encoding UTF8
}

# ---------- 3. 校验 ----------
Write-Host ""
Write-Host "[3/4] 校验 ROCm 运行时" -ForegroundColor Cyan
$checkScript = Join-Path $VENV "verify_rocm.py"
@'
import sys, os, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("u", os.path.join(ROOT, "faster_whisper_GUI", "util.py"))
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

if not util.setupROCm():
    print("  [失败] 未探测到 HIP SDK，AMD ROCm 选项不会出现在下拉框中")
    sys.exit(1)

import ctranslate2
count = ctranslate2.get_cuda_device_count()
print(f"  ctranslate2 {ctranslate2.__version__}")
print(f"  可见 GPU 设备数: {count}")
if count < 1:
    print("  [失败] 未检测到 GPU 设备")
    sys.exit(1)
print("  [成功] ROCm 运行时可用")
'@ | Set-Content $checkScript -Encoding UTF8
& $PY $checkScript
$checkExit = $LASTEXITCODE
Remove-Item $checkScript -Force -ErrorAction SilentlyContinue

# ---------- 4. 完成 ----------
Write-Host ""
Write-Host "[4/4] 完成" -ForegroundColor Cyan
if ($checkExit -eq 0) {
    Write-Host "=== 环境就绪 ===" -ForegroundColor Green
    Write-Host "    双击 启动GUI.bat"
    Write-Host "    或运行 uv run FasterWhisperGUI.py"
    Write-Host ""
    Write-Host "  提示：uv sync 是精确同步。若之后 ctranslate2 被重装，" -ForegroundColor Yellow
    Write-Host "        自建 DLL 会被官方版覆盖，重跑本脚本即可恢复。" -ForegroundColor Yellow
}
else {
    Write-Host "=== 环境就绪但 ROCm 校验未通过 ===" -ForegroundColor Yellow
    Write-Host "    GUI 仍可启动，但设备下拉框里不会有 AMD ROCm 选项。"
    Write-Host "    请确认已安装 AMD HIP SDK 与 Intel oneAPI。"
}
