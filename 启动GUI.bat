@echo off
REM ===================================================================
REM  faster-whisper-GUI launcher (source run, with AMD ROCm support)
REM
REM  Prerequisites:
REM    1. AMD HIP SDK installed (default: C:\Program Files\AMD\ROCm\<ver>)
REM    2. Intel oneAPI installed (provides dnnl for the ROCm ctranslate2.dll)
REM    3. Dependencies installed via setup.ps1 (uv-managed: pyproject.toml + uv.lock)
REM
REM  ROCm environment variables and DLL directories are configured
REM  automatically inside the application runtime - nothing to set here.
REM
REM  NOTE: This file must keep CRLF line endings. cmd.exe mis-parses
REM        LF-only batch files (comments get split and executed).
REM ===================================================================

setlocal

REM Switch to the repository root (this script's directory)
cd /d "%~dp0"

REM Use UTF-8 so the Chinese messages below display correctly
chcp 65001 >nul

REM Check the virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run this first:
    echo         powershell -ExecutionPolicy Bypass -File setup.ps1
    echo.
    pause
    exit /b 1
)

REM ffmpeg: prefer the bundled copy, otherwise rely on PATH
if exist "ffmpeg\bin" (
    set "PATH=%CD%\ffmpeg\bin;%PATH%"
)

echo Starting faster-whisper-GUI ...
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "PYTHONPYCACHEPREFIX=%CD%\.cache\pycache"
".venv\Scripts\python.exe" -m faster_whisper_GUI %*

endlocal
