"""whisper.cpp CLI 适配；可选的独立进程后端，不导入 GPU Python 库。"""

import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import threading
import math

from faster_whisper_GUI.domain.segments import segment_Transcribe
from .base import Recognition, TranscriptionInfo
from .audio import prepared_audio


class WhisperCppBackend:
    sample_rate = 16000

    def __init__(self, executable, model, device="vulkan", threads=4, timeout=300):
        if device not in {"cpu", "vulkan", "rocm"}:
            raise ValueError("whisper.cpp 实验后端仅支持 cpu / vulkan / rocm")
        self.executable = str(Path(executable).resolve())
        self.model = str(Path(model).resolve())
        self.device = device
        self.threads = threads
        self.timeout = timeout
        self.last_diagnostics = ""
        self.process = None
        self._recognition_lock = threading.Lock()

    def recognize(self, audio, options):
        with self._recognition_lock:
            return self._recognize_input(audio, options)

    def _recognize_input(self, audio, options):
        if isinstance(audio, (str, Path)):
            with prepared_audio(audio) as normalized:
                return self._recognize(normalized, options)
        return self._recognize(audio, options)

    def _recognize(self, audio, options):
        import soundfile as sf
        supported = {"language", "task", "initial_prompt", "word_timestamps", "beam_size", "best_of"}
        unsupported = set(options) - supported
        if unsupported:
            raise ValueError(f"whisper.cpp 后端尚未映射参数：{sorted(unsupported)}")
        if options.get("word_timestamps"):
            raise ValueError("CLI 适配暂不提供词级时间戳；请使用片段时间戳或后续对齐")
        task = options.get("task", "transcribe")
        if task not in {"transcribe", "translate"}:
            raise ValueError(f"不支持的识别任务：{task}")
        with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as temporary:
            output = str(Path(temporary) / "result")
            if isinstance(audio, (str, Path)):
                input_path = str(Path(audio).resolve())
                audio_info = sf.info(input_path)
                duration = audio_info.frames / audio_info.samplerate
            else:
                import numpy as np
                samples = np.asarray(audio, dtype=np.float32)
                if samples.ndim != 1 or not np.isfinite(samples).all():
                    raise ValueError("音频数组必须为 16kHz 有限的单声道样本")
                input_path = str(Path(temporary) / "input.wav")
                sf.write(input_path, samples, self.sample_rate, subtype="PCM_16")
                duration = len(samples) / self.sample_rate
            command = [self.executable, "-m", self.model, "-f", input_path,
                       "-oj", "-of", output, "-t", str(self.threads),
                       "-l", options.get("language") or "auto"]
            if self.device == "cpu":
                command.append("-ng")
            for key, flag in [("beam_size", "-bs"), ("best_of", "-bo")]:
                if key in options:
                    value = int(options[key])
                    if value < 1:
                        raise ValueError(f"{key} 必须为正数")
                    command.extend([flag, str(value)])
            if task == "translate":
                command.append("-tr")
            if options.get("initial_prompt"):
                command.extend(["--prompt", options["initial_prompt"]])
            env = os.environ.copy()
            if self.device == 'rocm':
                rocm_bin = Path(env.get('HIP_PATH', r'C:\Program Files\AMD\ROCm\7.1')) / 'bin'
                env['PATH'] = str(rocm_bin) + os.pathsep + env.get('PATH', '')
                env.setdefault('ROCBLAS_USE_HIPBLASLT', '1')
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process = self.process
            try:
                _, diagnostics = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                _, self.last_diagnostics = process.communicate()
                raise
            finally:
                self.process = None
            self.last_diagnostics = diagnostics
            if process.returncode:
                raise RuntimeError(f"whisper.cpp 失败（{process.returncode}）：{diagnostics[-2000:]}")
            self._verify_device(diagnostics)
            try:
                payload = json.loads(Path(output + ".json").read_text(encoding="utf-8"))
                segments = [segment_Transcribe(start=item["offsets"]["from"] / 1000,
                                              end=item["offsets"]["to"] / 1000,
                                              text=item["text"], words=[])
                            for item in payload["transcription"]]
                if any(not math.isfinite(s.start) or not math.isfinite(s.end) or s.start < 0 or s.end < s.start for s in segments):
                    raise ValueError('invalid cue times')
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise RuntimeError(f'whisper.cpp 输出无效：{error}；诊断：{diagnostics[-2000:]}') from error
            language = payload.get("result", {}).get("language", options.get("language") or "en")
            # CLI 不暴露语言概率或 VAD 后时长；未知值保持显式 None。
            info = TranscriptionInfo(language, None, duration, None)
            return Recognition(segments, info)

    def _verify_device(self, diagnostics):
        if self.device == 'vulkan' and not re.search(r'using Vulkan|ggml_vulkan[^\n]*(?:Found [1-9]\d* Vulkan devices|Using Vulkan device)', diagnostics, re.I):
            raise RuntimeError('当前 whisper-cli 未使用 Vulkan，请检查程序路径与 GPU 驱动')
        if self.device == 'rocm' and not (re.search(r'ROCm|HIP', diagnostics, re.I) and re.search(r'using (?:ROCm|CUDA|HIP)\d* backend', diagnostics, re.I)):
            raise RuntimeError('当前 whisper-cli 未使用 ROCm，请选择 ROCm 编译版本')

    def cancel(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
