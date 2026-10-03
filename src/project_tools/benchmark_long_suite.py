"""串行执行长音频实验，避免不同后端争抢 CPU/GPU。"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / ".cache/reports"
AUDIO = ".cache/fixtures/librispeech-long.wav"
jobs = [
    ("faster-whisper-turbo-long-rocm-int8", None, "benchmark_faster_whisper.py", ["--audio", AUDIO, "--model", "CONFIGURED_MODEL", "--device", "cuda", "--compute-type", "int8_float16"]),
    ("qwen-long-gpu", "qwen", "benchmark_qwen.py", ["--audio", AUDIO, "--device", "cuda", "--language", "en", "--runs", "3"]),
    ("qwen-long-cpu", "qwen", "benchmark_qwen.py", ["--audio", AUDIO, "--device", "cpu", "--language", "en"]),
    ("streaming-long-fast", "streaming", "benchmark_streaming.py", ["--audio", AUDIO, "--model-dir", ".cache/models/streaming/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"]),
    ("streaming-long-paced", "streaming", "benchmark_streaming.py", ["--audio", AUDIO, "--model-dir", ".cache/models/streaming/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17", "--pace"]),
    ("community-long", None, "benchmark_diarization.py", ["--audio", AUDIO, "--seconds", "1000", "--runs", "3"]),
    ("community-long-rocm", None, "benchmark_diarization.py", ["--audio", AUDIO, "--seconds", "1000", "--device", "cuda", "--runs", "3"]),
    ("separation-long", "separation", "benchmark_separation.py", ["--audio", ".cache/fixtures/hdemucs_mix_60s.wav", "--runs", "3"]),
    ("demucs-long", None, "benchmark_demucs.py", ["--audio", ".cache/fixtures/hdemucs_mix_60s.wav", "--runs", "3"]),
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "fasterWhisperGUIConfig.json").read_text(encoding="utf-8-sig"))
    configured_model = config["model_param"]["model_path"]
    results = []
    for name, environment, script, arguments in jobs:
        python = ROOT / (f".cache/experiments/{environment}-env/Scripts/python.exe" if environment else ".venv/Scripts/python.exe")
        arguments = [configured_model if value == "CONFIGURED_MODEL" else value for value in arguments]
        command = [str(python), str(ROOT / "src/project_tools" / script), *arguments,
                   "--report", str(REPORTS / f"{name}.json")]
        print(f"Starting {name}", flush=True)
        started = time.perf_counter()
        with (REPORTS / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        results.append({"name": name, "returncode": process.returncode,
                        "process_seconds": time.perf_counter() - started})
        (REPORTS / "long-suite-status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Finished {name}: exit={process.returncode}, process={results[-1]['process_seconds']:.2f}s", flush=True)
    if any(result["returncode"] for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
