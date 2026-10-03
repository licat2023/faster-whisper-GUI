"""同一 whisper.cpp 程序和模型的 CPU/Vulkan 实际对比，包含进程启动耗时。"""

import argparse
import json
from pathlib import Path
import statistics
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from faster_whisper_GUI.backends.whisper_cpp import WhisperCppBackend
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "vulkan", "rocm"], default=["cpu", "vulkan"])
    parser.add_argument("--report", default=str(ROOT / ".cache/reports/vulkan.json"))
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs 必须为正数")
    reports = []
    (ROOT / ".cache/reports").mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        backend = WhisperCppBackend(args.executable, args.model, device=device, timeout=1800)
        runs = []
        for _ in range(args.runs):
            started = time.perf_counter()
            result = backend.recognize(args.audio, {"language": args.language, "beam_size": args.beam_size, "best_of": 1})
            elapsed = time.perf_counter() - started
            if device == "vulkan" and "using Vulkan" not in backend.last_diagnostics:
                raise RuntimeError("程序没有确认使用 Vulkan 后端，不能计为 GPU 测试")
            if device == "rocm" and ("ROCm" not in backend.last_diagnostics or not re.search(r"using (?:ROCm|CUDA)\d+ backend", backend.last_diagnostics)):
                raise RuntimeError("程序没有确认 ROCm GPU 推理，不能计为 ROCm 测试")
            timing = {name.strip().replace(" ", "_"): float(value) / 1000
                      for name, value in re.findall(r"whisper_print_timings:\s*(load time|total time)\s*=\s*([\d.]+) ms", backend.last_diagnostics)}
            runs.append({"elapsed_seconds": elapsed, "native_timings_seconds": timing,
                         "text": "".join(segment.text for segment in result.segments)})
        reports.append({"backend": "whisper.cpp", "device": device, "runs": runs,
                        "model": str(Path(args.model).resolve()),
                        "median_seconds": statistics.median(run["elapsed_seconds"] for run in runs),
                        "audio_seconds": result.info.duration, "language": args.language, "beam_size": args.beam_size, "threads": 4,
                        "timing_scope": "process start, model load, inference and JSON output"})
        (ROOT / f".cache/reports/whisper-cpp-{device}.log").write_text(backend.last_diagnostics, encoding="utf-8")
        print(json.dumps(reports[-1]), flush=True)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
