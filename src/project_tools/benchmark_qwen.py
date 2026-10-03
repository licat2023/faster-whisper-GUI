"""在独立 Qwen 实验环境执行实际识别；不修改项目依赖。"""

import argparse
import json
from pathlib import Path
import sys
import time
import statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
    prepare_inference_runtime()
    import torch
    from qwen_asr import Qwen3ASRModel
    from faster_whisper_GUI.backends.qwen import QwenBackend
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--report", default=str(ROOT / ".cache/reports/qwen.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前实验环境没有可用的 GPU")
    print(f"Loading Qwen3-ASR-0.6B on {args.device}", flush=True)
    started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B", dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="cuda:0" if args.device == "cuda" else "cpu",
        cache_dir=str(ROOT / ".cache/models/qwen"), attn_implementation="eager",
        max_new_tokens=args.max_new_tokens, max_inference_batch_size=1,
    )
    load_seconds = time.perf_counter() - started
    if args.runs < 1:
        raise ValueError("runs 必须为正数")
    runs = []
    for index in range(args.runs):
        started = time.perf_counter()
        result = QwenBackend(model).recognize(str(Path(args.audio).resolve()), {"language": args.language})
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        runs.append({"elapsed_seconds": elapsed, "text": result.text})
        print(f"Run {index + 1}: {elapsed:.3f}s", flush=True)
    report = {"backend": "Qwen3-ASR-0.6B", "device": args.device,
              "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
              "load_seconds": load_seconds, "elapsed_seconds": statistics.median(run["elapsed_seconds"] for run in runs), "runs": runs,
              "audio_seconds": result.info.duration, "language": result.info.language,
              "text": result.text,
              "time_stamps_available": result.segment_timestamps,
              "adapter": "QwenBackend"}
    report["max_new_tokens"] = args.max_new_tokens
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
