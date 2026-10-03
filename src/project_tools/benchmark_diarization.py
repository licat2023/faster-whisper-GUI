"""使用本地配置令牌测试 community-1，不输出凭据。"""

import argparse
import json
from pathlib import Path
import sys
import time
import statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--report", default=str(ROOT / ".cache/reports/community-1.json"))
    args = parser.parse_args()
    from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
    prepare_inference_runtime()
    from whisperx.audio import load_audio, SAMPLE_RATE
    from whisperx.diarize import DiarizationPipeline
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 GPU")
    config = json.loads((ROOT / "fasterWhisperGUIConfig.json").read_text(encoding="utf-8-sig"))
    token = config["setting"]["huggingface_user_token"]
    started = time.perf_counter()
    pipeline = DiarizationPipeline(use_auth_token=token, cache_dir=str(ROOT / ".cache/models"), device=args.device)
    actual_device = str(pipeline.model.device)
    if args.device == "cuda" and not actual_device.startswith("cuda"):
        raise RuntimeError("community-1 未搬到 GPU，不能标记为 GPU 测试")
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    audio = load_audio(args.audio)
    if args.seconds <= 0:
        raise ValueError("测试长度必须大于零")
    audio = audio[:int(args.seconds * SAMPLE_RATE)]
    if args.runs < 1:
        raise ValueError("runs 必须为正数")
    runs = []
    for index in range(args.runs):
        started = time.perf_counter()
        result = pipeline(audio)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        runs.append({"elapsed_seconds": elapsed, "turns": len(result), "speakers": int(result["speaker"].nunique())})
        print(f"Run {index + 1}: {elapsed:.3f}s", flush=True)
    report = {"model": "pyannote/speaker-diarization-community-1", "device": actual_device,
              "load_seconds": load_seconds,
              "fixture": str(Path(args.audio).resolve()), "audio_seconds": len(audio) / SAMPLE_RATE,
              "elapsed_seconds": statistics.median(run["elapsed_seconds"] for run in runs), "runs": runs, "turns": len(result),
              "speakers": int(result["speaker"].nunique()),
              "output_type": type(pipeline.last_output).__name__, "exclusive": pipeline.exclusive}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
