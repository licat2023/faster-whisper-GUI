"""本机自建 CTranslate2 的 CPU/ROCm 测量，必须消费惰性片段。"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="Systran/faster-whisper-tiny.en")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--compute-type", default="float32")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--no-condition-on-previous-text", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs 必须为正数")
    from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
    prepare_inference_runtime()
    import ctranslate2
    from faster_whisper import WhisperModel
    import soundfile as sf
    if args.device == "cuda" and ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError("没有可用的 CTranslate2 GPU")
    print(f"Loading {args.model} on {args.device}", flush=True)
    started = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type,
                         cpu_threads=4, num_workers=1, download_root=str(ROOT / ".cache/models/faster-whisper"))
    loaded = time.perf_counter() - started
    runs = []
    for index in range(args.runs):
        started = time.perf_counter()
        segments, info = model.transcribe(args.audio, language=args.language, beam_size=args.beam_size, best_of=args.best_of,
                                         vad_filter=False, word_timestamps=False,
                                         condition_on_previous_text=not args.no_condition_on_previous_text, temperature=0)
        segments = list(segments)
        runs.append({"elapsed_seconds": time.perf_counter() - started,
                     "text": "".join(s.text for s in segments), "segments": len(segments)})
        print(f"Run {index + 1}: {runs[-1]['elapsed_seconds']:.3f}s", flush=True)
    duration = sf.info(args.audio).duration
    median = statistics.median(r["elapsed_seconds"] for r in runs)
    report = {"backend": "faster-whisper", "device": args.device, "compute_type": args.compute_type,
              "actual_compute_type": getattr(model.model, "compute_type", None),
              "model": args.model, "load_seconds": loaded, "audio_seconds": duration,
              "beam_size": args.beam_size, "language": args.language, "threads": 4, "vad": False, "runs": runs,
              "condition_on_previous_text": not args.no_condition_on_previous_text,
              "best_of": args.best_of,
              "median_seconds": median, "rtf": median / duration,
              "ctranslate2_version": ctranslate2.__version__}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
