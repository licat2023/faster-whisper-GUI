"""按固定音频帧重放文件，测量真实流式会话；不是麦克风端到端延迟测试。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import soundfile as sf
    from faster_whisper_GUI.backends.sherpa import SherpaStreamingBackend
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--report", default=str(ROOT / ".cache/reports/streaming.json"))
    args = parser.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or args.chunk_ms <= 0:
        raise ValueError("测试音频须为 16kHz 单声道；chunk-ms 必须为正数")
    started = time.perf_counter()
    backend = SherpaStreamingBackend.from_directory(args.model_dir)
    load_seconds = time.perf_counter() - started
    session = backend.start()
    frame_size = max(1, int(rate * args.chunk_ms / 1000))
    started = time.perf_counter()
    first = None
    revisions = []
    previous = -1
    for offset in range(0, len(audio), frame_size):
        frame = audio[offset:offset + frame_size]
        if args.pace:
            remaining = (offset + len(frame)) / rate - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        update = session.accept(frame)
        if update.text and first is None:
            first = {"audio_seconds": update.audio_seconds,
                     "wall_seconds": time.perf_counter() - started}
        if update.revision != previous:
            revisions.append(asdict(update))
            previous = update.revision
    final = session.finish()
    elapsed = time.perf_counter() - started
    report = {"backend": "sherpa-onnx", "model_dir": str(Path(args.model_dir).resolve()),
              "chunk_ms": args.chunk_ms, "paced_replay": args.pace,
              "audio_seconds": len(audio) / rate, "load_seconds": load_seconds,
              "elapsed_seconds": elapsed, "first_result": first,
              "final": asdict(final), "updates": revisions}
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "updates"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
