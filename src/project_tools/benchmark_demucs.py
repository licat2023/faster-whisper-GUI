"""用现有 Demucs 分块方法建立同输入的长音频基线。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["TORCH_HOME"] = str(ROOT / ".cache/models/torch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    from faster_whisper_GUI.runtime.inference import prepare_inference_runtime
    prepare_inference_runtime()
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
    from faster_whisper_GUI.tasks.separation import DemucsWorker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, rate = sf.read(args.audio, always_2d=True, dtype="float32")
    if rate != 44100 or audio.shape[1] != 2:
        raise ValueError("Demucs 基线输入须为 44.1 kHz 双声道")
    started = time.perf_counter()
    model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    load = time.perf_counter() - started
    worker = DemucsWorker(None, [], 0, "", segment=7.8, overlap=0.1)
    worker.is_running = True
    folder = ROOT / ".cache/experiments/demucs-output"
    folder.mkdir(parents=True, exist_ok=True)
    if args.runs < 1:
        raise ValueError("runs 必须为正数")
    runs = []
    for index in range(args.runs):
        started = time.perf_counter()
        output = worker.separate_sources(model, audio.T[None], segment=7.8, overlap=0.1,
                                         device=device, sample_rate=rate)[0].cpu().numpy()
        if not np.isfinite(output).all() or output.shape[-1] != len(audio):
            raise ValueError("Demucs 输出样本或时长无效")
        for stem, samples in zip(model.sources, output):
            sf.write(folder / f"{Path(args.audio).stem}_{stem}.wav", samples.T, rate, subtype="FLOAT")
        elapsed = time.perf_counter() - started
        runs.append({"elapsed_seconds": elapsed})
        print(f"Run {index + 1}: {elapsed:.3f}s", flush=True)
    report = {"model": "HDEMUCS_HIGH_MUSDB_PLUS", "torch_device": device,
              "load_seconds": load, "elapsed_seconds": statistics.median(run["elapsed_seconds"] for run in runs), "runs": runs, "audio_seconds": len(audio) / rate,
              "segment_seconds": 7.8, "overlap": 0.1, "precision": "float32",
              "stems": list(model.sources), "output_frames": int(output.shape[-1]),
              "output_directory": str(folder)}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
