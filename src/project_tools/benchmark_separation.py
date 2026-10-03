"""在独立环境测试新分离模型，并可与提供的参考人声比较。"""

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
    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--model", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--report", default=str(ROOT / ".cache/reports/separation.json"))
    args = parser.parse_args()
    destination = ROOT / ".cache/experiments/separation-output"
    separator = Separator(
        model_file_dir=str(ROOT / ".cache/models/separation"), output_dir=str(destination),
        output_format="WAV", use_soundfile=True, use_autocast=True,
        mdxc_params={"segment_size": 64, "override_model_segment_size": True,
                     "batch_size": 1, "overlap": 2, "pitch_shift": 0},
    )
    started = time.perf_counter()
    separator.load_model(args.model)
    load_seconds = time.perf_counter() - started
    if args.runs < 1:
        raise ValueError("runs 必须为正数")
    runs = []
    for index in range(args.runs):
        started = time.perf_counter()
        files = separator.separate(str(Path(args.audio).resolve()))
        elapsed = time.perf_counter() - started
        runs.append({"elapsed_seconds": elapsed})
        print(f"Run {index + 1}: {elapsed:.3f}s", flush=True)
    outputs = []
    input_seconds = sf.info(args.audio).duration
    for filename in files:
        path = destination / filename
        audio, rate = sf.read(path, always_2d=True)
        if not np.isfinite(audio).all() or not len(audio):
            raise ValueError(f"分离输出包含无效样本：{filename}")
        if abs(len(audio) / rate - input_seconds) > 1 / rate:
            raise ValueError(f"分离输出时长不完整：{filename}")
        outputs.append({"file": str(path), "frames": len(audio), "sample_rate": rate,
                        "channels": audio.shape[1], "rms": float(np.sqrt(np.mean(audio**2)))})
    report = {"model": args.model, "torch_device": str(separator.torch_device),
              "load_seconds": load_seconds, "elapsed_seconds": statistics.median(run["elapsed_seconds"] for run in runs), "runs": runs,
              "audio_seconds": sf.info(args.audio).duration, "segment_size": 64,
              "autocast": True, "outputs": outputs}
    if args.reference:
        vocals = next(item for item in outputs if "(vocals)" in item["file"].lower())
        estimated, rate = sf.read(vocals["file"], always_2d=True)
        reference, reference_rate = sf.read(args.reference, always_2d=True)
        if estimated.shape != reference.shape or rate != reference_rate:
            raise ValueError("参考人声与模型输出的帧数、声道或采样率不一致")
        target = reference.reshape(-1) - reference.mean()
        estimate = estimated.reshape(-1) - estimated.mean()
        projected = target * (np.dot(estimate, target) / (np.dot(target, target) + 1e-12))
        residual = estimate - projected
        report["reference_si_sdr_db"] = float(10 * np.log10((np.dot(projected, projected) + 1e-12) /
                                                           (np.dot(residual, residual) + 1e-12)))
        report["reference"] = str(Path(args.reference).resolve())
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
