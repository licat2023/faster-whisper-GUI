"""拼接不同 LibriSpeech 验证片段，保留来源、文字和音频哈希。"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
folder = ROOT / ".cache/fixtures"
rows = pq.read_table(folder / "librispeech-validation.parquet").to_pylist()
samples, clips, total = [], [], 0
for row in rows:
    audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("fixture audio must be mono 16 kHz")
    samples.append(audio)
    clips.append({"id": row["id"], "seconds": len(audio) / rate, "text": row["text"]})
    total += len(audio)
    if total >= 90 * rate:
        break
path = folder / "librispeech-long.wav"
if not samples:
    raise ValueError("dataset contains no audio clips")
sf.write(path, np.concatenate(samples), 16000, subtype="PCM_16")
manifest = {"source": "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy",
            "construction": "concatenated distinct validation utterances, no repetition or added silence",
            "seconds": total / rate, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "clips": clips}
(folder / "librispeech-long.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"seconds": manifest["seconds"], "clips": len(clips), "sha256": manifest["sha256"]}))
