"""固定 AISHELL-3 仓库版本，准备三位说话人的中文长音频与参考文本。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import urllib.error
import http.client
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".cache/fixtures/chinese"
REVISION = "f20d5db4a31fe779ef07bb1af4ea92da5c786622"
BASE = f"https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/{REVISION}/test"


def fetch(relative):
    path = DEST / "original" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"{BASE}/{relative}", timeout=60) as response:
                    path.write_bytes(response.read())
                break
            except (OSError, http.client.HTTPException):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return path


def main():
    transcript = fetch("content.txt").read_text(encoding="utf-8")
    entries = {}
    for line in transcript.splitlines():
        if not line.strip():
            continue
        filename, tokens = line.split(maxsplit=1)
        tokens = tokens.split()
        if len(tokens) % 2:
            raise ValueError(f"字符/拼音标注不是成对结构：{filename}")
        entries[filename] = "".join(tokens[::2])
    panels = []
    for speaker in ["SSB0693", "SSB0711", "SSB0716"]:
        filenames = [name for name in entries if name.startswith(speaker)][:45]
        with ThreadPoolExecutor(max_workers=4) as executor:
            paths = list(executor.map(fetch, [f"wav/{speaker}/{name}" for name in filenames]))
        chunks, clips, total = [], [], 0
        for path, name in zip(paths, filenames):
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim != 1:
                raise ValueError("参考音频必须是单声道")
            factor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
            clips.append({"id": name, "speaker": speaker, "text": entries[name],
                          "start": total / 16000, "end": (total + len(audio)) / 16000,
                          "source": f"{BASE}/wav/{speaker}/{name}"})
            chunks.append(audio)
            total += len(audio)
            if total >= 90 * 16000:
                break
        if total < 90 * 16000:
            raise ValueError(f"{speaker} 可用音频不足 90 秒")
        output = DEST / f"{speaker}-long.wav"
        sf.write(output, np.concatenate(chunks), 16000, subtype="PCM_16")
        panels.append({"id": speaker, "audio": str(output), "audio_seconds": total / 16000,
                       "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                       "reference": "".join(item["text"] for item in clips), "clips": clips})
        print(f"Prepared {speaker}: {total / 16000:.3f}s, {len(clips)} utterances", flush=True)
    manifest = {"dataset": "AISHELL/AISHELL-3", "revision": REVISION,
                "construction": "distinct test utterances concatenated per speaker; no repeats or added silence",
                "panels": panels}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
