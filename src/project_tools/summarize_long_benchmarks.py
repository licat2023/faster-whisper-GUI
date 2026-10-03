"""汇总长音频实测；固定英文文本规范化后记录简单 WER。"""
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / ".cache/reports"


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))


def wer(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1] / len(reference) if reference else None


def main():
    fixture = json.loads((ROOT / ".cache/fixtures/librispeech-long.json").read_text())
    reference = words(" ".join(item["text"] for item in fixture["clips"]))
    results = []
    names = ["whisper-cpp-long", "whisper-cpp-long-rocm", "whisper-cpp-turbo-long",
             "whisper-cpp-turbo-long-rocm", "faster-whisper-long-cpu", "faster-whisper-long-rocm",
             "faster-whisper-turbo-long-cpu", "faster-whisper-turbo-long-rocm",
             "faster-whisper-turbo-long-rocm-int8",
             "qwen-long-gpu", "qwen-long-cpu", "streaming-long-fast", "streaming-long-paced",
             "community-long", "community-long-rocm", "separation-long", "demucs-long"]
    for name in names:
        path = FOLDER / f"{name}.json"
        if not path.exists():
            results.append({"report": name, "status": "missing"})
            continue
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        for item in loaded if isinstance(loaded, list) else [loaded]:
            if not isinstance(item, dict):
                results.append({'report': name, 'status': 'invalid', 'reason': 'report must be an object'})
                continue
            seconds = item.get("median_seconds", item.get("elapsed_seconds"))
            duration = item.get('audio_seconds')
            if not isinstance(seconds, (int, float)) or seconds <= 0 or not isinstance(duration, (int, float)) or duration <= 0:
                results.append({'report': name, 'status': 'invalid', 'reason': 'missing or nonpositive timing/duration'})
                continue
            result = {"report": name, "backend": item.get("backend", item.get("model")),
                      "device": item.get("device", item.get("torch_device", "cpu")),
                      "seconds": seconds, "audio_seconds": item["audio_seconds"],
                      "rtf": seconds / item["audio_seconds"],
                      "realtime_factor": item["audio_seconds"] / seconds}
            runs = item.get('runs') or []
            if item.get("backend") == "whisper.cpp":
                timings = [run.get('native_timings_seconds', {}) for run in runs]
                valid = [t for t in timings if isinstance(t.get('total_time'), (int,float)) and isinstance(t.get('load_time'),(int,float)) and t['total_time'] > t['load_time']]
                if len(valid) == len(runs) and valid:
                    processing = statistics.median(t['total_time'] - t['load_time'] for t in valid)
                    result['processing_seconds'] = processing
                    result['processing_realtime_factor'] = duration / processing
                    result['load_seconds'] = statistics.median(t['load_time'] for t in valid)
                else:
                    result['native_timing_status'] = 'unavailable'
            text = runs[-1].get('text') if runs else item.get('text', item.get('final', {}).get('text'))
            if text is not None:
                result["wer"] = wer(reference, words(text))
                result["hypothesis_words"] = len(words(text))
                result["reference_words"] = len(reference)
            if "load_seconds" in item:
                result["load_seconds"] = item["load_seconds"]
            results.append(result)
    output = {"fixture": fixture, "wer_normalization": "lowercase, remove apostrophes, tokenize ASCII alphanumeric; last run text",
              "results": results}
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / "long-comparison.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
