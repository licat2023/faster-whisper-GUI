"""可选 sherpa-onnx 后端；每次识别使用独立的有状态会话。"""

from pathlib import Path
from .streaming import StreamUpdate


class SherpaStreamingBackend:
    sample_rate = 16000

    def __init__(self, recognizer):
        self.recognizer = recognizer

    @classmethod
    def from_directory(cls, directory, threads=4):
        import sherpa_onnx
        root = Path(directory)
        def model(name):
            paths = sorted(root.glob(f"{name}*.int8.onnx")) or sorted(root.glob(f"{name}*.onnx"))
            if len(paths) != 1:
                raise ValueError(f"模型目录应包含一个 {name} 模型：{root}")
            return str(paths[0])
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(root / "tokens.txt"), encoder=model("encoder"),
            decoder=model("decoder"), joiner=model("joiner"),
            num_threads=threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", enable_endpoint_detection=False,
        )
        return cls(recognizer)

    def start(self):
        return _Session(self.recognizer, self.sample_rate)


class _Session:
    def __init__(self, recognizer, sample_rate):
        self.recognizer = recognizer
        self.sample_rate = sample_rate
        self.stream = recognizer.create_stream()
        self.samples = 0
        self.revision = 0
        self.text = ""
        self.closed = False

    def _update(self, final=False):
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream)
        if text != self.text:
            self.text = text
            self.revision += 1
        return StreamUpdate(self.text, self.revision, final, self.samples / self.sample_rate)

    def accept(self, samples):
        import numpy as np
        if self.closed:
            raise RuntimeError("流式会话已结束")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("流式输入必须为有限的单声道 float32 样本")
        self.samples += samples.size
        self.stream.accept_waveform(self.sample_rate, samples)
        return self._update()

    def finish(self):
        import numpy as np
        if self.closed:
            raise RuntimeError("流式会话已结束")
        self.stream.accept_waveform(self.sample_rate, np.zeros(int(0.66 * self.sample_rate), dtype=np.float32))
        self.stream.input_finished()
        self.closed = True
        return self._update(final=True)
