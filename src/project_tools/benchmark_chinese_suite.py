"""Run independent Chinese panels serially, retaining native benchmark evidence."""
import json
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mitigations', action='store_true', help='Also check float16 decoding alternatives')
    args = parser.parse_args()
    panels = json.loads((ROOT / '.cache/fixtures/chinese/manifest.json').read_text(encoding='utf-8'))['panels']
    output = ROOT / '.cache/reports/chinese'
    output.mkdir(parents=True, exist_ok=True)
    python = ROOT / '.venv/Scripts/python.exe'
    qwen_python = ROOT / '.cache/experiments/qwen-env/Scripts/python.exe'
    model = 'D:/Tools/fasterwhisperGUI-0.8.5.sfx/faster-whisper-large-v3-turbo-ct2'
    env = os.environ.copy()
    env['PATH'] = r'C:\Program Files\AMD\ROCm\7.1\bin' + os.pathsep + env['PATH']
    env['ROCBLAS_USE_HIPBLASLT'] = '1'
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    for panel in panels:
        jobs = []
        for device in ['cpu', 'vulkan', 'rocm']:
            build = 'rocm' if device == 'rocm' else 'vulkan'
            jobs.append((f'cpp-{device}', python, 'benchmark_whisper_cpp.py', [
                '--executable', str(ROOT / f'.cache/experiments/whisper.cpp/build-{build}/bin/whisper-cli.exe'),
                '--model', str(ROOT / '.cache/models/ggml-large-v3-turbo.bin'), '--devices', device]))
        for device, precision in [('cpu', 'int8'), ('cuda', 'float16'), ('cuda', 'int8_float16')]:
            jobs.append((f'fw-{device}-{precision}', python, 'benchmark_faster_whisper.py', [
                '--model', model, '--device', device, '--compute-type', precision]))
        jobs.append(('qwen-cuda', qwen_python, 'benchmark_qwen.py', ['--device', 'cuda']))
        jobs.append(('sherpa-zh', ROOT / '.cache/experiments/streaming-env/Scripts/python.exe',
                     'benchmark_streaming.py', ['--model-dir', str(ROOT / '.cache/models/streaming/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23')]))
        if args.mitigations:
            jobs.append(('cpp-rocm-beam5', python, 'benchmark_whisper_cpp.py', [
                '--executable', str(ROOT / '.cache/experiments/whisper.cpp/build-rocm/bin/whisper-cli.exe'),
                '--model', str(ROOT / '.cache/models/ggml-large-v3-turbo.bin'), '--devices', 'rocm', '--beam-size', '5']))
            for name, flags in [('fw-cuda-float16-no-context', ['--no-condition-on-previous-text']),
                                ('fw-cuda-float16-beam5', ['--beam-size', '5']),
                                ('fw-cuda-float16-beam5-no-context', ['--beam-size', '5', '--best-of', '5', '--no-condition-on-previous-text'])]:
                jobs.append((name, python, 'benchmark_faster_whisper.py', [
                    '--model', model, '--device', 'cuda', '--compute-type', 'float16', *flags]))
            jobs.append(('fw-cuda-int8-beam5-no-context', python, 'benchmark_faster_whisper.py', [
                '--model', model, '--device', 'cuda', '--compute-type', 'int8', '--beam-size', '5',
                '--best-of', '5', '--no-condition-on-previous-text']))
        for name, interpreter, script, options in jobs:
            report = output / f'{panel["id"]}-{name}.json'
            if report.exists():
                print(f'Cached {report.name}', flush=True)
                continue
            recognition_options = [] if script == 'benchmark_streaming.py' else ['--language', 'zh', '--runs', '1']
            command = [str(interpreter), str(ROOT / 'src/project_tools' / script),
                       '--audio', panel['audio'], *recognition_options, '--report', str(report), *options]
            print(f'Running {panel["id"]} {name}', flush=True)
            with report.with_suffix('.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'Finished {report.name}', flush=True)


if __name__ == '__main__':
    main()
