"""Character edit counts with explicit script normalization, retaining transcripts."""
import json
import re
from pathlib import Path
import unicodedata
from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = OpenCC('t2s')


def normalize(text, simplify=True):
    text = unicodedata.normalize('NFKC', text)
    if simplify:
        text = CONVERTER.convert(text)
    return ''.join(c.lower() for c in text if not c.isspace() and
                   not unicodedata.category(c).startswith('P'))


def spoken_numbers(text):
    """Secondary diagnostic only: corpus-relevant phones, years, percentages and integers."""
    digits = '零一二三四五六七八九'
    def integer(value):
        n = int(value)
        if n < 10:
            return digits[n]
        if n >= 10000:
            return ''.join(digits[int(c)] for c in value)
        result, pending_zero = '', False
        for divisor, unit in [(1000, '千'), (100, '百'), (10, '十'), (1, '')]:
            digit, n = divmod(n, divisor)
            if digit:
                if pending_zero:
                    result += '零'
                result += ('' if divisor == 10 and digit == 1 and not result else digits[digit]) + unit
                pending_zero = False
            elif result and n:
                pending_zero = True
        return result
    def decimal(value):
        parts = value.split('.')
        return integer(parts[0]) + ('点' + ''.join(digits[int(c)] for c in parts[1]) if len(parts) == 2 else '')
    text = unicodedata.normalize('NFKC', text).replace('幺', '一')
    text = re.sub(r'(?<!\d)\d{11}(?!\d)', lambda m: ''.join(digits[int(c)] for c in m[0]), text)
    text = re.sub(r'(?<!\d)\d{4}(?=年)', lambda m: ''.join(digits[int(c)] for c in m[0]), text)
    text = re.sub(r'\d+(?:\.\d+)?%', lambda m: '百分之' + decimal(m[0][:-1]), text)
    text = re.sub(r'\d+(?:\.\d+)?', lambda m: decimal(m[0]), text)
    return normalize(text)


def errors(reference, hypothesis):
    # Store the shortest edit path counts. Ties prefer substitution, then deletion.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
            else:
                cost, s, d, ins = previous[j - 1]
                substitution = (cost + 1, s + 1, d, ins)
                cost, s, d, ins = previous[j]
                deletion = (cost + 1, s, d + 1, ins)
                cost, s, d, ins = current[j - 1]
                insertion = (cost + 1, s, d, ins + 1)
                current.append(min([substitution, deletion, insertion], key=lambda x: x[0]))
        previous = current
    count, s, d, ins = previous[-1]
    return {'reference_chars': len(reference), 'substitutions': s, 'deletions': d,
            'insertions': ins, 'errors': count, 'cer': count / len(reference) if reference else None}


def self_check():
    assert errors('中国', '中国')['errors'] == 0
    assert errors('中国', '中果')['substitutions'] == 1
    assert errors('中国', '中')['deletions'] == 1
    assert errors('中国', '中国人')['insertions'] == 1
    assert errors('中国', '')['deletions'] == 2
    assert normalize('國，Ａ！') == '国a'
    assert normalize('國，Ａ！', False) == '國a'
    assert spoken_numbers('2015年1月底，22.6%，23，13504307035') == spoken_numbers('二零一五年一月底百分之二十二点六二十三幺三五零四三零七零三五')
    assert spoken_numbers('101和1010和1001') == '一百零一和一千零一十和一千零一'


def main():
    self_check()
    manifest = json.loads((ROOT / '.cache/fixtures/chinese/manifest.json').read_text(encoding='utf-8'))
    grouped = {}
    invalid_reports = []
    for panel in manifest['panels']:
        for path in sorted((ROOT / '.cache/reports/chinese').glob(f'{panel["id"]}-*.json')):
            raw = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(raw, list) and not raw:
                invalid_reports.append({"report": str(path), "reason": "empty report"})
                continue
            report = raw[0] if isinstance(raw, list) else raw
            if not isinstance(report, dict) or not report.get("backend") or (report["backend"] != "sherpa-onnx" and not report.get("runs")):
                invalid_reports.append({"report": str(path), "reason": "missing backend or runs"})
                continue
            if report['backend'] == 'sherpa-onnx':
                run = {'text': report['final']['text'], 'elapsed_seconds': report['elapsed_seconds']}
            else:
                run = report['runs'][-1]
            name = path.stem.removeprefix(panel['id'] + '-')
            ref, hyp = normalize(panel['reference']), normalize(run['text'])
            timing = run.get('native_timings_seconds', {})
            seconds = timing['total_time'] - timing.get('load_time', 0) if timing.get('total_time') is not None else run['elapsed_seconds']
            if seconds < 0:
                invalid_reports.append({'report': str(path), 'reason': 'negative processing time'})
                continue
            grouped.setdefault(name, []).append({
                'panel': panel['id'], 'source_report': str(path), 'audio_seconds': panel['audio_seconds'],
                'processing_seconds': seconds, 'external_seconds': run['elapsed_seconds'],
                'load_seconds': timing.get('load_time', report.get('load_seconds')),
                'reference': panel['reference'], 'hypothesis': run['text'],
                'normalized_reference': ref, 'normalized_hypothesis': hyp,
                'score': errors(ref, hyp),
                'number_format_score': errors(spoken_numbers(panel['reference']), spoken_numbers(run['text'])),
                'strict_script_score': errors(normalize(panel['reference'], False), normalize(run['text'], False))})
    results = []
    for name, panels in grouped.items():
        combined = {key: sum(p['score'][key] for p in panels) for key in
                    ['reference_chars', 'substitutions', 'deletions', 'insertions', 'errors']}
        combined['cer'] = combined['errors'] / combined['reference_chars'] if combined['reference_chars'] else None
        strict_chars = sum(p['strict_script_score']['reference_chars'] for p in panels)
        combined['strict_script_cer'] = sum(p['strict_script_score']['errors'] for p in panels) / strict_chars if strict_chars else None
        number_chars = sum(p['number_format_score']['reference_chars'] for p in panels)
        combined['number_format_cer'] = sum(p['number_format_score']['errors'] for p in panels) / number_chars if number_chars else None
        combined['number_format_counts'] = {key: sum(p['number_format_score'][key] for p in panels)
            for key in ['reference_chars', 'errors', 'substitutions', 'deletions', 'insertions']}
        seconds = sum(p['processing_seconds'] for p in panels)
        duration = sum(p['audio_seconds'] for p in panels)
        results.append({'backend_config': name, 'panels_count': len(panels), 'score': combined,
                        'processing_seconds': seconds, 'mean_processing_seconds': seconds / len(panels),
                        'audio_seconds': duration, 'rtf': seconds / duration if duration > 0 else None, 'panels': panels})
    output = ROOT / '.cache/reports/chinese-comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'dataset_manifest': str(ROOT / '.cache/fixtures/chinese/manifest.json'),
                                  'dataset_revision': manifest['revision'],
                                  'utterances': sum(len(p['clips']) for p in manifest['panels']),
                                  'normalization': 'NFKC, lowercase, remove punctuation/whitespace, OpenCC t2s; no number verbalization',
                                  'number_format_normalization': 'Secondary: 11-digit phones and 4-digit years spelled by digit; integers below 10000 in Chinese units; decimals and percentages expanded; yao/one unified',
                                  'results': results, 'invalid_reports': invalid_reports}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in results:
        print(row['backend_config'], row['panels_count'], f"CER={row['score']['cer']}",
              f"strict={row['score']['strict_script_cer']}", f"mean={row['mean_processing_seconds']:.3f}s", row['score'])


if __name__ == '__main__':
    main()
