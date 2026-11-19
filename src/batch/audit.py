"""研究层只派生事实，不将原生退出码映射成有效性。"""
import json
import math
from collections import Counter
from pathlib import Path
from .config import file_hash
from .storage import read


def jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def outcome(record):
    return 'unknown' if record.get('error_kind') == 'evidence_missing' else record['outcome']


def validity(attempt, cleanup):
    manifest = read(attempt / 'repetition.json')
    required = ('events.jsonl', 'requests.jsonl')
    if manifest.get('window_end_seconds') is not None:
        required += ('measurement.json',)
    for name in required:
        if not (attempt / name).is_file() or manifest.get('artifact_sha256', {}).get(name) != file_hash(attempt / name):
            return dict(valid=None, reason='原生证据缺失或摘要不符')
    requests = jsonl(attempt / 'requests.jsonl')
    counts = dict(Counter(outcome(record) for record in requests))
    if counts.get('unknown'):
        invalid = manifest.get('finished') is True and manifest.get('infrastructure_error') and manifest.get('window_kind') != 'complete'
        return dict(valid=False if invalid else None, reason='请求终态证据缺失；保留实际中断事实', outcomes=counts)
    valid = (manifest.get('finished') is True and manifest.get('raw_evidence_complete') is True
             and manifest.get('cleanup_verified') is True and cleanup
             and manifest.get('window_kind') == 'complete' and not manifest.get('infrastructure_error'))
    return dict(valid=valid, reason='完整封窗与证据' if valid else '中断或基础设施证据不完整', outcomes=counts)


def adopt(original, rerun=None):
    if original['research']['valid'] is True:
        return original['path']
    if rerun and rerun['research']['valid'] is True:
        return rerun['path']
    return None


def external_validity(result, native_result, config, attempt):
    if native_result['valid'] is not True:
        return native_result
    path = Path(result['lifecycle']) / 'kv.jsonl'
    if not path.is_file():
        return dict(native_result, valid=None, reason='KV附加证据缺失')
    start = read(attempt / 'measurement.json')['wall_clock_origin_seconds']
    window = read(attempt / 'repetition.json')['window_end_seconds']
    samples = [s for s in jsonl(path) if start <= s['epoch'] <= start + window]
    tolerance = config['sample_seconds'] + config['probe_seconds'] + config['poll_seconds']
    if not covered_samples(samples, start, window, tolerance):
        return dict(native_result, valid=None, reason='KV测量窗覆盖不足或存在采样间隙')
    reason = sample_issue(samples, config)
    return dict(native_result, valid=None, reason=reason) if reason else native_result


def sample_issue(samples, config):
    expected = {(m['id'], rank) for m in config['models'] for rank in range(m['tp'])}
    for sample in samples:
        rows = sample.get('derived') or []
        if sample.get('error') or len(rows) != len(expected) or {(r['model'], r['rank']) for r in rows} != expected:
            return 'KV采样缺失或rank覆盖不全'
        for row in rows:
            observed = row.get('observed_epoch')
            if observed is None or abs(sample['epoch'] - observed) > config['sample_seconds'] + config['probe_seconds']:
                return 'KV观测时刻未知或陈旧，不能用读取时间冒充采样'
            if row.get('logical_bytes') is None or row.get('physical_bytes') is None:
                return '逻辑或物理KV缺失，不能以块数或读取时间补造'
    return None


def covered_samples(samples, start, window, tolerance):
    if not samples:
        return False
    times = [s['epoch'] for s in samples]
    return (times[0] - start <= tolerance and start + window - times[-1] <= tolerance
            and all(0 < right - left <= tolerance for left, right in zip(times, times[1:])))


def weighted_quantile(values, quantile):
    values = sorted((float(v), int(w)) for v, w in values if w > 0)
    total = sum(w for _, w in values)
    if not total:
        return None
    position = (total - 1) * quantile
    indices = (math.floor(position), math.ceil(position))
    selected, cursor = [], 0
    for value, weight in values:
        for index in indices[len(selected):]:
            if cursor <= index < cursor + weight:
                selected.append(value)
        cursor += weight
    return selected[0] + (selected[-1] - selected[0]) * (position % 1)


def kv_bytes(tokens, geometry):
    if type(tokens) is not int or tokens < 0:
        raise ValueError('active_tokens必须为非负整数，不能以块数代替')
    width = geometry_width(geometry)
    return tokens * geometry['layers'] * width * geometry['element_bytes']


def geometry_width(geometry):
    keys = ('layers', 'element_bytes')
    keys += ('kv_lora_rank', 'qk_rope_head_dim') if geometry['kind'] == 'mla' else ('local_kv_heads', 'head_dim')
    if any(type(geometry.get(key)) is not int or geometry[key] <= 0 for key in keys):
        raise ValueError('KV几何必须为实际rank的正整数维度')
    if geometry['kind'] == 'mla':
        width = geometry['kv_lora_rank'] + geometry['qk_rope_head_dim']
    elif geometry['kind'] == 'mha':
        width = 2 * geometry['local_kv_heads'] * geometry['head_dim']
    else:
        raise ValueError('未知KV结构；不能猜测TP/DP换算')
    if width <= 0:
        raise ValueError('几何必须由实际checkpoint/rank提供')
    return width


def paired(on, off):
    return dict(difference=on - off, ratio=on / off if off else None)
