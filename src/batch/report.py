"""离线研究派生；在已准备客户端环境运行，复用原生读取和token区间。"""
from collections import Counter, defaultdict
import math
from pathlib import Path
from .audit import outcome, paired, weighted_quantile
from .storage import read, write


def load_native(case_directory, attempt):
    from xbench.harness.serving.measure import BenchCaseManifest, load_measurement, RequestRecord, StreamEvent
    from xbench.harness.serving.workload import read_jsonl
    manifest = BenchCaseManifest.model_validate_json((case_directory / 'case.json').read_bytes())
    workload = manifest.load_workload(case_directory)
    seal, summary = load_measurement(attempt, workload)
    requests = read_jsonl(attempt / 'requests.jsonl', RequestRecord)
    events = read_jsonl(attempt / 'events.jsonl', StreamEvent)
    return workload, seal, summary, requests, events


def quantiles(values):
    return dict(min=min(values, default=None), max=max(values, default=None), count=len(values),
                **{f'p{q}': weighted_quantile([(v, 1) for v in values], q / 100) for q in (50, 95, 99)})


def delay_values(request):
    times = dict(A=request.arrival_seconds, E=request.enqueued_at_seconds,
                 H=request.http_started_at_seconds, F=request.first_token_at_seconds, D=request.ended_at_seconds)
    return {end + '-' + start: times[end] - times[start]
            for end, start in (('F', 'A'), ('F', 'E'), ('F', 'H'), ('D', 'A'), ('D', 'E'),
                               ('D', 'H'), ('E', 'A'), ('H', 'E'), ('H', 'A'))
            if times[end] is not None and times[start] is not None}


def intervals(request, events):
    from xbench.harness.serving.measure import token_intervals
    values = token_intervals(events)
    count = request.completion_tokens
    # 06请求等权mean ITL按正输出区间等权；上游mean_itl按token权重，单独保留差异。
    mean = sum(v.value_seconds for v in values) / len(values) if values and count and count >= 2 else None
    return values, mean


def model_delays(requests, grouped):
    success, arrival = defaultdict(list), defaultdict(list)
    tokens, means, tpots = defaultdict(list), [], []
    for request in requests:
        values = delay_values(request)
        for name in ('E-A', 'H-E', 'H-A'):
            if name in values:
                arrival[name].append(values[name])
        if outcome(request.model_dump()) != 'success':
            continue
        for name, value in values.items():
            success[name].append(value)
        add_intervals(request, grouped[request.request_id], tokens, means, tpots)
    return dict(coverage=coverage(requests, tokens), success={k: quantiles(v) for k, v in success.items()},
                arrival={k: dict(values=quantiles(v), effective=len(v), missing=len(requests) - len(v)) for k, v in arrival.items()},
                token_itl={k: dict(weight=sum(w for _, w in v), p95=weighted_quantile(v, .95),
                                   p99=weighted_quantile(v, .99)) for k, v in tokens.items()},
                mean_itl=dict(values=quantiles(means), requests=len(means)),
                tpot=dict(values=quantiles(tpots), requests=len(tpots)))


def coverage(requests, tokens):
    successful = [r for r in requests if r.outcome == 'success']
    possible = sum(max(0, r.completion_tokens - 1) for r in successful if r.completion_tokens is not None)
    covered = sum(weight for _, weight in tokens['combined'])
    return dict(possible_interval_weight=possible, covered_weight=covered,
                fraction=covered / possible if possible else None,
                successful_requests=len(successful), sparse_request_p99=len(successful) < 100,
                completion_lengths=quantiles([r.completion_tokens for r in successful if r.completion_tokens is not None]),
                finish_reasons=dict(Counter(r.finish_reason for r in successful)))


def add_intervals(request, events, tokens, means, tpots):
    values, mean = intervals(request, events)
    for interval in values:
        tokens[interval.kind].append((interval.value_seconds, interval.weight))
        tokens['combined'].append((interval.value_seconds, interval.weight))
    if mean is not None:
        means.append(mean)
    if request.metrics.tpot_seconds is not None:
        tpots.append(request.metrics.tpot_seconds)


def throughput(requests, horizon, window, target, planned):
    if window is None or window <= 0:
        return dict(available=False)
    successful = {r.request_id: r for r in requests if r.outcome == 'success'
                  and r.ended_at_seconds is not None and r.ended_at_seconds <= window}
    counts = arrival_counts(requests, horizon, planned)
    return dict(available=True, window=window, arrival_horizon=horizon, drain=max(0, window - horizon),
                success_requests_per_second=len(successful) / window,
                input_tokens_per_second=target.input_tokens_per_second,
                positive_output_tokens_per_second=target.output_tokens_per_second,
                partial_output_tokens=target.partial_output_tokens,
                partial_output_tokens_per_second=target.partial_output_tokens / window,
                total_observed_output_per_second=(target.output_tokens + target.partial_output_tokens) / window,
                arrival_window=counts)


def arrival_counts(requests, horizon, planned):
    sent = [r for r in requests if r.http_started_at_seconds is not None and 0 <= r.http_started_at_seconds < horizon]
    counts = dict(planned=planned, sent=len(sent), success=sum(r.outcome == 'success' and r.ended_at_seconds is not None
                  and 0 <= r.ended_at_seconds < horizon for r in requests),
                  sent_terminal=sum(outcome(r.model_dump()) in ('success', 'failed', 'cancelled') and r.ended_at_seconds is not None for r in sent))
    return dict(counts=counts, seconds=horizon, rates={k: v / horizon if horizon else None for k, v in counts.items()},
                sent_fraction=len(sent) / planned if planned else None, sent_fraction_denominator=planned)


def outcome_summary(records, planned):
    counts = Counter(outcome(r.model_dump()) for r in records)
    counts['unknown'] += max(0, planned - len(records))
    fixed = {name: counts[name] for name in ('success', 'failed', 'cancelled', 'not_sent', 'unknown')}
    timeout = sum(outcome(r.model_dump()) == 'failed' and r.error_kind == 'timeout' for r in records)
    failures = dict(timeout=timeout, other=counts['failed'] - timeout)
    return dict(counts=fixed, denominator=planned,
                fractions={k: v / planned if planned else None for k, v in fixed.items()},
                failure_subclasses=failures,
                failure_fractions={k: v / planned if planned else None for k, v in failures.items()},
                failure_subclass_denominator=planned)


def partial_output(records, events, window):
    groups = {r.request_id: ('timeout' if r.error_kind == 'timeout' else 'cancelled' if r.outcome == 'cancelled' else 'other_failed')
              for r in records if outcome(r.model_dump()) in ('failed', 'cancelled')}
    totals = dict(timeout=0, other_failed=0, cancelled=0)
    previous = {}
    for event in events:
        if event.kind == 'data' and event.accepted and event.completion_tokens is not None:
            before = previous.get(event.request_id, 0)
            if event.request_id in groups and window is not None and event.observed_at_seconds <= window:
                totals[groups[event.request_id]] += max(0, event.completion_tokens - before)
            previous[event.request_id] = event.completion_tokens
    missing = sum(outcome(r.model_dump()) == 'unknown' for r in records)
    return dict(observed_tokens=totals if window is not None else None, evidence_missing_requests=missing,
                seconds=window, rates={k: v / window if window else None for k, v in totals.items()},
                basis='已知非成功请求的accepted累计输出增量；unknown不补零，原生总partial另存')


def arrival_buckets(workload, records, model, window):
    end = max(workload.arrival_horizon_seconds, window or 0)
    width = workload.bucket_seconds
    result = [dict(start=i * width, end=min((i + 1) * width, end), planned=0, sent=0)
              for i in range(math.ceil(end / width))]
    times = [('planned', r.arrival_seconds) for r in workload.requests if model == 'aggregate' or r.model_id == model]
    times += [('sent', r.http_started_at_seconds) for r in records]
    for field, when in times:
        if when is not None and 0 <= when < end:
            result[int(when / width)][field] += 1
    for bucket in result:
        span = bucket['end'] - bucket['start']
        bucket.update(planned_rate=bucket['planned'] / span, sent_rate=bucket['sent'] / span,
                      phase='arrival' if bucket['start'] < workload.arrival_horizon_seconds else 'drain')
    return result


def report_attempt(path):
    workload, seal, native, requests, events = load_native(path.parent, path)
    grouped = defaultdict(list)
    for event in events:
        grouped[event.request_id].append(event)
    models = {}
    for model in (*workload.model_ids, 'aggregate'):
        records = [r for r in requests if model == 'aggregate' or r.model_id == model]
        ids = {r.request_id for r in records}
        selected_events = [e for e in events if e.request_id in ids]
        planned = sum(model == 'aggregate' or r.model_id == model for r in workload.requests)
        models[str(model)] = dict(outcomes=outcome_summary(records, planned),
              partial_output=partial_output(records, selected_events, seal.window_end_seconds),
              arrival_buckets=arrival_buckets(workload, records, model, seal.window_end_seconds),
              at_arrival_horizon=at_horizon(records, workload.arrival_horizon_seconds),
              delays=model_delays(records, grouped), throughput=throughput(records,
              workload.arrival_horizon_seconds, seal.window_end_seconds, native.targets[str(model)], planned))
    return dict(attempt=str(path), native=native.model_dump(mode='json'), models=models,
                input_usage=pool_usage(workload, requests), window_kind=seal.window_kind)


def at_horizon(records, horizon):
    queued, inflight, unknown = 0, 0, 0
    for row in records:
        enqueue, sent, ended = row.enqueued_at_seconds, row.http_started_at_seconds, row.ended_at_seconds
        queued += enqueue is not None and enqueue < horizon and sent is not None and sent >= horizon
        inflight += sent is not None and sent < horizon and ended is not None and ended >= horizon
        unknown += (row.error_kind == 'evidence_missing' or (sent is not None and sent < horizon and ended is None))
    return dict(known_queued=queued, known_inflight=inflight, state_unknown=unknown)


def pool_usage(workload, requests):
    result = {}
    for model in workload.model_ids:
        measured = [r.prompt_id for r in workload.requests if r.model_id == model]
        warmup = {r.prompt_id for r in workload.warmup if r.model_id == model}
        result[str(model)] = dict(planned=len(measured), repeated_fraction=1 - len(set(measured)) / len(measured) if measured else None,
              warmup_overlap=len(set(measured) & warmup), actual_prompt_lengths=quantiles([r.prompt_tokens for r in requests
              if r.model_id == model and r.prompt_tokens is not None]))
    return result


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + '.' + key if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif type(item) in (int, float):
            result[name] = item
        elif item is None:
            result[name] = None
    return result


def comparison_metrics(report):
    # 原生CDF/吞吐桶保留在逐case报告；跨seed只投影同口径标量，避免按桶序号错配时间。
    models = {name: {k: v for k, v in model.items() if k != 'arrival_buckets'} for name, model in report['models'].items()}
    kv = {rank: {field: stats for field, stats in values.items() if field in ('logical_bytes', 'physical_bytes')}
          for rank, values in report.get('kv', {}).get('ranks', {}).items()}
    return flatten(dict(models=models, kv=kv,
                        native_targets=report['native'].get('targets', {}), ideal_kv=report.get('ideal_kv', {})))


def compare(manifest, reports):
    pairs = defaultdict(dict)
    for row in manifest['cases']:
        if row['case_id'] in reports:
            pairs[row['pair_id']][row['mps']] = comparison_metrics(reports[row['case_id']])
    result = {}
    for pair_id, states in pairs.items():
        if set(states) != {False, True}:
            result[pair_id] = dict(available=False)
            continue
        result[pair_id] = {key: paired(states[True][key], states[False][key])
                           if states[True].get(key) is not None and states[False].get(key) is not None
                           else dict(difference=None, ratio=None, missing=True)
                           for key in states[True].keys() | states[False].keys()}
    return result


def report(output, destination):
    output, destination = Path(output), Path(destination)
    destination.mkdir(exist_ok=False)
    index, manifest = read(output / 'index.json'), read(output / 'manifest.json')
    config = read(output / 'configuration.json')['values']
    reports = {}
    for case_id, entry in index['cases'].items():
        if entry.get('adopted'):
            path = Path(entry['adopted'])
            reports[case_id] = report_attempt(path)
            external = next(a['external'] for a in entry['attempts'] if a['path'] == str(path))
            reports[case_id]['kv'] = kv_summary(Path(external), path, config['sample_seconds'] + config['probe_seconds'])
            reports[case_id]['ideal_kv'] = ideal_kv(path, config)
    pairs = compare(manifest, reports)
    write(destination / 'research.json', dict(cases=reports, pairs=pairs, cross_pair=cross_pair(manifest, pairs),
                                              cross_seed=cross_seed(manifest, reports)))


def cross_seed(manifest, reports):
    grouped = defaultdict(list)
    for row in manifest['cases']:
        key = (row['conditions']['rate'], row['mps'])
        grouped[key].append(comparison_metrics(reports[row['case_id']]) if row['case_id'] in reports else {})
    result = {}
    for (rate, mps), rows in grouped.items():
        keys = set().union(*(row.keys() for row in rows))
        result[f'{rate}:{mps}'] = dict(expected_seeds=len(rows), metrics={key: seed_range([row.get(key) for row in rows]) for key in keys})
    return result


def seed_range(values):
    valid = [v for v in values if v is not None]
    return dict(median=weighted_quantile([(v, 1) for v in valid], .5), min=min(valid, default=None),
                max=max(valid, default=None), seeds=len(valid), missing=len(values) - len(valid))


def cross_pair(manifest, pairs):
    grouped, seen = defaultdict(list), set()
    for row in manifest['cases']:
        pair_id = row['pair_id']
        if pair_id not in seen:
            value = pairs.get(pair_id, {})
            grouped[row['conditions']['rate']].append(flatten(value) if value.get('available') is not False else {})
            seen.add(pair_id)
    return {str(rate): dict(expected_seeds=len(rows), metrics={key: seed_range([row.get(key) for row in rows])
            for key in set().union(*(row.keys() for row in rows))}) for rate, rows in grouped.items()}


def ideal_kv(attempt, config):
    from .audit import jsonl, kv_bytes
    records = jsonl(attempt / 'requests.jsonl')
    result = {}
    for model in config['models']:
        rows = [r for r in records if r['model_id'] == model['id'] and outcome(r) == 'success']
        lengths = [r['prompt_tokens'] + r['completion_tokens'] for r in rows
                   if r.get('prompt_tokens') is not None and r.get('completion_tokens') is not None]
        result[model['id']] = dict(successful_requests=len(rows), effective_requests=len(lengths),
             missing_requests=len(rows) - len(lengths), per_rank={rank: quantiles([kv_bytes(n, geometry) for n in lengths])
             for rank, geometry in model['geometry_by_rank'].items()}, basis='成功请求服务端最终P+O理想KV；非瞬时驻留/物理占用，不乘DP')
    return result


def kv_summary(lifecycle, attempt, max_gap):
    from .audit import jsonl
    origin = read(attempt / 'measurement.json')['wall_clock_origin_seconds']
    window = read(attempt / 'repetition.json')['window_end_seconds']
    groups, errors, breaks = defaultdict(list), [], []
    for sample in jsonl(lifecycle / 'kv.jsonl'):
        if sample.get('error'):
            errors.append(dict(epoch=sample['epoch'], error=sample['error']))
            breaks.append(sample['epoch'] - origin)
        for row in sample.get('derived') or []:
            when = row.get('observed_epoch')
            if when is not None and 0 <= when - origin <= window:
                groups[(row['model'], row['rank'], row['gpu'])].append((when - origin, row))
    return dict(errors=errors, ranks={f'{m}:{rank}:{gpu}': occupancy(values, window, max_gap, breaks)
                                     for (m, rank, gpu), values in groups.items()})


def occupancy(values, window, max_gap, breaks=()):
    unique = {when: row for when, row in values}
    ordered = sorted(unique.items())
    result = dict(window=window, samples=len(ordered), observation_times=[t for t, _ in ordered])
    for field in ('logical_bytes', 'physical_bytes'):
        valid = [(when, row[field]) for when, row in ordered if row.get(field) is not None]
        intervals = [(b[0] - a[0], (a[1].get(field) + b[1].get(field)) / 2)
                     for a, b in zip(ordered, ordered[1:]) if a[1].get(field) is not None and b[1].get(field) is not None
                     and not any(a[0] <= t <= b[0] for t in breaks)]
        usable = [(dt, value) for dt, value in intervals if 0 < dt <= max_gap]
        duration = sum(dt for dt, _ in usable)
        result[field] = dict(peak=max((v for _, v in valid), default=None),
                average_over_observed_span=sum(dt * value for dt, value in usable) / duration if duration else None,
                observed_span=duration, missing_window=max(0, window - duration),
                gaps=[dt for dt, _ in intervals if dt > max_gap], basis='有效相邻观察梯形积分；不外推窗边界或跨缺口积分')
    return result
