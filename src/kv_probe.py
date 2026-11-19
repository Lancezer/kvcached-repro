"""Read atomic per-rank observations and verify current lifecycle ownership."""
import argparse
import json
import time
from pathlib import Path
from batch.ownership import identity, matches
from batch.probe import mps_clients, serving_metadata, with_runtime_programs


def validate_freshness(path, value, checked_epoch):
    observed_epoch = value.get('observed_epoch')
    age = checked_epoch - observed_epoch if isinstance(observed_epoch, (int, float)) else None
    if value.get('error'):
        reason = f"exporter reported error {value['error']!r}"
    elif age is not None and 0 <= age <= 5:
        return
    elif age is not None and age < 0:
        reason = 'negative age'
    elif age is not None and age > 5:
        reason = 'stale observation (age exceeds 5 s)'
    else:
        reason = 'invalid observation timestamp'
    # The batch runner retains the probe traceback in the failed sample's stderr.
    raise ValueError(f"KV rank {reason}: pid={value['pid']} source={str(path)!r} "
                     f"observed_epoch={observed_epoch!r} checked_epoch={checked_epoch!r} "
                     f"age_seconds={age!r}")


def live_row(path, sessions, models):
    value = json.loads(path.read_text())
    checked_epoch = time.time()
    process = identity(value['pid'])
    if process is None or process['session'] not in sessions:
        return None
    validate_freshness(path, value, checked_epoch)
    record = value.get('process_identity')
    if not record or record['pid'] != value['pid'] or not matches(record):
        raise ValueError('Export PID identity changed')
    model = models[value['checkpoint']]
    return dict(value, model=model['id'], source=str(path))


def rows(config, directory, resources):
    if not resources['records'] or not all(matches(item) for item in resources['records']):
        raise ValueError('Lifecycle process identity is no longer current')
    sessions = {item['session'] for item in resources['records']}
    models = {item['checkpoint']: item for item in config['models']}
    result = []
    for path in Path(directory).glob('*.json'):
        row = live_row(path, sessions, models)
        if row is not None:
            result.append(row)
    expected = {(m['id'], rank) for m in config['models'] for rank in range(m['tp'])}
    if len(result) != len(expected) or {(r['model'], r['rank']) for r in result} != expected:
        raise ValueError('Incomplete or duplicate live rank observations')
    return result


def observe(rules, kind):
    import os
    rules = with_runtime_programs(rules)
    root = Path(os.environ['KVREPRO_RESOURCE_DIRECTORY'])
    resources = json.loads((root / 'resources.json').read_text())
    config = json.loads(Path(rules['config']).read_text())
    ranks = rows(config, rules['directory'], resources)
    clients = mps_clients(resources, rules)
    for rank in ranks:
        servers = [server for server, pids in clients.items() if rank['pid'] in pids]
        if len(servers) != int(bool(resources['server_pids'])):
            raise ValueError('Actual private MPS client membership differs')
        rank['mps_server_pid'] = servers[0] if servers else None
    value = dict(ranks=ranks, basis='process-local allocator snapshots; physical includes reserved mapped pages')
    if kind == 'rank':
        value.update(mps_clients=clients, serving_metadata=serving_metadata(ranks, rules))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rules', required=True)
    parser.add_argument('--kind', choices=('rank', 'kv'), required=True)
    args = parser.parse_args()
    print(json.dumps(observe(json.loads(Path(args.rules).read_text()), args.kind), allow_nan=False))
