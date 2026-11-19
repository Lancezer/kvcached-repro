"""复用已冻结批次及其原生证据，预览或登记独立 attempt。"""
from contextlib import ExitStack
from pathlib import Path

from localpaths import allowed
from . import native, prepared, runner
from .ownership import SharedFault
from .resources import validate_ranks
from .storage import lock, read, write


def case(manifest, case_id):
    for row in manifest['cases']:
        if row['case_id'] == case_id:
            return row
    raise SharedFault('指定 case 不属于冻结批次: ' + case_id)


def workload(directory, row):
    seal = read(directory / 'kvrepro-frozen.json')
    contract = dict(pair_id=row['pair_id'], conditions=row['conditions'])
    if seal['identity']['contract'] != contract or seal['files'] != prepared.validate_files(directory):
        raise SharedFault('prepared 合同或文件摘要不符')
    return seal


def original_attempt(source, row, path, config):
    run = path.parents[2]
    directory = run.parents[2]
    expected = source / 'cases' / row['case_id']
    if directory not in (expected, expected / 'research-rerun') or native.locate(directory, row['case_id']) != run:
        raise SharedFault('attempt 不属于指定来源 case 的原生 run')
    attempt = next((a for a in native.attempts(run, row, config) if Path(a['path']) == path), None)
    if attempt is None:
        raise SharedFault('来源索引之外的 attempt 或已被原生替换')
    execution = runner.attempt_execution(directory, run, row, path)
    entry = read(source / 'index.json')['cases'].get(row['case_id'], {})
    indexed = any(a['path'] == str(path) and a['external'] == execution['lifecycle']
                  for a in entry.get('attempts', []))
    if not indexed:
        raise SharedFault('来源索引缺少与本次 lifecycle 一致的 attempt')
    verdict = runner.research_verdict(config, row, path, execution, persist=False)
    if verdict['valid'] is not True:
        raise SharedFault('来源 attempt 无法通过有效性核验: ' + verdict['reason'])
    return dict(attempt, research=verdict, external=execution['lifecycle'], source_run=str(run))


def check_allocation(attempt, frozen, row, expected):
    lifecycle = allowed(attempt['external'])
    observation = read(lifecycle / 'hardware.json')
    runner.verify_hardware(frozen['hardware_observation'], observation)
    observed = read(lifecycle / 'metadata-ranks.json')
    resources = read(lifecycle / 'resources.json')
    config = dict(frozen['values'], logical_devices=observation['mapping'])
    validate_ranks(config, row, observed, resources['server_pids'])
    if runner.metadata_identity(observed['serving_metadata'], observation) != expected:
        raise SharedFault('来源实际 rank 放置、软件或设备条件不同')
    return runner.rank_capacities(config, observed)


def check_conditions(target, source, target_row, source_row):
    if target_row['conditions'] != source_row['conditions'] or target_row['mps'] != source_row['mps']:
        raise SharedFault('case 工作负载、模型、协议或 MPS 状态不同')
    for name in ('code', 'patch'):
        if target[name] != source[name]:
            raise SharedFault('冻结软件证据不同: ' + name)
    names = ('process_environment', 'hardware', 'startup_seconds', 'poll_seconds',
             'cleanup_seconds', 'sample_seconds', 'probe_seconds')
    for name in names:
        if target['values'][name] != source['values'][name]:
            raise SharedFault('冻结运行条件不同: ' + name)
    if target['probe_protocol'] != source['probe_protocol']:
        raise SharedFault('冻结探针协议不同')
    runner.verify_hardware(target['hardware_observation'], source['hardware_observation'])


def compare(index, output, source, path, row, source_row, target, origin, config):
    check_conditions(target, origin, row, source_row)
    pair = index['pairs'].get(row['pair_id'])
    if not pair or not allowed(pair).is_relative_to(output / 'cases'):
        raise SharedFault('目标缺少本批次内可核验的 prepared 配对来源')
    pair = allowed(pair)
    other = workload(path.parent, source_row)
    reference = workload(pair, row)
    if reference['files'] != other['files'] or reference['content'] != other['content']:
        raise SharedFault('来源与目标配对 workload 不一致')
    actual = original_attempt(source, source_row, path, config)
    expected = runner.prepared_identity(pair)
    if runner.prepared_identity(path.parent) != expected:
        raise SharedFault('来源原生 prepared 的实际放置或软件条件不同')
    capacity = check_allocation(actual, origin, source_row, expected)
    runner.verify_capacity(source, source_row, capacity)
    runner.verify_capacity(output, row, capacity)
    return actual


def import_attempt(output, source, target_case, source_case, path, overrides, commit=False):
    output, source, path = (allowed(p).resolve(strict=True) for p in (output, source, path))
    if output == source or not path.is_relative_to(source / 'cases'):
        raise SharedFault('需要独立批次内的 attempt')
    with ExitStack() as locks:
        if commit:
            for batch in sorted((output, source)):
                locks.enter_context(lock(batch))
        target, manifest = runner.batch_documents(output)
        origin, source_manifest = runner.batch_documents(source)
        row = case(manifest, target_case)
        source_row = case(source_manifest, source_case)
        config = runner.runtime_config(target, overrides)
        runner.verify_assets(target, config)
        return register(output, source, path, row, source_row, target, origin, config, commit)


def register(output, source, path, row, source_row, target, origin, config, commit):
    index = read(output / 'index.json')
    entry = index['cases'].get(row['case_id'], {})
    if entry.get('adopted') not in (None, str(path)):
        raise SharedFault('目标已有采纳结果')
    if any(key != row['case_id'] and other.get('adopted') == str(path)
           for key, other in index['cases'].items()):
        raise SharedFault('来源 attempt 已采纳到其他 case')
    attempt = compare(index, output, source, path, row, source_row, target, origin, config)
    attempt['imported_from'] = dict(batch=str(source), case_id=source_row['case_id'])
    if commit and not entry.get('adopted'):
        attempts = [a for a in entry.get('attempts', []) if a['path'] != str(path)] + [attempt]
        index['cases'][row['case_id']] = dict(entry, adopted=str(path), attempts=attempts)
        write(output / 'index.json', index)
    return dict(target_case=row['case_id'], source_case=source_row['case_id'], attempt=str(path),
                status='adopted' if commit or entry.get('adopted') else 'eligible',
                checked=['workload', 'models', 'software', 'protocol', 'hardware', 'topology', 'rank_allocation',
                         'rank_capacity', 'native_evidence', 'cleanup', 'kv'], outcomes=attempt['research']['outcomes'])
