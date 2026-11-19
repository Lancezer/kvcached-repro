"""薄严格顺序驱动；无请求/排空deadline，不自动复跑非零结果。"""
import copy
import os
import json
import signal
import socket
import subprocess
import time
import re
from pathlib import Path
from . import config as settings, native
from .audit import adopt, validity, external_validity
from cleanup_diagnostics import emit_cleanup_diagnostic
from .ownership import SharedFault, audit, identity, matches, terminate, session_members, boot
from .resources import Resources, cleanup_ipc
from . import ipc
from .environment import snapshot as environment_snapshot
from .storage import lock, read, write
from localpaths import allowed


def freeze(config_path, output, overrides=None):
    if not Path(__file__).is_file():
        raise ValueError('批次冻结需展开src源码部署，以保留可核对代码身份')
    frozen = settings.load(config_path, overrides)
    config = frozen['values']
    hashes = settings.assets(config)
    verify_pool(config, hashes)
    frozen['assets'], frozen['patch'] = hashes, native.patch_identity(config)
    frozen['probe_protocol'] = settings.probe_protocol(config, hashes)
    capture_environment(frozen)
    frozen['hardware_observation'] = observe_hardware(config)
    frozen['code'] = current_code()
    rows = settings.matrix(config, hashes)
    output = allowed(output)
    output.mkdir(exist_ok=False)
    write(output / 'configuration.json', frozen)
    write(output / 'manifest.json', dict(batch_id=settings.digest(frozen), cases=rows))
    write(output / 'index.json', dict(cases={}, pairs={}, invocations=[]))
    return output


def capture_environment(frozen):
    frozen['values']['process_environment'] = environment_snapshot()
    frozen['sources']['process_environment'] = 'process'


def verify_pool(config, hashes):
    from data.source import SHA256
    pool = read(config['pool_manifest'])
    if pool['source']['actual_sha256'] != SHA256 or pool['source']['expected_sha256'] != SHA256:
        raise ValueError('输入池来源不匹配固定普通V3')
    if pool['files']['prompts.jsonl'] != settings.file_hash(config['prompts']):
        raise ValueError('输入池prompts摘要不匹配')
    if not pool['size'] or pool['rules']['output_tokens'] != config['output_tokens']:
        raise ValueError('输入池大小/生成预算不匹配')
    if [m['id'] for m in pool['rules']['models']] != [m['id'] for m in config['models']]:
        raise ValueError('输入池tokenizer交集模型顺序不匹配')
    if pool['tokenizer_assets'] != {m['id']: hashes[m['id']] for m in config['models']}:
        raise ValueError('输入池使用的checkpoint/tokenizer内容与本批次不同')


def shared_preflight(config):
    for model in config['models']:
        if not Path(model['checkpoint']).is_dir():
            raise SharedFault('共享模型目录不可用')
    for model in config['models']:
        with socket.socket() as listener:
            try:
                listener.bind(('127.0.0.1', model['port']))
            except OSError as error:
                raise SharedFault('端口被占用，禁止附着或清理已有服务') from error
    if not config.get('rank_probe'):
        raise SharedFault('必须配置实际rank/路由/MPS接入探针')


def resume_audit(output, clean_orphans, config):
    driver = output / 'driver.json'
    if driver.exists() and matches(read(driver)):
        raise SharedFault('原驱动仍活，拒绝第二执行器')
    for path in output.glob('cases/*/**/lifecycle-*/resources.json'):
        record = read(path)
        if record['intent'] is not None and not intent_from_old_boot(record['intent']):
            raise SharedFault(f'启动登记间隙，归属不明需人工审计: {path}')
        alive = audit(record['records'] + record.get('descendants', []))
        if alive and not clean_orphans:
            raise SharedFault(f'自有孤儿仍活，需显式--clean-orphans: {path}')
        for process in reversed(alive):
            if matches(process):
                try:
                    terminate(process, config['cleanup_seconds'])
                except Exception as error:
                    emit_cleanup_diagnostic(error, operation='terminate', pid=process['pid'],
                                            sid=process['session'], role=process.get('role'))
                    raise
        if any(session_members(r['session']) for r in alive):
            raise SharedFault('孤儿session未清空')
        if alive:
            write(path.parent / ('orphan-audit-' + str(time.time_ns()) + '.json'), dict(cleaned=alive))
        audit_ipc(record, path, clean_orphans, config)


def audit_ipc(record, path, clean_orphans, config):
    allocations = record.get('allocations', [])
    remaining = [p for allocation in allocations for p in ipc.paths(allocation)]
    remaining.extend(root for allocation in allocations if (root := ipc.existing_socket_root(allocation)))
    if remaining and not clean_orphans:
        raise SharedFault('私有IPC残留，需显式--clean-orphans并核对inode/活跃锁')
    if remaining:
        try:
            cleanup_ipc(config, path)
        except Exception as error:
            emit_cleanup_diagnostic(error, operation='cleanup_ipc')
            raise


def intent_from_old_boot(intent):
    owner = intent.get('owner')
    if not owner or owner['host'] != socket.gethostname():
        return False
    return owner['boot'] != boot()


def stopping(signum, frame):
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, signal.SIG_IGN)
    raise InterruptedError(f'执行器收到信号{signum}')


def monitor(proc, resources):
    next_sample = time.monotonic()
    while proc.poll() is None:
        if resources.dead():
            resources.service_death = resources.dead_services()
            proc.send_signal(signal.SIGINT)
            proc.wait()  # 原生取消负责封窗；这里没有测量或排空截止。
            raise RuntimeError('外部自有服务确定死亡，已请求原生取消')
        if time.monotonic() >= next_sample:
            resources.sample()
            next_sample = time.monotonic() + resources.config['sample_seconds']
        time.sleep(resources.config['poll_seconds'])
    return proc.returncode


def invoke(config, row, directory, resources, source, continuing, rerun):
    catalog_path = None
    if not continuing:
        catalog_path = native.catalog(config, row, directory)
        write(directory / 'prepared-hardware.json', config['hardware_observation'])
    args = native.command(config, row, directory, catalog_path, continuing, rerun)
    env = native.environment(config, row, source)
    with (resources.directory / 'client-command.json').open('x') as output:
        import json
        json.dump(args, output)
    proc = resources.launch(args, env, 'client')
    try:
        return monitor(proc, resources)
    except BaseException:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait()
        raise


def metadata(config, directory, observed):
    # 从本态探针观察构建metadata，绝不复制另一态case.json。
    value = observed['serving_metadata']
    if value.get('schema_version') != 1:
        raise SharedFault('实际服务metadata需要schema_version=1')
    expected = {m['id']: [config['logical_devices'][str(i)] for i in m['placement']] for m in config['models']}
    if value.get('target_device_uuids') != expected:
        raise SharedFault('实际服务metadata设备映射不符')
    validate_hardware(config, value)
    path = directory / 'serving-metadata.json'
    write(path, value)
    return value


def validate_hardware(config, observed):
    requirements = config['hardware']
    devices = observed.get('devices') or []
    expected = set(config['logical_devices'].values())
    if {d['uuid'] for d in devices} != expected:
        raise SharedFault('实际硬件设备集合不完整')
    for device in devices:
        if requirements['name_contains'] not in device.get('name', ''):
            raise SharedFault('实际设备型号不符')
        if not requirements['minimum_memory_bytes'] <= device.get('total_memory_bytes', -1) <= requirements['maximum_memory_bytes']:
            raise SharedFault('实际显存资源预算不符')
    if requirements['nvlink']:
        links = {(r['source_uuid'], r['destination_uuid']): r['link'] for r in observed.get('links') or []}
        if any(not links.get((a, b), '').startswith('NV') for a in expected for b in expected if a != b):
            raise SharedFault('无法证明所选设备实际NVLink连接')


def execute(config, row, directory, source, continuing=None, rerun=False, skip=False):
    lifecycle = directory / ('lifecycle-' + str(time.time_ns()))
    lifecycle.mkdir()
    resources = Resources(config, row, lifecycle)
    result = dict(native_code=None, fault=None, cleanup=False, lifecycle=str(lifecycle),
                  started_epoch=time.time(), role='rerun' if rerun else 'original')
    try:
        if not skip:
            start_resources(resources, config, directory, source, continuing)
        result['native_code'] = invoke(config, row, directory, resources, source, continuing, rerun)
    except InterruptedError as error:
        result.update(fault=str(error), interrupted_epoch=time.time())
        raise
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        result['fault'] = str(error)
        if isinstance(error, SharedFault):
            raise
    finally:
        finish_resources(resources, lifecycle, result)
    return result


def finish_resources(resources, lifecycle, result):
    try:
        resources.close()
        result['cleanup'] = True
    finally:
        result['service_death'] = resources.service_death
        code = resources.client_code()
        if code is not None:
            result['native_code'] = code
        result['finished_epoch'] = time.time()
        write(lifecycle / 'execution.json', result)


def start_resources(resources, config, directory, source, continuing):
    resources.start_mps()
    resources.start_services()
    observed = resources.probe('metadata')
    current = metadata(config, directory, observed)
    # 原生 continue 保留旧 case.json，本次实际设备证据以 lifecycle 为准。
    observation = config['hardware_observation']
    current_identity = metadata_identity(current, observation)
    write(resources.directory / 'hardware.json', observation)
    write(resources.directory / 'serving-metadata.json', current)
    if source:
        verify_pair_metadata(Path(source), current_identity)
    if continuing:
        verify_pair_metadata(continuing / 'cases' / resources.row['case_id'], current_identity)
    seal_capacity(config, resources.row, observed)


def rank_capacities(config, observed):
    actual = {model['id']: [None] * model['tp'] for model in config['models']}
    for rank in observed['ranks']:
        model, index = rank['model'], rank['rank']
        capacity = rank.get('capacity_tokens')
        if model not in actual or not 0 <= index < len(actual[model]) or actual[model][index] is not None:
            raise SharedFault('实际 rank 容量归属不完整或重复')
        if type(capacity) is not int or capacity <= 0:
            raise SharedFault('缺少实际 rank 容量证据')
        actual[model][index] = capacity
    if any(None in ranks for ranks in actual.values()):
        raise SharedFault('缺少实际 rank 容量证据')
    return actual


def capacity_path(output, row):
    return output / ('capacity-on.json' if row['mps'] else 'capacity-off.json')


def verify_capacity(output, row, actual):
    path = capacity_path(output, row)
    if not path.is_file():
        raise SharedFault('缺少目标同态 rank 容量基准')
    if read(path) != actual:
        raise SharedFault('同态实际 rank 容量不等价')


def seal_capacity(config, row, observed):
    output = config['capacity_directory']
    actual = rank_capacities(config, observed)
    path = capacity_path(output, row)
    if config['changed_devices'] or path.exists():
        verify_capacity(output, row, actual)
    else:
        write(path, actual)


def verify_pair_metadata(source, current_identity):
    if prepared_identity(source) != current_identity:
        raise SharedFault('配对实际设备/版本/拓扑条件变化')


def prepared_identity(source):
    metadata = read(source / 'case.json')['serving_metadata']
    recorded = read(source.parents[4] / 'prepared-hardware.json')
    return metadata_identity(metadata, recorded)


def hardware_identity(value, mapping):
    expected = set(range(settings.GPU_COUNT))
    logical = {uuid: int(index) for index, uuid in mapping.items()}
    if len(mapping) != settings.GPU_COUNT or set(logical.values()) != expected:
        raise SharedFault('缺少一一对应的 CUDA 逻辑设备证据')
    devices = {logical[d['uuid']]: {k: v for k, v in d.items() if k != 'uuid' and v is not None} for d in value['devices']}
    links = {(logical[r['source_uuid']], logical[r['destination_uuid']]): r['link'] for r in value['links']}
    complete = all(d.get('name') and type(d.get('total_memory_bytes')) is int and d['total_memory_bytes'] > 0
                   for d in devices.values())
    if not complete or set(devices) != expected or len(value['devices']) != len(expected) or not value['driver_version']:
        raise SharedFault('设备形态或驱动证据不完整')
    if set(links) != {(a, b) for a in expected for b in expected if a != b} or len(links) != len(value['links']):
        raise SharedFault('实际设备拓扑证据不完整')
    if any(not link or link in ('N/A', 'X') for link in links.values()):
        raise SharedFault('实际设备拓扑含未知连接')
    return dict(devices=[devices[i] for i in sorted(expected)], driver=value['driver_version'],
                links=[[a, b, links[a, b]] for a, b in sorted(links)])


def metadata_identity(value, observation):
    mapping = observation['mapping']
    hardware = hardware_identity(value, mapping)
    if hardware != hardware_identity(observation, mapping):
        raise SharedFault('服务设备形态或拓扑与本次 CUDA 观察不同')
    logical = {uuid: index for index, uuid in mapping.items()}
    packages = value['packages']
    if packages != observation['packages'] or any(not packages.get(name) for name in ('torch', 'sglang', 'kvcached')):
        raise SharedFault('服务软件版本证据不完整')
    return dict(hardware=hardware, packages=packages,
                placement={model: [logical[uuid] for uuid in ranks]
                           for model, ranks in value['target_device_uuids'].items()})


def record_result(index, row, directory, result, config):
    root = directory / 'native/bench-runs'
    if not root.exists():
        return record_unstarted(index, row, result)
    run = native.locate(directory, row['case_id'])
    values = native.attempts(run, row, config)
    for attempt in values:
        path = Path(attempt['path'])
        original = attempt_execution(directory, run, row, path)
        attempt['research'] = research_verdict(config, row, path, original)
        attempt['external'] = original['lifecycle']
    publish_attempts(index, row, run, values, result, config)


def attempt_execution(directory, run, row, path):
    # 测量窗必须完整落在唯一一次客户端调用内；continue跳过不会取得旧窗归属。
    measurement = path / 'measurement.json'
    if not measurement.is_file():
        raise SharedFault('attempt缺少测量原点，无法唯一恢复外部归属，需人工审计')
    origin = read(measurement)['wall_clock_origin_seconds']
    window = read(path / 'repetition.json').get('window_end_seconds')
    candidates = []
    for execution in directory.glob('lifecycle-*/execution.json'):
        saved = read(execution)
        if lifecycle_contains(execution, saved, run, row, origin, window):
            candidates.append(dict(saved, lifecycle=str(execution.parent)))
    if len(candidates) != 1:
        raise SharedFault('原attempt外部lifecycle关联不唯一，保留证据等待审计')
    return candidates[0]


def lifecycle_contains(execution, saved, run, row, origin, window):
    """命令身份与完整测量时间包围共同证明一次调用的归属。"""
    command = execution.parent / 'client-command.json'
    if not command.is_file() or window is None:
        return False
    args = read(command)
    case = '--case' in args and args[args.index('--case') + 1] == row['case_id']
    continuing = '--continue' in args and Path(args[args.index('--continue') + 1]).resolve() == run.resolve()
    start = saved.get('started_epoch', int(execution.parent.name.removeprefix('lifecycle-')) / 1e9)
    end = saved.get('finished_epoch', execution.stat().st_mtime)
    return (case or continuing) and start <= origin and origin + window <= end


def record_unstarted(index, row, result):
    if result['native_code'] is not None:
        raise SharedFault('客户端未建立原生run；保留日志并阻塞共享链路')
    previous = index['cases'].get(row['case_id'], {})
    failures = previous.get('preparation_failures', []) + [result['lifecycle']]
    saved = dict(result, preparation_failures=failures)
    index['cases'][row['case_id']] = dict(run=None, attempts=[], adopted=None, **saved)


def publish_attempts(index, row, run, values, result, config):
    previous = index['cases'].get(row['case_id'], {})
    known = {a['path']: a for a in previous.get('attempts', [])}
    for value in values:
        known[value['path']] = value  # 允许补齐证据后重审unknown，仍引用原lifecycle。
    attempts = list(known.values())
    adopted = adopt(attempts[0], attempts[1] if len(attempts) > 1 else None) if attempts else None
    entry = dict(previous, **result)
    entry['service_death'] = result.get('service_death', [])
    entry.update(run=str(run), attempts=attempts, adopted=adopted)
    index['cases'][row['case_id']] = entry
    original = read(run / 'run.json')
    index['cases'][row['case_id']]['native_infrastructure_error'] = original.get('infrastructure_error')
    classify_native_fault(config, index['cases'][row['case_id']], original, run)
    prepared = run / 'cases' / row['case_id']
    if (prepared / 'kvrepro-frozen.json').is_file():
        index['pairs'].setdefault(row['pair_id'], str(prepared))
    else:
        raise SharedFault('原生run没有完整冻结来源；禁止另一态重新抽样冒充配对')


def classify_native_fault(config, entry, original, run):
    error = original.get('infrastructure_error')
    if 'service_death' in entry and entry['service_death'] is None:
        raise SharedFault('外层观察到非服务自有根进程死亡，共享前提不明')
    if not error and not entry.get('service_death'):
        return
    scope = local_fault_scope(entry, error)
    if scope is None or not entry['cleanup']:
        raise SharedFault('共享或安全边界不确定的原生故障: ' + str(error))
    frozen = next((p / 'configuration.json' for p in run.parents if (p / 'configuration.json').is_file()), None)
    if frozen is None:
        raise SharedFault('无法复核共享冻结前提')
    verify_frozen(read(frozen), config)
    shared_preflight(config)
    recheck_devices(config)
    entry['native_fault_scope'] = scope


def local_fault_scope(entry, error):
    if re.fullmatch(r'benchmark worker exited with code -?\d+', error or ''):
        return 'local_worker_exit_after_verified_cleanup'
    cancellations = (None, 'benchmark measurement did not complete',
                     'worker lost before terminal request accounting', 'task cancelled; scope safely drained')
    # 事实仅由monitor观察自有服务Popen退出后记录；错误字符串本身不能证明服务死亡。
    deaths = entry.get('service_death')
    confirmed = isinstance(deaths, list) and deaths and all(
        isinstance(d, dict) and type(d.get('pid')) is int and type(d.get('returncode')) is int
        and isinstance(d.get('observed_epoch'), (int, float)) for d in deaths)
    if confirmed and error in cancellations:
        return 'local_service_death_after_verified_cleanup'
    return None


def cuda_observation(config):
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise SharedFault('正式批次需要显式 CUDA_VISIBLE_DEVICES')
    script = (
        'import json, torch; from importlib.metadata import version; '
        'uuids=[str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]; '
        'print(json.dumps(dict(uuids=uuids, packages={n:version(n) for n in ("torch","sglang","kvcached")})))'
    )
    result = subprocess.run([config['server_python'], '-s', '-c', script], capture_output=True,
                            text=True, check=True, timeout=config['probe_seconds'],
                            env=dict(os.environ, ENABLE_KVCACHED='0', KVCACHED_AUTOPATCH='0'))
    values = json.loads(result.stdout)
    uuids = [v if v.startswith('GPU-') else 'GPU-' + v for v in values['uuids']]
    if len(uuids) != settings.GPU_COUNT or len(set(uuids)) != settings.GPU_COUNT:
        raise SharedFault('CUDA 运行时须实际可见五张不同设备')
    return uuids, values['packages']


def observe_hardware(config):
    from .probe import gpu_inventory, topology
    uuids, packages = cuda_observation(config)
    rules = {name: config[name] for name in ('nvidia_smi', 'probe_seconds')}
    devices, driver, indices = gpu_inventory(rules)
    mapping = {str(i): uuid for i, uuid in enumerate(uuids)}
    observation = dict(mapping=mapping, packages=packages, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                       devices=[d for d in devices if d['uuid'] in uuids], driver_version=driver,
                       links=topology(rules, indices, set(uuids)))
    validate_hardware(dict(config, logical_devices=mapping), observation)
    hardware_identity(observation, mapping)
    return observation


def verify_hardware(baseline, observed):
    if hardware_identity(baseline, baseline['mapping']) != hardware_identity(observed, observed['mapping']):
        raise SharedFault('实际设备形态、驱动或逻辑拓扑不等价')
    if baseline['packages'] != observed['packages']:
        raise SharedFault('实际服务软件版本不等价')


def recheck_devices(config):
    verify_hardware(config['hardware_observation'], observe_hardware(config))


def check_rerun(index, row, selected, reason, directory=None):
    if row['case_id'] not in selected:
        return False
    entry = index['cases'].get(row['case_id'])
    if not entry or not reason:
        raise ValueError('复跑必须引用已有无效attempt并说明原因')
    if entry.get('rerun_used') or (directory is not None and persisted_rerun(directory)):
        raise ValueError('该case已使用一次显式复跑')
    if not entry['attempts']:
        if not entry.get('fault'):
            raise ValueError('没有可证明无效的启动链路')
        return True
    if len(entry['attempts']) >= 2 or entry['attempts'][0]['research']['valid'] is not False:
        raise ValueError('只允许明确无效原尝试的一次显式复跑；未知须先补审计')
    return True


def persisted_rerun(directory):
    # marker先于执行落盘；即使启动前失败或索引写入中断，也不能再消费复跑额度。
    if (directory / 'research-rerun').exists() or any(directory.rglob('rerun.json')):
        return True
    return any(read(path).get('role') == 'rerun' for path in directory.glob('**/execution.json'))


def run_case(config, row, output, index, selected, reason):
    case_id = row['case_id']
    directory = output / 'cases' / case_id
    directory.mkdir(parents=True, exist_ok=True)
    rerun = check_rerun(index, row, selected, reason, directory)
    entry = index['cases'].get(case_id)
    directory, continuing = case_location(index, row, directory, entry, rerun)
    if not rerun and recover_existing(config, row, directory, continuing, index):
        return
    if rerun:
        write(directory / 'rerun.json', dict(role='rerun', reason=reason, epoch=time.time()))
    result = launch_case(config, row, directory, output, index, continuing, rerun)
    result = case_audit(entry or {}, result, directory)
    record_result(index, row, directory, result, config)


def recover_existing(config, row, directory, run, index):
    try:
        if run and recover_completed(config, row, directory, run, index):
            return True
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SharedFault('恢复证据无法完成审计，禁止重新发压') from error
    if directory.name == 'research-rerun':
        raise SharedFault('固定复跑角色缺少完整可恢复证据，保留一次额度等待审计')
    entry = index['cases'].get(row['case_id'])
    return bool(entry and entry.get('run') is None and run is None)


def case_audit(previous, result, directory):
    saved = dict(result)
    for key in ('rerun_used', 'rerun_reason', 'preparation_failures'):
        if key not in saved and key in previous:
            saved[key] = previous[key]
    marker = directory / 'rerun.json'
    if marker.is_file():
        saved.update(rerun_used=True, rerun_reason=read(marker)['reason'])
    # 返回结果只携带执行和case审计字段，原生attempt仍由record_result重新审计。
    return {key: value for key, value in saved.items()
            if key not in ('run', 'attempts', 'adopted', 'native_infrastructure_error', 'native_fault_scope')}


def recover_completed(config, row, directory, run, index):
    attempts = native.attempts(run, row, config)
    if not attempts or not all(a['manifest'].get('finished') for a in attempts):
        return False
    original = attempt_execution(directory, run, row, Path(attempts[-1]['path']))
    result = case_audit(index['cases'].get(row['case_id'], {}), original, directory)
    record_result(index, row, directory, result, config)
    return True


def case_location(index, row, directory, entry, rerun):
    continuing = continuing_run(index, row, directory)
    if continuing:
        directory = continuing.parents[2]
    if rerun and (not entry['attempts'] or entry['attempts'][0]['manifest'].get('result_code') == 0):
        directory = directory / 'research-rerun'
        directory.mkdir(exist_ok=False)
        continuing = None
    return directory, continuing


def launch_case(config, row, directory, output, index, continuing, rerun):
    entry = index['cases'].get(row['case_id'])
    skip = bool(continuing and entry and entry['attempts'] and not rerun)
    if not skip:
        if config['changed_devices'] and not capacity_path(output, row).is_file():
            raise SharedFault('换卡续跑缺少目标同态 rank 容量基准')
        shared_preflight(config)
    source = index['pairs'].get(row['pair_id'])
    if source and not Path(source).resolve().is_relative_to(output / 'cases'):
        raise SharedFault('索引prepared来源越出自有批次域')
    return execute(config, row, directory, source, continuing, rerun, skip)


def research_verdict(config, row, path, result, persist=True):
    verdict = validity(path, result['cleanup'])
    if result['fault']:
        verdict.update(valid=False, reason='外部执行链路确定中断: ' + result['fault'])
    try:
        observed = native.validate_attempt(config, row, path)
        if persist:
            write(Path(result['lifecycle']) / ('verification-' + path.name + '.json'), observed)
        if not observed['evidence_complete'] and verdict['valid'] is not False:
            return dict(verdict, valid=None, reason='原生校验表明证据不完整')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if verdict['valid'] is False:
            return dict(verdict, missing_native_evidence=str(error))
        return dict(verdict, valid=None, reason='原生资产校验拒绝: ' + str(error))
    return external_validity(result, verdict, config, path)


def continuing_run(index, row, directory):
    recovered = research_rerun_run(directory, row)
    if recovered is not None:
        return recovered
    entry = index['cases'].get(row['case_id'])
    if entry and entry.get('run'):
        run = Path(entry['run']).resolve()
        if not run.is_relative_to(directory) or native.locate(run.parents[2], row['case_id']) != run:
            raise SharedFault('索引原生run归属或身份不符')
        return run
    if (directory / 'native/bench-runs').exists():
        return native.locate(directory, row['case_id'])
    return None


def research_rerun_run(directory, row):
    # 只发现固定角色；完成态与lifecycle归属仍由既有恢复链路审计。
    role = directory / 'research-rerun'
    if not role.exists() and not role.is_symlink():
        return None
    try:
        marker = read(role / 'rerun.json')
        if role.is_symlink() or not isinstance(marker, dict) or marker.get('role') != 'rerun' or not marker.get('reason'):
            raise ValueError('固定角色marker不完整或目录非自有')
        run = native.locate(role, row['case_id'])
        if not run.resolve().is_relative_to(role.resolve()):
            raise ValueError('固定角色原生run越出自有目录')
        return run
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SharedFault('固定复跑角色无唯一自有run证据，保留一次额度等待审计') from error


def batch_documents(output):
    frozen, manifest = read(output / 'configuration.json'), read(output / 'manifest.json')
    if not {'hardware_observation', 'probe_protocol'} <= frozen.keys():
        raise SharedFault('批次缺少当前格式的冻结设备形态与拓扑证据')
    if manifest['batch_id'] != settings.digest(frozen) or manifest['cases'] != settings.matrix(frozen['values'], frozen['assets']):
        raise SharedFault('冻结配置/矩阵身份不符')
    return frozen, manifest


def run(output, clean_orphans=False, selected=(), reason=None, overrides=None):
    output = allowed(output).resolve(strict=True)
    frozen, manifest = batch_documents(output)
    config = runtime_config(frozen, overrides)
    verify_frozen(frozen, config)
    with lock(output):
        index = read(output / 'index.json')
        observation = observe_hardware(config)
        verify_hardware(frozen['hardware_observation'], observation)
        config.update(logical_devices=observation['mapping'], hardware_observation=observation,
                      capacity_directory=output,
                      changed_devices=observation['mapping'] != frozen['hardware_observation']['mapping'])
        resume_audit(output, clean_orphans, config)
        write(output / 'driver.json', identity(os.getpid()))
        handlers = {s: signal.signal(s, stopping) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            return drive(config, manifest, output, index, selected, reason)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def runtime_config(frozen, overrides):
    config = copy.deepcopy(frozen['values'])
    settings.resolve_paths(config, overrides)
    if config != frozen['values']:
        raise SharedFault('运行路径与冻结条件不同，原生 catalog 仍绑定原路径')
    return config


def verify_frozen(frozen, config):
    verify_assets(frozen, config)
    if environment_snapshot() != config['process_environment']:
        raise SharedFault('已冻结的服务运行环境变化，必须新建批次')


def verify_assets(frozen, config):
    if settings.assets(config) != frozen['assets'] or native.patch_identity(config) != frozen['patch']:
        raise SharedFault('冻结资产或独立客户端补丁身份变化，必须新建批次')
    if settings.probe_protocol(config, frozen['assets']) != frozen['probe_protocol']:
        raise SharedFault('冻结探针协议变化')
    if current_code() != frozen['code']:
        raise SharedFault('执行器代码变化；不能静默恢复旧批次')


def current_code():
    root = Path(__file__).parents[1]
    return {str(p.relative_to(root)): settings.file_hash(p) for p in root.rglob('*.py')}


def drive(config, manifest, output, index, selected, reason):
    if set(selected) - {row['case_id'] for row in manifest['cases']}:
        raise ValueError('显式复跑case不属于本批次')
    invocation = dict(start=time.time(), status='running')
    index['invocations'].append(invocation)
    try:
        for row in manifest['cases']:
            drive_case(config, row, output, index, selected, reason)
        invocation['status'] = 'finished'
    except InterruptedError as error:
        invocation.update(status='interrupted', reason=str(error))
    except Exception as error:
        invocation.update(status='blocked', reason=str(error))
    finally:
        invocation['end'] = time.time()
        invocation['case_summary'] = case_summary(index)
        write(output / 'index.json', index)
        note('invocation-end', **invocation)
    return 0 if invocation['status'] == 'finished' else 2


def drive_case(config, row, output, index, selected, reason):
    if index['cases'].get(row['case_id'], {}).get('adopted') and row['case_id'] not in selected:
        return
    note('case-start', case_id=row['case_id'], pair_id=row['pair_id'], mps=row['mps'])
    run_case(config, row, output, index, selected, reason)
    write(output / 'index.json', index)
    entry = index['cases'].get(row['case_id'], {})
    note('case-end', case_id=row['case_id'], native_code=entry.get('native_code'),
         adopted=entry.get('adopted'), fault=entry.get('fault'))


def note(stage, **fields):
    print(json.dumps(dict(epoch=time.time(), stage=stage, **fields), ensure_ascii=False), flush=True)


def case_summary(index):
    result = dict(adopted=0, invalid=0, unknown=0, preparation_failed=0)
    for entry in index['cases'].values():
        if entry.get('adopted'):
            result['adopted'] += 1
        elif not entry['attempts']:
            result['preparation_failed' if entry.get('fault') else 'unknown'] += 1
        elif any(a['research']['valid'] is None for a in entry['attempts']):
            result['unknown'] += 1
        else:
            result['invalid'] += 1
    return result
