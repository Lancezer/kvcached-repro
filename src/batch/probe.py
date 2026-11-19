"""可配置本地日志探针；正则必须对应已核对的实际服务日志，缺失即拒绝。"""
import csv
import importlib.metadata
import io
import json
import os
import re
import subprocess
from pathlib import Path
from .storage import read
from .ipc import model_log_name

BOOL_FIELDS = ('prefix_cache_enabled', 'dp_attention', 'kvcached_enabled')
GEOMETRY = ('layers', 'element_bytes', 'kv_lora_rank', 'qk_rope_head_dim', 'local_kv_heads', 'head_dim')
INT_FIELDS = ('rank', 'pid', 'mps_server_pid', 'active_tokens', 'physical_bytes', 'capacity_tokens', *GEOMETRY)


def cast_fields(value):
    result = dict(value)
    for key in BOOL_FIELDS:
        if key in result:
            if result[key] not in ('true', 'false'):
                raise ValueError(f'{key}须为明确true/false观测')
            result[key] = result[key] == 'true'
    for key in INT_FIELDS:
        if key in result:
            result[key] = None if result[key] == 'none' else int(result[key])
    if 'observed_epoch' in result:
        result['observed_epoch'] = float(result['observed_epoch'])
    return result


def log_rows(directory, rules, kind):
    rows = {}
    for model, patterns in rules[kind + '_patterns'].items():
        path = directory / (model_log_name(model) + '.log')
        expressions = [re.compile(pattern) for pattern in patterns]
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                apply_patterns(rows, model, line, line_number, expressions, path)
    return list(rows.values())


def apply_patterns(rows, model, line, line_number, expressions, path):
    for expression in expressions:
        match = expression.search(line)
        if not match:
            continue
        values = cast_fields({k: v for k, v in match.groupdict().items() if v is not None})
        if 'rank' not in values:
            raise ValueError('每条探针正则必须包含rank命名组，禁止猜测归属')
        key = (model, values['rank'])
        row = rows.setdefault(key, dict(model=model, rank=values['rank'], sources=[]))
        row.update(values)
        row['sources'].append(dict(path=str(path), line=line_number))


def mps_clients(resources, rules):
    if not resources['server_pids']:
        return {}
    env = dict(os.environ, **resources['mps_env'])
    clients = {}
    for server in resources['server_pids']:
        result = subprocess.run([rules['mps_control']], env=env, input=f'get_client_list {server}\n',
                                capture_output=True, text=True, check=True, timeout=rules['probe_seconds'])
        if not all(value.isdigit() for value in result.stdout.split()):
            raise ValueError('无法解析私有MPS实际client列表')
        clients[server] = [int(value) for value in result.stdout.split()]
    return clients


def gpu_inventory(rules):
    args = [rules['nvidia_smi'], '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits']
    result = subprocess.run(args, text=True, capture_output=True, check=True, timeout=rules['probe_seconds'])
    devices, drivers, indices = [], set(), {}
    for index, uuid, name, memory, driver in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        devices.append(dict(uuid=uuid.strip(), name=name.strip(), total_memory_bytes=int(memory) * 1024 * 1024))
        indices['GPU' + index.strip()] = uuid.strip()
        drivers.add(driver.strip())
    return devices, next(iter(drivers)) if len(drivers) == 1 else None, indices


def serving_metadata(ranks, rules):
    devices, driver, indices = gpu_inventory(rules)
    targets = {}
    for model in sorted({r['model'] for r in ranks}):
        targets[model] = [r['gpu'] for r in sorted(ranks, key=lambda r: r['rank']) if r['model'] == model]
    selected = {gpu for values in targets.values() for gpu in values}
    packages = {}
    for name in ('sglang', 'torch', 'kvcached'):
        packages[name] = importlib.metadata.version(name)
    return dict(schema_version=1, devices=[d for d in devices if d['uuid'] in selected],
                target_device_uuids=targets, driver_version=driver, packages=packages,
                links=topology(rules, indices, selected))


def topology(rules, indices, selected):
    if len(selected) < 2:
        return []
    result = subprocess.run([rules['nvidia_smi'], 'topo', '-m'], capture_output=True,
                            text=True, check=True, timeout=rules['probe_seconds'])
    table = re.sub(r"\x1b\[[0-9:;]*m", "", result.stdout)
    lines = [line.split() for line in table.splitlines() if line.strip()]
    header = next((line for line in lines if line[0].startswith('GPU') and line[0] in indices and line[1] in indices), None)
    if header is None:
        raise ValueError('无法解析实际GPU拓扑表')
    columns = [name for name in header if name in indices]
    links = []
    for row in lines:
        if row[0] not in indices or indices[row[0]] not in selected or row == header:
            continue
        for name, link in zip(columns, row[1:]):
            if name != row[0] and indices[name] in selected:
                links.append(dict(source_uuid=indices[row[0]], destination_uuid=indices[name], link=link))
    return links


def with_runtime_programs(rules):
    return dict(rules, mps_control=os.environ['KVREPRO_MPS_CONTROL'],
                nvidia_smi=os.environ['KVREPRO_NVIDIA_SMI'])


def observe(rules_path, kind):
    rules = with_runtime_programs(read(rules_path))
    directory = Path(os.environ['KVREPRO_RESOURCE_DIRECTORY']).resolve(strict=True)
    resources = read(directory / 'resources.json')
    rows = log_rows(directory, rules, kind)
    if not rows:
        raise ValueError('实际日志中无可解析rank证据')
    if kind == 'kv':
        return dict(ranks=rows, basis='日志观测；sources保留行号，相同观测不代表新采样')
    clients = mps_clients(resources, rules)
    for row in rows:
        if 'kind' in row:
            row['geometry'] = {key: row[key] for key in ('kind', *GEOMETRY) if key in row}
        server = row['mps_server_pid']
        if server is not None and row['pid'] not in clients.get(server, []):
            raise ValueError('日志rank PID未在私有MPS server实际接入列表中')
    return dict(ranks=rows, mps_clients=clients, serving_metadata=serving_metadata(rows, rules))
