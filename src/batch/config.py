"""配置冻结与稳定矩阵身份；不探测设备、不读取环境文件。"""
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from uuid import UUID
from localpaths import allowed

DEFAULTS = dict(bucket_seconds=1.0, warmup=1, max_inflight=128, startup_seconds=1800.0,
                cleanup_seconds=10.0, poll_seconds=.2, sample_seconds=1.0, probe_seconds=10.0,
                mps_states=[False, True], sampling=dict(temperature=0.0, ignore_eos=False, stream_interval=1))
PATHS = ('source', 'suite', 'client_python', 'server_python', 'prompts', 'pool_manifest',
         'probe_assets_manifest', 'mps_control', 'nvidia_smi', 'rank_probe', 'kv_probe')
MODEL_PATHS = ('checkpoint', 'tokenizer', 'metadata', 'assets_manifest')
EXECUTABLES = {'client_python', 'server_python', 'mps_control', 'nvidia_smi'}
GPU_COUNT = 5


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with allowed(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def readable(path):
    path = allowed(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'需要本地文件: {path}')
    return str(path)


def input_value(name, overrides):
    key = 'KVREPRO_' + name.upper()
    value = (overrides or {}).get(name)
    if value is not None:
        return value, 'cli'
    if key in os.environ:
        return os.environ[key], 'environment'
    raise ValueError(f'缺少 --{name.replace("_", "-")} 或 {key}')


def local_path(value, name, directory=False, executable=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} 需要非空本地路径')
    path = allowed(Path.cwd() / value, preserve_symlinks=executable)
    usable = path.is_dir() if directory else path.is_file()
    if not usable or not os.access(path, os.X_OK if executable else os.R_OK):
        raise ValueError(f'{name} 路径不可用: {path}')
    return str(path)


def probe_command(value, name):
    args = json.loads(value)
    if not isinstance(args, list) or not args or not all(isinstance(arg, str) for arg in args):
        raise ValueError(f'{name} 需要 JSON 命令数组')
    if not Path(args[0]).is_absolute():
        raise ValueError(f'{name} 可执行程序需要绝对路径')
    args[0] = local_path(args[0], name, executable=True)
    return args


def resolve_paths(config, overrides):
    sources = {}
    for name in PATHS:
        value, sources[name] = input_value(name, overrides)
        config[name] = (probe_command(value, name) if name in ('rank_probe', 'kv_probe') else
                        local_path(value, name, directory=name == 'source', executable=name in EXECUTABLES))
    for name in MODEL_PATHS:
        key = 'model_' + name
        values, sources[key] = input_value(key, overrides)
        values = json.loads(values) if isinstance(values, str) else values
        if not isinstance(values, list) or len(values) != len(config['models']):
            raise ValueError(f'{key} 须按模型顺序提供每个路径')
        for model, value in zip(config['models'], values):
            model[name] = local_path(value, key, directory=name != 'assets_manifest')
    return sources


def load(path, overrides=None):
    config = json.loads(allowed(path).read_text())
    reject_retired(config)
    sources = {key: 'config' for key in config}
    environment_defaults(config, sources)
    if (overrides or {}).get('duration') is not None:
        config['duration'], sources['duration'] = overrides['duration'], 'cli'
    sources.update(resolve_paths(config, overrides))
    validate(config)
    return dict(values=config, sources=sources)


def reject_retired(config):
    retired = set(config) & {*PATHS, 'device_map', 'devices', 'mps'}
    for model in config['models']:
        retired.update(set(model) & {*MODEL_PATHS, 'devices'})
    if retired:
        raise ValueError('实验 JSON 含已移除的字段: ' + ', '.join(sorted(retired)))


def environment_defaults(config, sources):
    names = (*DEFAULTS, 'duration', 'rates', 'seeds', 'output_tokens')
    for name in names:
        key = 'KVREPRO_' + name.upper()
        if name not in config and key in os.environ:
            config[name] = json.loads(os.environ[key])
            sources[name] = 'environment'
    for name, value in DEFAULTS.items():
        if name not in config:
            config[name], sources[name] = copy.deepcopy(value), 'default'


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} 必须为数值')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} 必须为有限正数')


def validate(config):
    validate_runtime(config)
    validate_workload(config)
    for name in ('duration', 'bucket_seconds', 'max_inflight', 'output_tokens'):
        positive(config[name], name)
    ids = [model['id'] for model in config['models']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('模型ID必须非空且唯一')
    if len(set(config['seeds'])) != len(config['seeds']) or not config['seeds']:
        raise ValueError('负载seed必须独立且非空')
    if any(type(seed) is not int for seed in config['seeds']):
        raise ValueError('seed必须为整数')
    if len(config['mps_states']) != 2 or any(type(v) is not bool for v in config['mps_states']) or set(config['mps_states']) != {False, True}:
        raise ValueError('MPS配对需要且仅需要两态')
    for rate in config['rates']:
        positive(rate, 'rate')
    if len(set(config['rates'])) != len(config['rates']) or not config['rates']:
        raise ValueError('RPS重复或为空')
    validate_models(config)


def validate_workload(config):
    for key in ('max_inflight', 'output_tokens'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key}必须为正整数')
    sampling = config['sampling']
    if set(sampling) != {'temperature', 'stream_interval', 'ignore_eos'}:
        raise ValueError('sampling必须显式包含temperature/stream_interval/ignore_eos')
    temperature = sampling['temperature']
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature非法')
    if type(sampling['stream_interval']) is not int or sampling['stream_interval'] <= 0:
        raise ValueError('stream_interval必须为正整数')
    if type(sampling['ignore_eos']) is not bool:
        raise ValueError('ignore_eos必须显式为布尔值')


def validate_runtime(config):
    validate_hardware_config(config)
    for name in ('source', 'suite', 'client_python', 'server_python', 'prompts', 'pool_manifest'):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f'必须显式配置本地{name}；模板null不能运行')
    for name in ('startup_seconds', 'cleanup_seconds', 'poll_seconds', 'sample_seconds', 'probe_seconds'):
        positive(config[name], name)
    for name in ('rank_probe', 'kv_probe'):
        if not isinstance(config.get(name), list) or not config[name] or not all(isinstance(v, str) for v in config[name]):
            raise ValueError(f'{name}必须为无shell命令数组')
    if type(config['warmup']) is not int or config['warmup'] < 0:
        raise ValueError('warmup必须为非负整数')


def validate_hardware_config(config):
    hardware = config.get('hardware', {})
    for key in ('minimum_memory_bytes', 'maximum_memory_bytes'):
        positive(hardware.get(key), key)
    if not isinstance(hardware.get('name_contains'), str) or not hardware['name_contains']:
        raise ValueError('必须显式提供设备型号条件')
    if type(hardware.get('nvlink')) is not bool:
        raise ValueError('nvlink必须显式提供布尔条件')
    if hardware['minimum_memory_bytes'] > hardware['maximum_memory_bytes']:
        raise ValueError('显存边界不合法')


def validate_models(config):
    ports = []
    for model in config['models']:
        validate_model_types(model)
        validate_model_identity(model['id'])
        if len(model['placement']) != model['tp'] or model['tp'] % model['dp']:
            raise ValueError('placement必须匹配TP，DP必须整除TP')
        if len(set(model['placement'])) != len(model['placement']):
            raise ValueError('模型内设备重复')
        if model['context_length'] < config['output_tokens']:
            raise ValueError('生成预算超过上下文')
        validate_geometry(model)
        ports.append(model['port'])
    if len(ports) != len(set(ports)):
        raise ValueError('服务端口重复')
    if {i for model in config['models'] for i in model['placement']} != set(range(GPU_COUNT)):
        raise ValueError('正式 placement 必须覆盖逻辑设备 0 至 4')


def validate_model_identity(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value) is None:
        raise ValueError('模型ID须符合原生namespace/name格式')
    if any(part in ('.', '..') for part in value.split('/')):
        raise ValueError('模型ID不能含路径转义分量')


def placement_step(placement):
    step = placement[1] - placement[0] if len(placement) > 1 else 1
    if step <= 0 or list(range(placement[0], placement[0] + step * len(placement), step)) != placement:
        raise ValueError('placement 需要可由 SGLang 起始逻辑设备和步长表达')
    return step


def validate_model_types(model):
    positive(model.get('mem_fraction_static'), 'mem_fraction_static')
    for key in ('tp', 'dp', 'context_length', 'port'):
        if type(model.get(key)) is not int or model[key] <= 0:
            raise ValueError(f'{model["id"]}: {key}必须为正整数')
    if model.get('max_total_tokens') != 'auto' and (type(model.get('max_total_tokens')) is not int or model['max_total_tokens'] <= 0):
        raise ValueError('max_total_tokens须为auto或显式正整数虚拟容量上界')
    for key in ('checkpoint', 'tokenizer', 'metadata', 'assets_manifest'):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f'{model["id"]}: 必须提供本地{key}')
        allowed(model[key])
    if model['port'] > 65535 or not 0 < model['mem_fraction_static'] <= 1:
        raise ValueError('端口或静态显存余量策略非法')
    if type(model['dp_attention']) is not bool:
        raise ValueError('dp_attention必须为布尔值')
    if not model['placement'] or any(type(rank) is not int or rank < 0 for rank in model['placement']):
        raise ValueError('逻辑设备编号必须为非负整数')
    placement_step(model['placement'])


def validate_geometry(model):
    from .audit import kv_bytes
    values = model.get('geometry_by_rank')
    if not isinstance(values, dict) or set(values) != {str(i) for i in range(model['tp'])}:
        raise ValueError('每个rank必须提供实际几何，不能统一除TP或乘DP')
    for geometry in values.values():
        kv_bytes(1, geometry)
    reserved = ('--tp-size', '--dp-size', '--model-path', '--port', '--host', '--context-length',
                '--base-gpu-id', '--gpu-id-step', '--tokenizer-path', '--max-total-tokens', '--mem-fraction-static', '--enable-dp-attention', '--disable-radix-cache')
    if any(arg.split('=')[0] in reserved for arg in model.get('arguments', [])):
        raise ValueError('额外服务参数不得覆盖冻结核心条件')


def assets(config):
    names = ('prompts', 'pool_manifest', 'client_python', 'server_python', 'mps_control', 'nvidia_smi')
    hashes = {name: file_hash(readable(config[name])) for name in names}
    hashes['probes'] = probe_assets(config)
    for model in config['models']:
        for name in ('checkpoint', 'tokenizer'):
            if not model[name] or not allowed(model[name]).is_dir():
                raise ValueError(f'{model["id"]}: 缺少{name}')
        if model['metadata'] != model['checkpoint'] or model['tokenizer'] != model['checkpoint']:
            raise ValueError('固定原生from_checkpoint要求metadata/tokenizer为同一完整checkpoint目录')
        hashes[model['id']] = verify_inventory(model)
    return hashes


def probe_assets(config):
    manifest = Path(readable(config['probe_assets_manifest']))
    declaration = json.loads(manifest.read_text())
    if not isinstance(declaration, dict) or not declaration:
        raise ValueError('探针清单须为非空对象')
    root = allowed(manifest.parent / declaration.get('root', '.'))
    inventory = declaration.get('files', declaration)
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('必须显式列出探针程序及脚本/rules资产摘要')
    actual = {}
    for relative, expected in inventory.items():
        path = allowed(root / relative).resolve(strict=True)
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError('探针清单须使用声明root内的相对路径')
        if file_hash(path) != expected:
            raise ValueError('探针资产摘要不符: ' + relative)
        actual[str(path)] = expected
    verify_probe_commands(config, actual)
    return dict(manifest=file_hash(manifest), files=actual)


def verify_probe_commands(config, inventory):
    for name in ('rank_probe', 'kv_probe'):
        args = config[name]
        if '-c' in args or '-m' in args or Path(args[0]).name == 'env':
            raise ValueError('探针须使用显式脚本路径，不能以-c/-m隐藏程序资产')
        if not Path(args[0]).is_absolute() or str(allowed(args[0]).resolve(strict=True)) not in inventory:
            raise ValueError('探针可执行程序必须纳入清单')
        verify_probe_script(args, inventory)
        rules = probe_rules(args)
        if rules is not None:
            if str(rules) not in inventory:
                raise ValueError('探针rules文件须纳入清单')
            verify_rule_programs(config, rules, inventory)
        for arg in args[1:]:
            candidate = arg.split('=', 1)[1] if arg.startswith('--') and '=' in arg else arg
            if '/' in candidate and (not Path(candidate).is_absolute() or str(allowed(candidate)) not in inventory):
                raise ValueError('探针路径参数须为绝对路径且纳入清单')


def verify_probe_script(args, inventory):
    executable = Path(args[0])
    interpreters = r'python(?:\d+(?:\.\d+)*)?|(?:ba|da|k|z)?sh'
    if not any(re.fullmatch(interpreters, name) for name in (executable.name, executable.resolve().name)):
        return
    options = {'-s', '-S', '-u', '-B', '-I', '-E', '-O', '-OO', '-P', '-Es', '-eu', '-e'}
    script = next((arg for arg in args[1:] if arg not in options), '')
    if not Path(script).is_absolute() or str(allowed(script)) not in inventory:
        raise ValueError('探针解释器须使用清单内的绝对脚本路径')


def probe_rules(args):
    """只接受唯一完整参数名，避免argparse缩写或末值覆盖产生歧义。"""
    values = []
    for index, arg in enumerate(args):
        if arg == '--':
            raise ValueError('探针参数不得使用位置参数分隔符隐藏rules')
        option = arg.split('=', 1)[0]
        if option.startswith('--') and option != '--' and '--rules'.startswith(option) and option != '--rules':
            raise ValueError('rules参数不得缩写')
        if option != '--rules':
            continue
        value = arg.split('=', 1)[1] if '=' in arg else (args[index + 1] if index + 1 < len(args) else '')
        if not value or value.startswith('-') or not Path(value).is_absolute():
            raise ValueError('rules须为非空绝对路径')
        values.append(allowed(value).resolve(strict=True))
    if len(values) > 1:
        raise ValueError('rules参数重复，实际消费有歧义')
    return values[0] if values else None


def verify_rule_programs(config, path, inventory):
    rules = json.loads(Path(readable(path)).read_text())
    for name in ('mps_control', 'nvidia_smi'):
        if name in rules:
            raise ValueError('探针 rules 含已移除的程序路径: ' + name)
        program = config[name]
        if not isinstance(program, str) or not Path(program).is_absolute():
            raise ValueError('实际探针程序须为显式绝对路径')
        if str(allowed(program).resolve(strict=True)) not in inventory:
            raise ValueError('实际探针程序未纳入资产清单')


def verify_inventory(model):
    root = allowed(model['checkpoint']).resolve(strict=True)
    manifest = json.loads(Path(readable(model['assets_manifest'])).read_text())
    if not manifest or not isinstance(manifest, dict):
        raise ValueError('checkpoint资产清单必须为相对路径到SHA256的非空对象')
    for relative, expected in manifest.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or file_hash(readable(path)) != expected:
            raise ValueError(f'checkpoint/tokenizer清单校验失败: {relative}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest):
        raise ValueError('资产清单必须覆盖完整checkpoint与tokenizer，不能省略权重')
    return digest(manifest)


def protocol_assets(asset_hashes):
    result = dict(asset_hashes)
    if 'probes' in result:
        result['probes'] = sorted(result['probes']['files'].values())
    return result


def probe_protocol(config, assets):
    # 以资产摘要代替路径，仍保留命令参数及其顺序。
    hashes = assets['probes']['files']
    result = {}
    for name in ('rank_probe', 'kv_probe'):
        args = []
        for arg in config[name]:
            flag, separator, value = arg.partition('=') if arg.startswith('--') else (arg, '', '')
            path = value if separator else flag
            if Path(path).is_absolute():
                path = hashes.get(str(Path(path).resolve()), path)
            args.append(flag + separator + path if separator else path)
        result[name] = args
    return result


def pair_conditions(config, rate, seed, asset_hashes):
    models = [{k: v for k, v in m.items() if k not in ('port', *MODEL_PATHS)}
              for m in config['models']]
    names = ('duration', 'bucket_seconds', 'warmup', 'output_tokens', 'sampling', 'max_inflight')
    return dict(models=models, rate=rate, seed=seed, assets=protocol_assets(asset_hashes),
                **{name: config[name] for name in names})


def matrix(config, asset_hashes):
    rows = []
    for rate in config['rates']:
        for repeat, seed in enumerate(config['seeds']):
            pair = pair_conditions(config, rate, seed, asset_hashes)
            states = config['mps_states'][::1 if repeat % 2 == 0 else -1]
            for enabled in states:
                execution = dict(pair=pair, mps=enabled,
                                 environment=config.get('process_environment'),
                                 startup=config['startup_seconds'])
                identity = digest(execution)
                # 原生要求UUIDv4格式；内容摘要仍另存完整256位，不能只凭UUID认同条件。
                case_id = str(UUID(hex=identity[:32], version=4))
                rows.append(dict(pair_id=digest(pair), case_id=case_id,
                                 fingerprint=identity, mps=enabled, conditions=copy.deepcopy(pair)))
    if len({row['case_id'] for row in rows}) != len(rows):
        raise ValueError('case身份重复')
    return rows
