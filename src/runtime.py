"""CLI 与已载入进程环境的简单配置；不读取环境文件。"""
import argparse
import os
import sys
from types import SimpleNamespace

DEFAULTS = {
    'devices': (str, None), 'tp_size': (int, 1),
    'model': (str, 'Qwen/Qwen2.5-0.5B'), 'model_id': (str, None),
    'server_python': (str, sys.executable), 'client_python': (str, None),
    'host': (str, '127.0.0.1'), 'port': (int, 30000),
    'dtype': (str, 'bfloat16'), 'attention_backend': (str, 'triton'),
    'page_size': (int, 1), 'context_length': (int, 2048),
    'max_total_tokens': (int, 8192), 'max_running_requests': (int, 1),
    'mem_fraction_static': (float, 0.2), 'startup_timeout': (float, 180),
    'request_timeout': (float, 120), 'client_timeout': (float, 120),
    'max_new_tokens': (int, 32), 'max_inflight': (int, 1),
    'repetitions': (int, 1), 'catalog': (str, None),
}


def defaults(overrides=None):
    values = {name: os.environ.get('KVREPRO_' + name.upper(), default)
              for name, (_, default) in DEFAULTS.items()}
    values.update(overrides or {})
    return {name: DEFAULTS[name][0](value) if value is not None else None
            for name, value in values.items()}


def options(cli):
    for name, (kind, _) in DEFAULTS.items():
        cli.add_argument('--' + name.replace('_', '-'), type=kind,
                         default=argparse.SUPPRESS, help='环境变量 KVREPRO_' + name.upper())


def configuration(args=None):
    overrides = {key: value for key, value in vars(args).items() if key in DEFAULTS} if args else {}
    result = SimpleNamespace(**defaults(overrides))
    validate_configuration(result)
    return result


def validate_configuration(config):
    positive = ('tp_size', 'port', 'page_size', 'context_length', 'max_total_tokens',
                'max_running_requests', 'startup_timeout', 'request_timeout',
                'client_timeout', 'max_new_tokens', 'max_inflight', 'repetitions')
    for name in positive:
        if not getattr(config, name) > 0:
            raise ValueError(name + ' 必须大于零')
    if config.port >= 65536 or not 0 < config.mem_fraction_static <= 1:
        raise ValueError('port 或 mem-fraction-static 超出范围')


def base_url(config):
    host = '127.0.0.1' if config.host == '0.0.0.0' else config.host
    host = '::1' if host == '::' else host
    return f'http://[{host}]:{config.port}' if ':' in host else f'http://{host}:{config.port}'
