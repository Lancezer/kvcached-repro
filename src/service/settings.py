"""服务命令和子进程环境；CUDA 可见性与 TP 分别配置。"""
import os
from runtime import configuration


def server_command(config=None):
    config = config or configuration()
    args = [config.server_python, '-s', '-m', 'sglang.launch_server']
    for flag, value in service_arguments(config).items():
        args.extend(['--' + flag, str(value)])
    return args + ['--disable-cuda-graph']


def service_arguments(config):
    names = ('host', 'port', 'tp_size', 'dtype', 'attention_backend', 'page_size',
             'context_length', 'max_total_tokens', 'max_running_requests', 'mem_fraction_static')
    return dict({'model-path': config.model},
                **{name.replace('_', '-'): getattr(config, name) for name in names})


def server_environment(enabled, ipc, config=None):
    config = config or configuration()
    env = dict(os.environ)
    if config.devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = config.devices
    env.update(ENABLE_KVCACHED=str(int(enabled)), KVCACHED_AUTOPATCH=str(int(enabled)),
               KVCACHED_IPC_NAME=ipc, PYTHONNOUSERSITE='1')
    env.setdefault('KVCACHED_LOG_LEVEL', 'DEBUG')
    env.setdefault('SGLANG_ENABLE_POST_CAPTURE_KV_SIZING', '0')
    return env


def visible_environment(env):
    prefixes = ('CUDA_', 'KVCACHED_', 'ENABLE_KVCACHED', 'SGLANG_', 'PYTHON')
    return {key: value for key, value in env.items() if key.startswith(prefixes)}
