"""独立 xbench CLI 的参数、环境与工作目录。"""
import os
import subprocess
from runtime import configuration
from .catalog import runtime_catalog


def environment(root):
    env = dict(os.environ)
    env.update(ENABLE_KVCACHED='0', KVCACHED_AUTOPATCH='0', PYTHONNOUSERSITE='1',
               PYTHONPATH=f'{root}/source/src:{root}/source/src/xpool-dev')
    return env


def command(root, out, config=None, catalog=None):
    config = config or configuration()
    return [config.client_python or str(root / 'client-env/bin/python'), '-s', '-c',
            'from xbench.cli import main; raise SystemExit(main())', 'run', '--all',
            '--catalog', str(catalog or root / 'config/catalog.toml'), '--repetitions', str(config.repetitions),
            '--cache-root', str(out / 'artifacts')]


def run(root, out, result, processes, config):
    catalog = runtime_catalog(root, out, config)
    env, args = environment(root), command(root, out, config, catalog)
    result['client_command'] = args
    result['client_environment'] = {key: env[key] for key in
                                    ('CUDA_VISIBLE_DEVICES', 'PYTHONNOUSERSITE', 'PYTHONPATH') if key in env}
    with (out / 'client.log').open('w') as log:
        proc = subprocess.Popen(args, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        result['client_pid'] = proc.pid
        result['client_rc'] = proc.wait(timeout=config.client_timeout)
    if result['client_rc']:
        raise RuntimeError('xbench 返回失败，见 client.log')
