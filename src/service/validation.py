"""串行验证 OFF/ON，并保存原始请求与响应。"""
import os
import subprocess
import time
from evidence import save
from cleanup_diagnostics import cleanup_diagnostic, cleanup_operation
from process import finish, failed, stop_group
from protocol import check_port, generate, wait_ready
from runtime import configuration
from .settings import server_command, server_environment, visible_environment


def request_body(config=None):
    config = config or configuration()
    return {'text': 'Explain what a key-value cache is in one sentence.',
            'stream': True, 'sampling_params': {
                'temperature': 0, 'max_new_tokens': config.max_new_tokens,
                'ignore_eos': False, 'stream_interval': 1}}


def run_mode(out, mode, body, result, config):
    args = server_command(config)
    env = server_environment(mode == 'on', 'kvcached_validate_' + str(os.getpid()), config)
    group = dict(mode=mode, command=args, environment=visible_environment(env))
    result['groups'].append(group)
    with (out / (mode + '.server.log')).open('w') as log:
        proc = subprocess.Popen(args, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        group['pid'] = proc.pid
        try:
            wait_ready(proc, config=config)
            group['usage'] = generate(out, mode, body, config)
        finally:
            with cleanup_operation('stop_group', role='service', pid=proc.pid, sid=proc.pid):
                stop_group(proc)


def main(out, config=None):
    config = config or configuration()
    out.mkdir(exist_ok=False)
    result = {'started_epoch': time.time(), 'groups': []}
    body = request_body(config)
    save(out, 'request.json', body)
    try:
        check_port(config)
        for mode in ('off', 'on'):
            run_mode(out, mode, body, result, config)
    except Exception as error:
        result['error'] = repr(error)
        if getattr(error, 'cleanup_context', None):
            result.setdefault('cleanup_diagnostics', []).append(cleanup_diagnostic(error))
    finally:
        # 各 mode 已清理；这里只传组 ID 保存快照，不再次发送信号。
        finish(out, 'result.json', result, [],
               [group['pid'] for group in result['groups'] if 'pid' in group])
    return failed(result)
