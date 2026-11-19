"""启动独立服务和单请求客户端，回收本次进程组。"""
import os
import subprocess
import time
from runtime import configuration
from process import finish, failed
from protocol import check_port, wait_ready
from service.settings import server_command, server_environment, visible_environment
from . import client


def run(root, out, result, processes, config):
    check_port(config)
    env = server_environment(True, 'kvcached_xbench_' + str(os.getpid()), config)
    args = server_command(config)
    result.update(server_command=args, server_environment=visible_environment(env))
    with (out / 'server.log').open('w') as log:
        proc = subprocess.Popen(args, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        result['server_pid'] = proc.pid
        wait_ready(proc, config=config)
        client.run(root, out, result, processes, config)


def main(root, out, config=None):
    config = config or configuration()
    root, out = root.resolve(), out.resolve()
    out.mkdir(exist_ok=False)
    processes = []
    result = {'started_epoch': time.time()}
    try:
        run(root, out, result, processes, config)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        finish(out, 'execution.json', result, processes)
    return failed(result)
