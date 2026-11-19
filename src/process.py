"""本次启动的进程组清理与收尾证据。"""
import os
import signal
import subprocess
import time
from evidence import save
from cleanup_diagnostics import cleanup_diagnostic, cleanup_operation


def stop_group(proc):
    # launcher 退出后 worker 仍可能存活，因此始终清理本次创建的进程组。
    for sig, delay in ((signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        try:
            with cleanup_operation('killpg', pid=proc.pid, signal=int(sig)):
                os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        with cleanup_operation('sleep'):
            time.sleep(delay)
    with cleanup_operation('wait'):
        proc.wait(timeout=10)


def cleanup(processes, result):
    for proc in reversed(processes):
        try:
            stop_group(proc)
        except Exception as error:
            result.setdefault('cleanup_errors', []).append(repr(error))
            role = None
            if proc.pid == result.get('server_pid'):
                role = 'service'
            elif proc.pid == result.get('client_pid'):
                role = 'client'
            result.setdefault('cleanup_diagnostics', []).append(
                cleanup_diagnostic(error, operation='stop_group', pid=proc.pid,
                                   role=role, sid=proc.pid if role else None))


def final_snapshot(out, groups=()):
    if groups:
        result = subprocess.run(['ps', '-eo', 'pid,ppid,pgid,args'],
                                capture_output=True, text=True, check=True)
        rows = [line for line in result.stdout.splitlines()[1:]
                if len(line.split(None, 3)) >= 3
                and int(line.split(None, 3)[2]) in groups]
        save(out, 'final.cpu.json', rows)


def finish(out, filename, result, processes, groups=()):
    cleanup(processes, result)
    result['process_groups'] = [proc.pid for proc in processes] + list(groups)
    result['end_epoch'] = time.time()
    try:
        final_snapshot(out, result['process_groups'])
    except Exception as error:
        result.setdefault('cleanup_errors', []).append(repr(error))
        result.setdefault('cleanup_diagnostics', []).append(
            cleanup_diagnostic(error, operation='final_snapshot'))
    save(out, filename, result)


def failed(result):
    return int('error' in result or bool(result.get('cleanup_errors')))
