"""批次截止时的信号延后、取消等待和外层超时命令。"""
from functools import wraps
import os
import signal
import subprocess

CANCEL_SECONDS = 10
DISPATCH_KILL_AFTER_SECONDS = 210


def dispatch_timeout(seconds, command):
    return ['timeout', '--foreground', '--signal=TERM',
            f'--kill-after={DISPATCH_KILL_AFTER_SECONDS}s', str(seconds), *command]


def defer_term(operation):
    @wraps(operation)
    def guarded(*args, **kwargs):
        previous = signal.getsignal(signal.SIGTERM)
        pending = None

        def remember(signum, frame):
            nonlocal pending
            pending = signum, frame

        signal.signal(signal.SIGTERM, remember)
        try:
            result = operation(*args, **kwargs)
        except BaseException:
            signal.signal(signal.SIGTERM, previous)
            if pending is not None:
                deliver(previous, pending)
            raise
        signal.signal(signal.SIGTERM, previous)
        if pending is not None:
            deliver(previous, pending)
        return result

    return guarded


def deliver(previous, pending):
    if callable(previous):
        previous(*pending)
    elif previous == signal.SIG_DFL:
        os.kill(os.getpid(), signal.SIGTERM)


def bound_cancel(proc):
    original = proc.wait
    expired = False

    def wait(timeout=None):
        nonlocal expired
        if timeout is not None:
            return original(timeout=timeout)
        if not expired:
            try:
                return original(timeout=CANCEL_SECONDS)
            except subprocess.TimeoutExpired:
                expired = True
        return proc.poll()

    proc.wait = wait


def install(runner, resources):
    original = resources.launch

    @defer_term
    def launch(self, args, env, name):
        proc = original(self, args, env, name)
        if name == 'client':
            bound_cancel(proc)
        return proc

    resources.launch = launch
    resources.reserve_ipc = defer_term(resources.reserve_ipc)
    runner.finish_resources = defer_term(runner.finish_resources)
