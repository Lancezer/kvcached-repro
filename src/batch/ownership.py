"""Linux进程身份与私有会话审计；旧PID不能作为清理授权。"""
import hashlib
import os
import signal
import socket
import time
from pathlib import Path
from cleanup_diagnostics import cleanup_operation


class SharedFault(RuntimeError):
    """无法证明共享前提或清理安全时阻塞整个批次。"""


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def identity(pid):
    directory = Path('/proc') / str(pid)
    try:
        uid = directory.stat().st_uid
        if uid != os.getuid():
            raise SharedFault('进程UID不属于本用户，禁止读取命令或清理')
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = (directory / 'cmdline').read_bytes()
        return dict(pid=pid, pgid=int(fields[2]), session=int(fields[3]), start=fields[19],
                    command_sha256=hashlib.sha256(command).hexdigest(), uid=uid,
                    host=socket.gethostname(), boot=boot())
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(record):
    if record['host'] != socket.gethostname():
        raise SharedFault('记录属于其他主机，禁止操作')
    if record['boot'] != boot():
        return False
    current = identity(record['pid'])
    if current is None:
        return False
    keys = ('pid', 'pgid', 'session', 'start', 'uid', 'host', 'boot')
    # exec/setproctitle会改cmdline；同boot/start/uid/session仍是同一进程。
    if any(current[key] != record[key] for key in keys):
        raise SharedFault('PID或会话身份变化，禁止操作')
    return True


def session_members(session):
    records = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            record = session_member(path, session)
            if record:
                records.append(record)
    return records


def session_member(path, session):
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return identity(int(path.name)) if int(fields[3]) == session else None
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def audit(records):
    alive = [record for record in records if matches(record)]
    sessions = {record['session'] for record in alive}
    for record in records:
        if record['boot'] == boot() and record['session'] not in sessions and session_members(record['session']):
            raise SharedFault('已登记成员均消失但session仍有进程，归属未知')
    return alive


def signal_member(record, signum):
    with cleanup_operation('signal_member', pid=record['pid'], sid=record.get('session'),
                           signal=int(signum)):
        send_member_signal(record, signum)


def send_member_signal(record, signum):
    if record['uid'] != os.getuid():
        raise SharedFault('不是本用户进程')
    try:
        with cleanup_operation('pidfd_open', flags=0):
            descriptor = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        return
    try:
        with cleanup_operation('matches'):
            current = matches(record)
        if current:
            with cleanup_operation('pidfd_send_signal', flags=0):
                signal.pidfd_send_signal(descriptor, signum)
    finally:
        with cleanup_operation('close_pidfd'):
            os.close(descriptor)


def drain_session(record, signum, grace):
    members = session_members(record['session'])
    for member in members:
        signal_member(member, signum)
    until = time.monotonic() + grace
    while time.monotonic() < until:
        remaining = session_members(record['session'])
        if not remaining:
            return []
        time.sleep(0.1)
    return remaining


def terminate(record, grace):
    if not matches(record):
        if record['boot'] == boot() and session_members(record['session']):
            raise SharedFault('已登记进程消失且session仍活，禁止凭旧PID清理')
        return
    if record['uid'] != os.getuid():
        raise SharedFault('不是本用户独立运行域')
    remaining = drain_session(record, signal.SIGTERM, grace)
    if remaining:
        # KILL只针对TERM后仍持有的具体身份；pidfd避免信号落在复用PID。
        for member in remaining:
            signal_member(member, signal.SIGKILL)
        until = time.monotonic() + grace
        while session_members(record['session']) and time.monotonic() < until:
            time.sleep(0.1)
    if session_members(record['session']):
        raise SharedFault('自有session未清空')
