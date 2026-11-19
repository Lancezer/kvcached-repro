"""自有IPC身份登记；删除复用固定kvcached的锁/活socket协议。"""
import hashlib
import os
import stat
import uuid
from pathlib import Path
from cleanup_diagnostics import cleanup_operation
from localpaths import allowed
from .storage import read


def allocation(model):
    name = 'kvrepro_' + uuid.uuid4().hex[:24]
    return dict(model=model['id'], port=model['port'], ipc_name=name, log_name=model_log_name(model['id']))


def model_log_name(model_id):
    return 'model-' + hashlib.sha256(model_id.encode()).hexdigest()[:16]


def socket_root(name):
    if not name.startswith('kvrepro_') or len(name) != 32 or not name[8:].isalnum():
        raise ValueError('不是本工具的私有IPC名称')
    return Path('/tmp') / ('kvcached-tp-' + name + '-' + uuid.uuid5(uuid.NAMESPACE_DNS, name).hex[:8])


def is_control(path, name):
    return path.parent == Path('/dev/shm') and (path.name == name or path.name.startswith(name + '_g'))


def paths(allocation):
    name = allocation['ipc_name']
    root = socket_root(name)
    controls = [p for p in Path('/dev/shm').glob(name + '*') if is_control(p, name)]
    sockets = list(root.rglob('*.sock')) if root.exists() and not root.is_symlink() else []
    if root.is_symlink():
        raise ValueError('私有IPC socket目录是符号链接')
    return controls + sockets


def evidence(path, value):
    return dict(path=str(path), dev=value.st_dev, ino=value.st_ino,
                uid=value.st_uid, mode=stat.S_IFMT(value.st_mode))


def journal_path(directory, name):
    return directory / ('ipc-' + name)


def remembered_objects(registry, directory):
    remembered = {r['path']: r for r in registry['ipc_evidence']}
    for allocation in registry['allocations']:
        for record in journal_records(directory, allocation['ipc_name']):
            if record['path'] in remembered and remembered[record['path']] != record:
                raise ValueError('IPC身份记录冲突')
            remembered[record['path']] = record
    return list(remembered.values())


def journal_records(directory, name):
    private = journal_path(directory, name)
    try:
        value = private.lstat()
    except FileNotFoundError:
        return  # 兼容创建记录上线前的 registry。
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or value.st_mode & 0o077:
        raise ValueError('IPC身份记录目录不属于本用户私有域')
    for entry in private.glob('*.json'):
        value = entry.lstat()
        if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or value.st_mode & 0o077:
            raise ValueError('IPC身份记录不属于本用户私有域')
        record = read(entry)
        path = Path(record['path'])
        if not is_control(path, name) and not (path.is_relative_to(socket_root(name)) and path.suffix == '.sock'):
            raise ValueError('IPC身份记录不属于本次分配')
        yield record


def same(record):
    try:
        value = Path(record['path']).lstat()
    except FileNotFoundError:
        return False
    if (value.st_dev, value.st_ino, value.st_uid, stat.S_IFMT(value.st_mode)) != (record['dev'], record['ino'], record['uid'], record['mode']):
        raise ValueError('IPC同名对象已替换，禁止删除')
    if value.st_uid != os.getuid():
        raise ValueError('IPC对象不属于本用户')
    return True


def clean(registry, directory):
    from ofd_compat import enable
    enable()
    from kvcached import control_segment
    from kvcached.utils import WorkerSocketCleanup
    allocated = registry['allocations']
    current = {str(p) for a in allocated for p in paths(a)}
    remembered = remembered_objects(registry, directory)
    if current - {r['path'] for r in remembered}:
        raise ValueError('存在未登记IPC对象，保留并阻塞')
    sockets = remove_objects(remembered, current, control_segment)
    cleanup = WorkerSocketCleanup.for_paths(sockets)
    for record in remembered:
        if record['path'] in sockets:
            same(record)  # native记住inode后再次核对，防止构造器前发生同名替换。
    with cleanup_operation('unlink_worker_sockets', paths=sockets):
        if not cleanup.unlink():
            raise ValueError('IPC socket仍被服务或归属不明')
    for allocation in allocated:
        remove_socket_root(allocation)


def existing_socket_root(allocation):
    record = allocation.get('socket_root')
    if record is None:
        return None  # 旧 registry 只有 socket 本身的身份记录。
    root = socket_root(allocation['ipc_name'])
    if record['path'] != str(root) or record['mode'] != stat.S_IFDIR:
        raise ValueError('IPC socket目录记录不符')
    return root if same(record) else None


def remove_socket_root(allocation):
    root = existing_socket_root(allocation)
    if root is not None:
        with cleanup_operation('remove_socket_root', path=str(root)):
            root.rmdir()  # 非空目录保留并阻塞，禁止递归删除。


def remove_objects(remembered, current, protocol):
    sockets = []
    for record in remembered:
        allowed(record['path'])
        if record['path'] not in current:
            if Path(record['path']).exists():
                raise ValueError('登记路径不在本次私有IPC域')
            continue
        if not same(record):
            continue
        if record['mode'] == stat.S_IFREG:
            remove_control(record, protocol)
        elif record['mode'] == stat.S_IFSOCK:
            sockets.append(record['path'])
        else:
            raise ValueError('未知IPC对象类型')
    return sockets


def remove_control(record, protocol):
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    with cleanup_operation('open_control', path=record['path'], flags=flags):
        descriptor = os.open(record['path'], flags)
    with cleanup_operation('remove_control', path=record['path']):
        unlink_control(record, protocol, descriptor)


def unlink_control(record, protocol, descriptor):
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_uid) != (record['dev'], record['ino'], record['uid']):
            raise ValueError('control inode已替换')
        with cleanup_operation('lock_control'):
            protocol.lock(descriptor, exclusive=True)
        if not protocol.is_current(descriptor, record['path']):
            raise ValueError('control路径在取得OFD锁时变化')
        # 启动可能在 tracker 打开之前退出，空的独占预留 inode 也可回收。
        empty_reservation = record.get('reserved') is True and os.fstat(descriptor).st_size == 0
        if not empty_reservation and not protocol.known(descriptor):
            raise ValueError('未知control协议，禁止删除')
        with cleanup_operation('unlink_control'):
            os.unlink(record['path'])
    finally:
        with cleanup_operation('close_control'):
            os.close(descriptor)
