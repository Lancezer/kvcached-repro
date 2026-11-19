"""在服务进程中登记实际创建的 IPC，覆盖 rank 探针前的退出。"""
import fcntl
import hashlib
import os
import socket
import stat
from contextlib import contextmanager
from pathlib import Path

from .ipc import evidence, is_control, same, socket_root
from .storage import read, write


class CreationJournal:
    def __init__(self, directory, name):
        self.directory, self.name = directory, name
        self.root = socket_root(name)
        self.open_file, self.bind_socket = os.open, socket.socket.bind

    @contextmanager
    def entry(self, path):
        # 多个 rank 可同时打开同一 control；创建与身份落盘在同一把锁内。
        descriptor = self.open_file(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            key = hashlib.sha256(os.fsencode(path)).hexdigest()
            yield self.directory / (key + '.json')
        finally:
            os.close(descriptor)

    def open(self, file, flags, mode=0o777, *, dir_fd=None):
        if not isinstance(file, (str, Path)) or not flags & os.O_CREAT:
            return self.open_file(file, flags, mode, dir_fd=dir_fd)
        path = Path(file)
        if not is_control(path, self.name):
            return self.open_file(file, flags, mode, dir_fd=dir_fd)
        if flags & os.O_TRUNC:
            raise ValueError('不允许在核验前截断control')
        with self.entry(path) as entry:
            return self.open_control(path, flags | os.O_NOFOLLOW, mode, entry)

    def open_control(self, path, flags, mode, entry):
        try:
            descriptor = self.open_file(path, flags | os.O_EXCL, mode)
        except FileExistsError:
            if flags & os.O_EXCL or not entry.is_file():
                raise
            return self.reopen_control(path, flags & ~os.O_CREAT, mode, read(entry))
        try:
            write(entry, evidence(path, os.fstat(descriptor)))
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def reserve_control(self):
        # 原生 tracker 会打开同一 inode，并在取得锁后扩容及写入协议标记。
        path = Path('/dev/shm') / self.name
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        with self.entry(path) as entry:
            descriptor = self.open_file(path, flags, 0o600)
            record = None
            try:
                record = evidence(path, os.fstat(descriptor))
                write(entry, dict(record, reserved=True))
            except BaseException:
                if record is not None and same(record):
                    path.unlink()
                raise
            finally:
                os.close(descriptor)

    def reopen_control(self, path, flags, mode, record):
        descriptor = self.open_file(path, flags, mode)
        if not evidence(path, os.fstat(descriptor)).items() <= record.items():
            os.close(descriptor)
            raise ValueError('control对象与创建记录不符')
        return descriptor

    def bind(self, sock, address):
        if sock.family != socket.AF_UNIX or not isinstance(address, str):
            return self.bind_socket(sock, address)
        path = Path(address)
        if not path.is_relative_to(self.root) or path.suffix != '.sock':
            return self.bind_socket(sock, address)
        with self.entry(path) as entry:
            root = read(self.directory / 'socket-root')
            if root['path'] != str(self.root) or not same(root):
                raise ValueError('IPC socket目录身份变化')
            if path.resolve() != path:
                raise ValueError('IPC socket路径经过符号链接')
            self.bind_socket(sock, address)
            record = evidence(path, path.lstat())
            if record['mode'] != stat.S_IFSOCK or record['uid'] != os.getuid() or not same(record):
                raise ValueError('无法登记新绑定的 IPC socket')
            write(entry, record)


def install():
    directory = os.environ.get('KVREPRO_IPC_JOURNAL')
    if not directory:
        return
    journal = CreationJournal(Path(directory), os.environ['KVCACHED_IPC_NAME'])
    os.open = journal.open

    def bind(sock, address):
        return journal.bind(sock, address)

    socket.socket.bind = bind
