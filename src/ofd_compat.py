"""补齐 Linux LP64 Python 缺失的 OFD 常量；先用私有临时 inode 核验内核。"""
import ctypes
import errno
import fcntl
import os
import platform
import sys
import tempfile


class Flock(ctypes.Structure):
    _fields_ = [('type', ctypes.c_short), ('whence', ctypes.c_short),
                ('start', ctypes.c_long), ('length', ctypes.c_long), ('pid', ctypes.c_int)]


def probe(path, descriptor):
    request = bytes(Flock(fcntl.F_RDLCK, os.SEEK_SET, 0, 1, 0))
    fcntl.fcntl(descriptor, 37, request)
    other = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
        exclusive = bytes(Flock(fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0))
        observed = Flock.from_buffer_copy(fcntl.fcntl(other, 36, exclusive))
        if observed.type != fcntl.F_RDLCK or observed.pid != -1:
            raise OSError(errno.ENOTSUP, '内核未返回预期OFD持有者')
        try:
            fcntl.fcntl(other, 37, exclusive)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EACCES):
                raise
        else:
            raise OSError(errno.ENOTSUP, '独立文件描述必须与活OFD锁冲突')
    finally:
        os.close(other)


def enable():
    names = {'F_OFD_GETLK': 36, 'F_OFD_SETLK': 37}
    if all(hasattr(fcntl, name) for name in names):
        return
    if (not sys.platform.startswith('linux') or platform.machine() not in ('x86_64', 'aarch64')
            or ctypes.sizeof(ctypes.c_long) != 8 or ctypes.sizeof(Flock) != 32):
        raise OSError(errno.ENOTSUP, '只允许已核实的Linux LP64 OFD ABI')
    if any(getattr(fcntl, name, value) != value for name, value in names.items()):
        raise OSError(errno.ENOTSUP, 'OFD常量与Linux ABI冲突')
    # 永不探测现有服务IPC；失败发生在任何常量发布之前。
    with tempfile.NamedTemporaryFile(prefix='kvrepro-ofd-', dir='/tmp') as handle:
        probe(handle.name, handle.fileno())
    for name, value in names.items():
        setattr(fcntl, name, value)
