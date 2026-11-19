"""仅批次索引的原子写和单执行器锁，不替代原生checkpoint。"""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from localpaths import allowed


def write(path, value):
    path = allowed(path)
    with tempfile.NamedTemporaryFile(mode='w', prefix=path.name + '.tmp-', dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read(path):
    return json.loads(allowed(path).read_text())


@contextmanager
def lock(directory):
    path = directory / 'batch.lock'
    if path.is_symlink():
        raise ValueError('拒绝链接锁')
    with path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
