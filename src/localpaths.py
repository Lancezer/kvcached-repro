"""显式本地路径的禁区保护，不扫描或读取凭据。"""
import os
from pathlib import Path


def reject(path):
    blocked = (Path('/home/incoming'), Path.home() / '.ssh')
    if any(path == root or path.is_relative_to(root) for root in blocked):
        raise ValueError('路径在禁止访问区域')
    if path == Path.home() / '.aws/credentials' or '.env' in path.parts or path.suffix.lower() == '.pem':
        raise ValueError('禁止读取凭据或环境文件')


def allowed(path, preserve_symlinks=False):
    path = Path(os.path.abspath(os.path.expanduser(path)))
    reject(path)  # 先做词法拒绝，避免resolve访问显式禁区。
    resolved = path.resolve()
    reject(resolved)
    return path if preserve_symlinks else resolved
