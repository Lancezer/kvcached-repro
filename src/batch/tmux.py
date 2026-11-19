"""只构造可审阅的tmux命令；工具不自动开启会话或真实批次。"""
import re
import shlex
import sys
from pathlib import Path
from localpaths import allowed


def command(output, session):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session):
        raise ValueError('tmux会话名仅允许字母数字下划线与连字符')
    output = allowed(output).resolve(strict=True)
    entrypoint = Path(__file__).resolve().parents[1] / 'main.py'
    driver = shlex.join([sys.executable, '-u', str(entrypoint), 'batch-run', '--out', str(output)])
    shell = driver + ' >> ' + shlex.quote(str(output / 'runner.log')) + ' 2>&1'
    return shlex.join(['tmux', 'new-session', '-d', '-s', session, shell])
