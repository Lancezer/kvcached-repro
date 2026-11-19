"""固定普通V3来源；下载是显式独立命令，永不在batch隐式发生。"""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
from localpaths import allowed

REPOSITORY = 'anon8231489123/ShareGPT_Vicuna_unfiltered'
REVISION = '192ab2185289094fc556ec8ce5ce1e8e587154ca'
FILENAME = 'ShareGPT_V3_unfiltered_cleaned_split.json'
SHA256 = '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
URL = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{FILENAME}'


def checksum(path):
    result = hashlib.sha256()
    with allowed(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(path):
    actual = checksum(path)
    if actual != SHA256:
        raise ValueError(f'普通V3本地SHA256不匹配: {actual}')
    return dict(repository=REPOSITORY, revision=REVISION, filename=FILENAME,
                expected_sha256=SHA256, actual_sha256=actual)


def download(destination):
    destination = allowed(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('下载目标必须不存在')
    temporary = destination.with_name(destination.name + '.partial')
    with temporary.open('xb') as output:
        with urlopen(URL, timeout=60) as response:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
        output.flush()
        os.fsync(output.fileno())
    provenance = verify(temporary)
    os.link(temporary, destination)  # 不覆盖并发创建的目标；失败的partial保留供人工审计。
    temporary.unlink()
    with destination.with_name(destination.name + '.source.json').open('x') as output:
        json.dump(provenance, output, indent=2)
    return provenance
