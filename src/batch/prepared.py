"""仅原生prepare_case调用的冻结边界；此文件复制到独立客户端。"""
import hashlib
import json
import os
import shutil
from pathlib import Path

FILES = ('workload.json', 'prompts.jsonl', 'trace.jsonl', 'warmup.jsonl')
INPUTS = dict(prompts='prompts.jsonl', requests='trace.jsonl', warmup='warmup.jsonl')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def case_conditions(case):
    value = case.model_dump(mode='json', exclude_none=True)
    for key in ('id', 'description', 'serving_metadata_path', 'module'):
        value.pop(key, None)
    for target in value['targets']:
        target.pop('base_url', None)
        for key in ('model_metadata_path',):
            if key in target:
                target[key] = str(Path(target[key]).resolve(strict=True))
        prompts = target['prompts']
        if prompts['kind'] == 'jsonl':
            prompts['path'] = sha(Path(prompts['path']))
    return value


def contract(case):
    value = json.loads(os.environ['KVREPRO_PREPARED_CONTRACT'])
    if set(value) != {'pair_id', 'conditions'}:
        raise ValueError('冻结来源合同字段不合法')
    return dict(contract=value, case=case_conditions(case))


def validate_files(directory):
    if directory.is_symlink() or directory.resolve() != directory.absolute():
        raise ValueError('冻结来源不能经过符号链接')
    for name in FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'冻结文件缺失或链接: {name}')
    metadata = json.loads((directory / 'workload.json').read_text())
    if metadata.get('inputs') != INPUTS:
        raise ValueError('冻结输入必须为case内四类规范文件')
    return {name: sha(directory / name) for name in FILES}


def load_source(case, loader):
    source = os.environ.get('KVREPRO_PREPARED_SOURCE')
    if not source:
        return None
    directory = Path(source)
    files = validate_files(directory)
    if (directory / 'kvrepro-frozen.json').is_symlink():
        raise ValueError('冻结合同不能为链接')
    frozen = json.loads((directory / 'kvrepro-frozen.json').read_text())
    if frozen['identity'] != contract(case):
        raise ValueError('prepared来源pair/模型顺序/seed/负载/生成条件不匹配')
    if frozen['files'] != files:
        raise ValueError('prepared来源文件SHA256不匹配')
    workload = loader.load(directory, case=case)
    if frozen['content'] != content_hashes(workload):
        raise ValueError('prepared原生内容摘要不匹配')
    return workload


def content_hashes(workload):
    return {name: getattr(workload, name) for name in
            ('prompt_sha256', 'trace_sha256', 'warmup_sha256')}


def materialize(directory, case, workload, loader):
    source = Path(os.environ['KVREPRO_PREPARED_SOURCE'])
    for name in FILES:
        with (source / name).open('rb') as reader, (directory / name).open('xb') as writer:
            shutil.copyfileobj(reader, writer)
    if validate_files(directory) != validate_files(source):
        raise ValueError('物化后文件身份不一致')
    if content_hashes(loader.load(directory, case=case)) != content_hashes(workload):
        raise ValueError('物化后原生内容身份不一致')


def freeze(directory, case, workload):
    value = dict(identity=contract(case), files=validate_files(directory),
                 content=content_hashes(workload))
    with (directory / 'kvrepro-frozen.json').open('x') as output:
        json.dump(value, output, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
