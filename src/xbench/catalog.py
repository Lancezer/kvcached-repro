"""为单服务 smoke 生成独立 catalog，保留用户原始配置和流量文件。"""
import json
from pathlib import Path
import re
import shutil
from runtime import base_url


def replace_value(text, key, value):
    pattern = rf'(?m)^(\s*{key}\s*=\s*)[^\n]+'
    updated, count = re.subn(pattern, lambda match: match[1] + json.dumps(value), text)
    if count == 0:
        raise ValueError(f'单服务 catalog 缺少 {key}')
    return updated


def workload(source, destination, config, model_id):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    for row in rows:
        row.update(model_id=model_id, max_new_tokens=config.max_new_tokens)
    destination.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def runtime_catalog(root, out, config):
    source = Path(config.catalog).resolve() if config.catalog else root / 'config/catalog.toml'
    text = source.read_text()
    model_ids = re.findall(r'(?m)^model_id\s*=\s*"([^"]+)"', text)
    if len(model_ids) != 1:
        raise ValueError('smoke catalog 必须只有一个服务目标')
    destination = out / 'config'
    destination.mkdir()
    shutil.copytree(source.parent / 'suites', destination / 'suites',
                    ignore=shutil.ignore_patterns('.env', '*.pem', '__pycache__'))
    model_id = config.model_id or config.model
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id) is None:
        raise ValueError('本地模型路径请用 --model-id 指定 namespace/name')
    text = configure_catalog(text, config, model_id)
    text = relocate_paths(text, source.parent, destination, config, model_id)
    catalog = destination / 'catalog.toml'
    catalog.write_text(text)
    return catalog


def configure_catalog(text, config, model_id):
    for key, value in {'model_id': model_id, 'base_url': base_url(config),
                       'max_inflight': config.max_inflight,
                       'request_timeout_seconds': config.request_timeout}.items():
        text = replace_value(text, key, value)
    return text


def relocate_paths(text, parent, destination, config, model_id):
    # catalog 路径按原文件位置解析，流量副本才改模型 ID 和生成长度。
    def relocated(match):
        source = (parent / json.loads(match[2])).resolve()
        return match[1] + ' = ' + json.dumps(str(source))
    keys = r'\b(path|model_metadata_path|serving_metadata_path)'
    text = re.sub(keys + r'\s*=\s*("(?:[^"\\]|\\.)*")', relocated, text)
    pattern = r'(\[serving_cases\.[^\n]+\.arrivals\]\s*kind\s*=\s*"jsonl"\s*path\s*=\s*)("[^"\n]+")'
    match = re.search(pattern, text)
    if not match:
        raise ValueError('smoke 需要 jsonl arrivals')
    arrivals = destination / 'arrivals.jsonl'
    workload(Path(json.loads(match[2])), arrivals, config, model_id)
    return text[:match.start(2)] + json.dumps(str(arrivals)) + text[match.end(2):]
