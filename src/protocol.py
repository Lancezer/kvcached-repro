"""SGLang 健康检查、原生生成协议和 SSE 校验。"""
import http.client
import json
import socket
import time
from evidence import save
from urllib.parse import urlsplit
from runtime import configuration, base_url


def check_port(config=None):
    config = config or configuration()
    with socket.socket(socket.AF_INET6 if ':' in config.host else socket.AF_INET) as probe:
        probe.bind((config.host, config.port))


def healthy(config=None):
    config = config or configuration()
    url = urlsplit(base_url(config))
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
    try:
        conn.request('GET', '/health')
        response = conn.getresponse()
        response.read()
        return response.status == 200
    except OSError:
        return False
    finally:
        conn.close()


def wait_ready(proc, timeout=None, config=None):
    config = config or configuration()
    timeout = config.startup_timeout if timeout is None else timeout
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('服务启动失败，见 server.log')
        if healthy(config):
            return
        time.sleep(1)
    raise TimeoutError('服务启动超时')


def sse_events(raw):
    events, done = [], False
    for line in raw.decode().splitlines():
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if done:
            raise ValueError('DONE 后仍有 data')
        if data == '[DONE]':
            done = True
        else:
            events.append(json.loads(data))
    if not done or not events:
        raise ValueError('缺少响应或 DONE')
    return events


def validate_sse(raw):
    meta = sse_events(raw)[-1]['meta_info']
    if meta.get('finish_reason', {}).get('type') not in ('stop', 'length'):
        raise ValueError('缺少正常终态')
    if meta.get('prompt_tokens', 0) <= 0 or meta.get('completion_tokens', 0) <= 0:
        raise ValueError('缺少 token 计数')
    return meta


def generate(out, mode, body, config=None):
    config = config or configuration()
    url = urlsplit(base_url(config))
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=config.request_timeout)
    try:
        payload = dict(body, rid='kvcached-switch-' + mode)
        conn.request('POST', '/generate', json.dumps(payload),
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        save(out, mode + '.response.json',
             {'status': response.status, 'headers': response.getheaders()})
        raw = response.read()
        (out / (mode + '.sse')).write_bytes(raw)
        if response.status != 200:
            raise RuntimeError(f'{mode}: HTTP {response.status}')
        return validate_sse(raw)
    finally:
        conn.close()
