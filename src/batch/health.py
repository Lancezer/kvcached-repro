"""固定回环健康检查；直接读取有界响应头，共享截止且不启动后台任务。"""
import socket
import time


def timeout(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('健康检查截止')
    return remaining


def healthy(port, deadline):
    deadline = min(deadline, time.monotonic() + 2)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout(deadline))
        connection.connect(('127.0.0.1', port))
        connection.settimeout(timeout(deadline))
        connection.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
        headers = bytearray()
        while b'\r\n\r\n' not in headers:
            connection.settimeout(timeout(deadline))
            part = connection.recv(4096)
            # 健康检查只需状态头，限制异常服务的头部大小。
            if not part or len(headers) + len(part) > 65536:
                return False
            headers.extend(part)
        status = headers.split(b'\r\n', 1)[0].split()
        return (time.monotonic() < deadline and len(status) >= 2
                and status[0] in (b'HTTP/1.0', b'HTTP/1.1') and status[1] == b'200')
