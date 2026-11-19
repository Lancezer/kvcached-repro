"""用真实慢HTTP检查服务就绪超时；不加载模型或使用GPU。"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest
from types import SimpleNamespace
from batch.resources import Resources


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.server.segmented:
            self.segmented_headers(self.server.delay)
        else:
            time.sleep(self.server.delay)
            self.send_response(200)
            self.end_headers()

    def segmented_headers(self, interval):
        parts = (b'HTTP/1.1 200 OK\r\n', b'X-One: 1\r\n', b'X-Two: 2\r\n',
                 b'Content-Length: 0\r\n', b'\r\n')
        try:
            for part in parts:
                time.sleep(interval)
                self.wfile.write(part)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@contextmanager
def health_server(delay, segmented=False):
    server = ThreadingHTTPServer(('127.0.0.1', 0), HealthHandler)
    server.delay, server.segmented = delay, segmented
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    worker.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class ReadyTests(unittest.TestCase):
    def test_health_slower_than_one_second_can_be_ready(self):
        with health_server(1.05) as port:
            resources = Resources({'startup_seconds': 3}, {}, None)
            start = time.monotonic()
            resources.wait_ready(SimpleNamespace(poll=lambda: None), {'port': port})
            elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 1)
        self.assertLess(elapsed, 2.5)

    def test_single_health_check_respects_remaining_startup_budget(self):
        with health_server(.6) as port:
            resources = Resources({'startup_seconds': .1}, {}, None)
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, '就绪预算耗尽'):
                resources.wait_ready(SimpleNamespace(poll=lambda: None), {'port': port})
            elapsed = time.monotonic() - start
        self.assertLess(elapsed, .4)

    def test_segmented_headers_share_absolute_deadline(self):
        with health_server(.06, segmented=True) as port:
            resources = Resources({'startup_seconds': .1}, {}, None)
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, '就绪预算耗尽'):
                resources.wait_ready(SimpleNamespace(poll=lambda: None), {'port': port})
            elapsed = time.monotonic() - start
        self.assertLess(elapsed, .25)
