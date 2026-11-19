"""CPU 替身验证；不启动服务，不查询真实 GPU。"""
import ast
import contextlib
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import zipapp
from unittest.mock import MagicMock, patch

import os
import runtime
import process
import protocol
from preflight import check, dependencies, symbols
from service import settings, validation
from xbench import client, prepare, smoke

ROOT = Path(__file__).resolve().parents[1]


def make_payload(directory):
    source = Path(directory) / 'payload-src'
    shutil.copytree(ROOT / 'src', source,
                    ignore=shutil.ignore_patterns('__pycache__'))
    payload = Path(directory) / 'tools.pyz'
    zipapp.create_archive(source, payload, main='main:entrypoint')
    return payload


def stream():
    meta = dict(finish_reason={'type': 'stop'}, prompt_tokens=5, completion_tokens=16)
    return ('data: ' + json.dumps({'meta_info': meta}) + '\n\ndata: [DONE]\n').encode()


def fake_response():
    conn = MagicMock()
    conn.getresponse.return_value.status = 200
    conn.getresponse.return_value.getheaders.return_value = []
    conn.getresponse.return_value.read.return_value = stream()
    return conn


def runtime_patches(stack):
    for module in (validation, smoke):
        stack.enter_context(patch.object(module, 'check_port'))
        stack.enter_context(patch.object(module, 'wait_ready'))
    stack.enter_context(patch.object(process, 'final_snapshot'))
    stack.enter_context(patch.object(protocol.http.client, 'HTTPConnection', return_value=fake_response()))
    return stack.enter_context(patch.object(validation, 'stop_group'))


class ToolsTests(unittest.TestCase):
    def test_settings(self):
        for visible in (None, '', '2', '3,1', 'GPU-first,GPU-second', 'MIG-example'):
            with self.subTest(visible=visible), patch.dict(os.environ, {}, clear=True):
                if visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = visible
                before = dict(os.environ)
                env = settings.server_environment(True, 'test')
                self.assertEqual(env.get('CUDA_VISIBLE_DEVICES'), visible)
                self.assertEqual(settings.server_command()[settings.server_command().index('--tp-size') + 1], '1')
                explicit = runtime.configuration()
                explicit.devices, explicit.tp_size = 'GPU-other,MIG-other', 2
                env = settings.server_environment(False, 'test', explicit)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], explicit.devices)
                self.assertIn('2', settings.server_command(explicit))
                self.assertEqual(dict(os.environ), before)

    def test_sse(self):
        self.assertEqual(protocol.validate_sse(stream())['completion_tokens'], 16)
        malformed = (b'data: [DONE]\n', stream().split(b'data: [DONE]')[0],
                     stream() + b'data: [DONE]\n', b'data: {}\n\ndata: [DONE]\n')
        for raw in malformed:
            with self.assertRaises((ValueError, KeyError)):
                protocol.validate_sse(raw)

    def test_health(self):
        proc = MagicMock()
        proc.poll.return_value = 2
        with self.assertRaises(RuntimeError):
            protocol.wait_ready(proc)
        proc.poll.return_value = None
        with patch.object(protocol, 'healthy', return_value=True):
            protocol.wait_ready(proc)
        with self.assertRaises(TimeoutError):
            protocol.wait_ready(proc, timeout=0)

    def test_http_closes(self):
        conn = fake_response()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(protocol.http.client, 'HTTPConnection', return_value=conn):
                protocol.generate(Path(directory), 'on', {})
                conn.getresponse.return_value.status = 500
                with self.assertRaises(RuntimeError):
                    protocol.generate(Path(directory), 'off', {})
        self.assertEqual(conn.close.call_count, 2)

    def test_cleanup_exited_launcher(self):
        proc = MagicMock(pid=12345)
        proc.poll.return_value = 0
        with patch.object(process.os, 'killpg') as kill, patch.object(process.time, 'sleep'):
            process.stop_group(proc)
        self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (12345, signal.SIGKILL))
        proc.wait.assert_called_once_with(timeout=10)

    def test_finish_failure_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            result = {}
            with patch.object(process, 'stop_group', side_effect=OSError('cleanup')):
                with patch.object(process, 'final_snapshot', side_effect=OSError('snapshot')):
                    process.finish(Path(directory), 'result.json', result, [MagicMock(pid=1)])
            self.assertEqual(process.failed(result), 1)
            self.assertEqual(len(result['cleanup_errors']), 2)
            self.assertTrue((Path(directory) / 'result.json').exists())

    def run_smoke_case(self, directory, outcome):
        with contextlib.ExitStack() as stack:
            runtime_patches(stack)
            server, worker = MagicMock(pid=123), MagicMock(pid=124)
            worker.wait.side_effect = outcome if isinstance(outcome, Exception) else None
            worker.wait.return_value = outcome
            launch = stack.enter_context(patch.object(smoke.subprocess, 'Popen', side_effect=[server, worker]))
            stop = stack.enter_context(patch.object(process, 'stop_group'))
            root = Path(directory)
            shutil.copytree(ROOT / 'config/xbench-smoke', root / 'config')
            (root / 'config/suites').mkdir()
            rc = smoke.main(root, root / 'out')
            self.assertEqual(rc, int(outcome != 0))
            self.assertEqual([call.args[0] for call in stop.call_args_list], [worker, server])
            self.assertEqual(launch.call_args_list[1].kwargs['cwd'], root)
            self.assertEqual(launch.call_args_list[1].kwargs['env'].get('CUDA_VISIBLE_DEVICES'),
                             os.environ.get('CUDA_VISIBLE_DEVICES'))
            result = json.loads((root / 'out/execution.json').read_text())
            self.assertEqual(result['client_command'][-1], str(root / 'out/artifacts'))

    def test_smoke_exit_and_timeout(self):
        for outcome in (0, 9, subprocess.TimeoutExpired('xbench', 120)):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                self.run_smoke_case(directory, outcome)

    def test_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = ROOT / 'external/CrossPool'
            for name in ('src/xpool/config.py', 'benches/suites/serving/multi_model.py'):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((upstream / name).read_bytes())
            payload = make_payload(directory)
            result = subprocess.run([sys.executable, '-I', '-S', str(payload),
                                     'prepare', '.', 'config'], cwd=root, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            suite = root / 'config/suites/serving/multi_model.py'
            tree = ast.parse(suite.read_text())
            self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module == 'xkit.serving.sglang.system'
                                 for n in tree.body))
            self.assertIn('        import xpool.native\n', (root / 'src/xpool/config.py').read_text())
            with self.assertRaises(ValueError):
                prepare.prepare(root, root / 'config')

    def test_payload_and_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = make_payload(directory)
            for flags in ([], ['-I', '-S']):
                result = subprocess.run([sys.executable, *flags, str(payload), '--help'],
                                        cwd='/', capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, str(payload), 'preflight'],
                                    cwd='/', capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('使用 -I -S', result.stderr)


    def test_payload_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = make_payload(directory)
            source = Path(directory) / 'payload-src'
            (source / 'preflight/check.py').write_text('def main(): return 7\n')
            zipapp.create_archive(source, payload, main='main:entrypoint')
            result = subprocess.run([sys.executable, '-I', '-S', str(payload), 'preflight'],
                                    cwd='/', capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)

    def test_source_cli(self):
        result = subprocess.run([sys.executable, str(ROOT / 'src/main.py'), '--help'],
                                cwd='/', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('usage:', result.stdout)

    def test_preflight_guard_before_site(self):
        with patch.object(check.site, 'main') as site_main:
            with self.assertRaises(SystemExit):
                check.main()
            site_main.assert_not_called()

    def test_preflight_result(self):
        result = dict(direct_dependency_issues=[], sglang_in_supported_range=True,
                      static_symbols={'method': True}, imports=dict.fromkeys(
                          ('cuda_initialized', 'sglang_imported', 'vllm_imported',
                           'sglang_autopatch_imported'), False))
        with patch.object(check, 'guard'), patch.object(check.site, 'main'):
            with patch.object(check, 'report', return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(), 0)
                result['static_symbols']['method'] = False
                self.assertEqual(check.main(), 1)

    def test_preflight_ast_and_dependencies(self):
        tree = ast.parse('class Pool:\n    def alloc(self):\n        pass\n')
        self.assertTrue(symbols.has_method(tree, 'Pool', 'alloc'))
        self.assertFalse(symbols.has_method(tree, 'Pool', 'missing'))
        with patch.object(dependencies.metadata, 'version', return_value='1.0'):
            self.assertIsNone(dependencies.requirement_issue('test', 'demo>=1'))
            self.assertIsNotNone(dependencies.requirement_issue('test', 'demo>=2'))
        with patch.object(dependencies.metadata, 'version', side_effect=dependencies.metadata.PackageNotFoundError):
            self.assertIsNone(dependencies.requirement_issue('test', 'demo; extra == "gpu"'))
            self.assertIsNone(dependencies.requirement_issue('test', 'demo')['installed'])
