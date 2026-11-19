"""启动失败前的 IPC 创建记录及探针错误证据的 CPU 回归测试。"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from batch import ipc, probe
from batch.ipc_journal import CreationJournal
from batch.resources import Resources, cleanup_ipc
from batch.storage import read


class LocalFixTests(unittest.TestCase):
    logical_devices = {'0': 'GPU-fixture'}

    def test_topology_strips_sgr_and_retains_links(self):
        table = '\x1b[4mGPU0 GPU1 GPU2\x1b[0m\nGPU0 X NV4 SYS\nGPU1 NV4 X PHB\nGPU2 SYS PHB X\n'
        indices = {'GPU0': 'a', 'GPU1': 'b', 'GPU2': 'c'}
        with patch.object(probe.subprocess, 'run', return_value=SimpleNamespace(stdout=table)):
            links = probe.topology(dict(nvidia_smi='unused', probe_seconds=1), indices, {'a', 'b'})
        self.assertEqual(links, [dict(source_uuid='a', destination_uuid='b', link='NV4'),
                                 dict(source_uuid='b', destination_uuid='a', link='NV4')])
        with patch.object(probe.subprocess, 'run', return_value=SimpleNamespace(stdout=table.replace('[4m', '[4G'))):
            with self.assertRaisesRegex(ValueError, '拓扑'):
                probe.topology(dict(nvidia_smi='unused', probe_seconds=1), indices, {'a', 'b'})

    def test_kv_failure_records_available_streams(self):
        cases = [
            ("import sys; print('partial'); print('detail', file=sys.stderr); sys.exit(3)", 2),
            ("import sys, time; print('partial', flush=True); "
             "print('detail', file=sys.stderr, flush=True); time.sleep(5)", 1),
            ("import sys; print('partial'); print('detail', file=sys.stderr)", 2),
            ("import sys; print('{}'); print('detail', file=sys.stderr)", 2),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for script, seconds in cases:
                config = dict(kv_probe=[sys.executable, '-c', script], probe_seconds=seconds, mps_control='/unused/mps',
                              nvidia_smi='/unused/smi')
                Resources(config, {}, directory).sample()
            rows = [json.loads(line) for line in (directory / 'kv.jsonl').read_text().splitlines()]
        for index, row in enumerate(rows):
            self.assertTrue(row['error'])
            self.assertEqual(row['stderr'], 'detail\n')
            self.assertEqual(row['stdout'], '{}\n' if index == 3 else 'partial\n')

    def test_startup_failure_cleans_from_child_journal_without_rank_probe(self):
        code = '''
import ctypes, fcntl, mmap, os, socket, struct, time
from pathlib import Path
from batch.ipc_journal import install
from batch.ipc import socket_root
from kvcached import control_segment
from ofd_compat import enable
enable()
install()
name = os.environ['KVCACHED_IPC_NAME']
path = Path('/dev/shm') / name
reserved = path.stat()
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.open(os.fsencode(path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o666)
if fd < 0:
    raise OSError(ctypes.get_errno(), str(path))
assert os.path.samestat(reserved, os.fstat(fd))
control_segment.lock(fd)
fcntl.flock(fd, fcntl.LOCK_EX)
assert control_segment.is_current(fd, path)
os.ftruncate(fd, control_segment.MARKER_OFFSET)
with mmap.mmap(fd, control_segment.MARKER_OFFSET) as payload:
    payload[:] = struct.pack('=qqq', 4096, 0, 0)
    control_segment.publish(fd)
fcntl.flock(fd, fcntl.LOCK_UN)
with socket.socket(socket.AF_UNIX) as worker:
    worker.bind(str(socket_root(name) / 'w0.sock'))
    worker.listen()
    time.sleep(0.1)
raise SystemExit(2)
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = dict(cleanup_seconds=2, startup_seconds=2, server_python=sys.executable, logical_devices=self.logical_devices)
            resources = Resources(config, {}, directory)
            model = dict(id='fixture', port=1, placement=[0])
            env = resources.service_env(model)
            allocation = resources.allocations[0]
            self.addCleanup(self.remove_fixture_ipc, allocation)
            self.addCleanup(self.close_handles, resources)
            control = Path('/dev/shm') / allocation['ipc_name']
            [reserved] = ipc.journal_records(directory, allocation['ipc_name'])
            self.assertEqual(reserved, dict(ipc.evidence(control, control.lstat()), reserved=True))
            env['PYTHONPATH'] = self.cpu_kvcached(directory)
            proc = resources.launch([sys.executable, '-c', code], env, 'fixture')
            self.assertEqual(proc.wait(timeout=2), 2)
            with self.assertRaisesRegex(RuntimeError, '确定死亡'):
                resources.wait_ready(proc, model)
            self.assertEqual(read(directory / 'resources.json')['ipc_evidence'], [])
            self.assertEqual(len(ipc.paths(allocation)), 2)
            self.assertEqual(control.read_bytes()[24:], b'KVCOFD01')
            self.assertTrue(ipc.same(reserved))
            with patch.dict(os.environ, {'PYTHONPATH': env['PYTHONPATH']}):
                resources.close()
            self.assertTrue(read(directory / 'cleanup.json')['verified'])
            self.assertEqual(ipc.paths(allocation), [])

    def test_missing_executable_clears_launch_intent_and_releases_ipc(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = dict(cleanup_seconds=2, server_python=sys.executable, logical_devices=self.logical_devices)
            resources = Resources(config, {}, directory)
            env = resources.service_env(dict(id='fixture', port=1, placement=[0]))
            allocation = resources.allocations[0]
            self.addCleanup(self.remove_fixture_ipc, allocation)
            self.addCleanup(self.close_handles, resources)
            root = ipc.socket_root(allocation['ipc_name'])
            self.assertTrue(root.is_dir())
            self.assertEqual(len(ipc.paths(allocation)), 1)
            with self.assertRaises(FileNotFoundError):
                resources.launch([str(directory / 'missing-executable')], env, 'fixture')
            self.assertIsNone(read(directory / 'resources.json')['intent'])
            with patch.dict(os.environ, {'PYTHONPATH': self.cpu_kvcached(directory)}):
                resources.close()
            self.assertTrue(read(directory / 'cleanup.json')['verified'])
            self.assertEqual(ipc.paths(allocation), [])
            self.assertFalse(root.exists())

    def test_reserved_inode_with_unknown_payload_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = dict(cleanup_seconds=2, server_python=sys.executable, logical_devices=self.logical_devices)
            resources = Resources(config, {}, directory)
            name = resources.service_env(dict(id='fixture', port=1, placement=[0]))['KVCACHED_IPC_NAME']
            self.addCleanup(self.remove_fixture_ipc, resources.allocations[0])
            control = Path('/dev/shm') / name
            control.write_bytes(b'unknown')
            with patch.dict(os.environ, {'PYTHONPATH': self.cpu_kvcached(directory)}):
                with self.assertRaises(subprocess.CalledProcessError):
                    cleanup_ipc(config, directory / 'resources.json')
            result = read(next(directory.glob('ipc-cleanup-*.json')))
            self.assertIn('未知control协议', result['stderr'])
            self.assertEqual(control.read_bytes(), b'unknown')

    @staticmethod
    def close_handles(resources):
        for handle in resources.handles:
            handle.close()

    @staticmethod
    def remove_fixture_ipc(allocation):
        for path in ipc.paths(allocation):
            path.unlink(missing_ok=True)
        ipc.remove_socket_root(allocation)

    @staticmethod
    def cpu_kvcached(directory):
        repo = Path(__file__).resolve().parents[1]
        package = directory / 'kvcached'
        package.mkdir()
        native = repo / 'external/kvcached/kvcached'
        # 只加载固定版本的纯 Python IPC 模块，绕过包初始化及 CUDA 引擎导入。
        (package / '__init__.py').write_text(f'__path__ = [{str(native)!r}]\n')
        return os.pathsep.join(map(str, (directory, repo / 'src')))

    def test_listening_socket_is_preserved_even_with_creation_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = dict(cleanup_seconds=2, server_python=sys.executable, logical_devices=self.logical_devices)
            resources = Resources(config, {}, directory)
            env = resources.service_env(dict(id='fixture', port=1, placement=[0]))
            allocation = resources.allocations[0]
            self.addCleanup(self.remove_fixture_ipc, allocation)
            resources.persist()
            journal = CreationJournal(Path(env['KVREPRO_IPC_JOURNAL']), env['KVCACHED_IPC_NAME'])
            path = ipc.socket_root(allocation['ipc_name']) / 'w0.sock'
            with socket.socket(socket.AF_UNIX) as worker, \
                    patch.dict(os.environ, {'PYTHONPATH': self.cpu_kvcached(directory)}):
                journal.bind(worker, str(path))
                worker.listen()
                with self.assertRaises(subprocess.CalledProcessError):
                    cleanup_ipc(config, directory / 'resources.json')
                self.assertTrue(path.is_socket())
                worker.close()
                cleanup_ipc(config, directory / 'resources.json')
            self.assertFalse(path.exists())
            self.assertFalse((Path('/dev/shm') / allocation['ipc_name']).exists())

    def test_unregistered_control_is_preserved_and_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            resources = Resources(dict(logical_devices=self.logical_devices), {}, directory)
            env = resources.service_env(dict(id='fixture', port=1, placement=[0]))
            path = Path('/dev/shm') / (env['KVCACHED_IPC_NAME'] + '_g1')
            self.addCleanup(self.remove_fixture_ipc, resources.allocations[0])
            resources.persist()
            with path.open('xb') as stream:
                stream.write(b'foreign')
            journal = CreationJournal(Path(env['KVREPRO_IPC_JOURNAL']), env['KVCACHED_IPC_NAME'])
            with self.assertRaises(FileExistsError):
                journal.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            config = dict(cleanup_seconds=2, server_python=sys.executable)
            with patch.dict(os.environ, {'PYTHONPATH': self.cpu_kvcached(directory)}):
                with self.assertRaises(subprocess.CalledProcessError):
                    cleanup_ipc(config, directory / 'resources.json')
            result = read(next(directory.glob('ipc-cleanup-*.json')))
            self.assertIn('未登记', result['stderr'])
            self.assertEqual(path.read_bytes(), b'foreign')

