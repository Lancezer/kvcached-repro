"""CPU-only failure injection for cleanup evidence and legacy behavior."""
import io
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import process
from batch import cli, ownership, resources


class CleanupDiagnosticsTests(unittest.TestCase):
    def test_batch_signal_error_records_target_and_continues_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manager = resources.Resources({'cleanup_seconds': 1}, {}, path)
            client, server = Mock(pid=101), Mock(pid=102)
            client.poll.return_value = server.poll.return_value = None
            manager.processes = [client, server]
            manager.records = [
                dict(pid=101, session=101),
                dict(pid=102, session=102, uid=os.getuid(), role='server'),
            ]
            manager.handles = [Mock()]
            manager.allocations = [object()]
            member = dict(pid=103, session=102, uid=os.getuid())
            pidfd_os = Mock(wraps=os)
            pidfd_os.pidfd_open.return_value = 42
            pidfd_os.close.return_value = None
            with patch.object(resources, 'session_members', return_value=[]), \
                 patch.object(ownership, 'session_members', return_value=[member]), \
                 patch.object(ownership, 'matches', side_effect=lambda r: r['pid'] != 101), \
                 patch.object(ownership, 'boot', return_value='current'), \
                 patch.dict(manager.records[0], boot='old'), \
                 patch.object(ownership, 'os', pidfd_os), \
                 patch.object(ownership.signal, 'pidfd_send_signal',
                              side_effect=PermissionError('signal denied')), \
                 patch.object(resources, 'cleanup_ipc') as cleanup_ipc:
                with self.assertRaisesRegex(ownership.SharedFault, '^signal denied$'):
                    manager.close()

            cleanup = json.loads((path / 'cleanup.json').read_text())
            self.assertFalse(cleanup['verified'])
            self.assertEqual(cleanup['errors'], ['signal denied'])
            diagnostic, = cleanup['diagnostics']
            self.assertIn('PermissionError: signal denied', diagnostic.pop('traceback'))
            self.assertEqual(diagnostic, dict(operation='pidfd_send_signal', role='server',
                                             pid=103, sid=102, signal=signal.SIGTERM, flags=0))
            pidfd_os.close.assert_called_once_with(42)
            server.wait.assert_not_called()
            client.wait.assert_called_once_with(timeout=1)
            manager.handles[0].close.assert_called_once_with()
            cleanup_ipc.assert_not_called()

    def test_ipc_open_failure_emits_flags_and_propagates_original_error(self):
        from batch import ipc

        args = Mock(action='batch-ipc-clean', registry=Path('/unused/resources.json'))
        error = PermissionError('open denied')
        with patch('batch.storage.read', return_value={'records': []}), \
             patch.object(ipc, 'clean', side_effect=lambda *_: ipc.remove_control({'path': '/unused'}, None)), \
             patch.object(ipc.os, 'open', side_effect=error), \
             patch('sys.stderr', new_callable=io.StringIO) as stderr:
            with self.assertRaises(PermissionError) as caught:
                cli.offline_command(args)

        self.assertIs(caught.exception, error)
        diagnostic = json.loads(stderr.getvalue())['cleanup_diagnostic']
        self.assertEqual(diagnostic['operation'], 'open_control')
        self.assertEqual(diagnostic['flags'], os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        self.assertIn('PermissionError: open denied', diagnostic['traceback'])

    def test_group_cleanup_preserves_esrch_and_error_status(self):
        proc = Mock(pid=81)
        result = {'client_pid': proc.pid}
        with patch.object(process.os, 'killpg', side_effect=ProcessLookupError) as kill, \
             patch.object(process.time, 'sleep') as sleep:
            process.cleanup([proc], result)
        kill.assert_called_once_with(proc.pid, signal.SIGTERM)
        sleep.assert_not_called()
        proc.wait.assert_called_once_with(timeout=10)
        self.assertEqual(process.failed(result), 0)
        self.assertNotIn('cleanup_diagnostics', result)

        with patch.object(process.os, 'killpg', side_effect=PermissionError('group denied')):
            process.cleanup([proc], result)
        self.assertEqual(result['cleanup_errors'], ["PermissionError('group denied')"])
        self.assertEqual(process.failed(result), 1)
        diagnostic, = result['cleanup_diagnostics']
        self.assertIn('PermissionError: group denied', diagnostic.pop('traceback'))
        self.assertEqual(diagnostic, dict(operation='killpg', role='client', pid=81,
                                         sid=81, signal=signal.SIGTERM))

    def test_descendant_failures_identify_descendant_and_keep_leader_role(self):
        for operation in ('matches', 'terminate'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                manager = resources.Resources({'cleanup_seconds': 1}, {}, Path(directory))
                leader = Mock(pid=201)
                leader.poll.return_value = 0
                manager.processes = [leader]
                manager.records = [dict(pid=201, session=201, role='server')]
                member = dict(pid=202, session=201)
                manager.descendants = [member]
                error = OSError('descendant cleanup denied')
                with patch.object(resources, 'matches', return_value=True) as match, \
                     patch.object(resources, 'terminate') as terminate, \
                     patch.object(resources, 'session_members') as scan:
                    (match if operation == 'matches' else terminate).side_effect = error
                    with self.assertRaisesRegex(ownership.SharedFault, '^descendant cleanup denied$'):
                        manager.close()
                match.assert_called_once_with(member)
                if operation == 'matches':
                    terminate.assert_not_called()
                else:
                    terminate.assert_called_once_with(member, 1)
                leader.wait.assert_not_called()
                scan.assert_not_called()
                cleanup = json.loads((manager.directory / 'cleanup.json').read_text())
                self.assertEqual(cleanup['errors'], ['descendant cleanup denied'])
                self.assertFalse(cleanup['verified'])
                diagnostic, = cleanup['diagnostics']
                self.assertIn('OSError: descendant cleanup denied', diagnostic.pop('traceback'))
                self.assertEqual(diagnostic, dict(operation=operation, pid=202, sid=201, role='server'))

    def test_socket_unlink_failure_identifies_operation_and_candidate_paths(self):
        from batch import ipc

        path = '/unused/worker.sock'
        cleanup = Mock()
        worker = Mock()
        worker.for_paths.return_value = cleanup
        modules = {'ofd_compat': Mock(), 'kvcached': Mock(),
                   'kvcached.utils': Mock(WorkerSocketCleanup=worker)}
        registry = dict(allocations=[{}])
        for error in (None, PermissionError('unlink denied')):
            with self.subTest(error=error), patch.dict('sys.modules', modules), \
                 patch.object(ipc, 'paths', return_value=[Path(path)]), \
                 patch.object(ipc, 'remembered_objects', return_value=[{'path': path}]), \
                 patch.object(ipc, 'remove_objects', return_value=[path]), \
                 patch.object(ipc, 'same'), patch.object(ipc, 'remove_socket_root') as rmdir:
                cleanup.unlink.side_effect = error
                cleanup.unlink.return_value = False
                expected = PermissionError if error else ValueError
                with self.assertRaises(expected) as caught:
                    ipc.clean(registry, Path('/unused'))
                rmdir.assert_not_called()
                self.assertEqual(caught.exception.cleanup_context,
                                 dict(operation='unlink_worker_sockets', paths=[path]))
                if error:
                    self.assertIs(caught.exception, error)
                else:
                    self.assertEqual(str(caught.exception), 'IPC socket仍被服务或归属不明')

    def test_socket_root_failure_identifies_path_and_propagates_original_error(self):
        from batch import ipc

        root = Path('/unused/socket-root')
        error = OSError('directory not empty')
        with patch.object(ipc, 'existing_socket_root', return_value=root), \
             patch.object(Path, 'rmdir', side_effect=error):
            with self.assertRaises(OSError) as caught:
                ipc.remove_socket_root({})
        self.assertIs(caught.exception, error)
        self.assertEqual(error.cleanup_context, dict(operation='remove_socket_root', path=str(root)))
