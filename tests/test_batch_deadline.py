"""批次截止和信号边界测试。"""
import signal
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from batch.deadline import CANCEL_SECONDS, bound_cancel, defer_term, dispatch_timeout, install
from batch.window import Window


class DeadlineTests(unittest.TestCase):
    def test_dispatch_timeout_has_bounded_final_wait(self):
        command = dispatch_timeout(30.5, ['python', 'dispatch.py'])
        self.assertEqual(command[1:5], [
            '--foreground', '--signal=TERM', '--kill-after=210s', '30.5'
        ])
        self.assertEqual(command[5:], ['python', 'dispatch.py'])

    def test_cancel_timeout_defers_to_registered_cleanup(self):
        wait = Mock(side_effect=[subprocess.TimeoutExpired('xbench', CANCEL_SECONDS), 0])
        proc = SimpleNamespace(wait=wait, poll=lambda: None)
        bound_cancel(proc)
        self.assertIsNone(proc.wait())
        self.assertIsNone(proc.wait())
        wait.assert_called_once_with(timeout=CANCEL_SECONDS)
        self.assertEqual(proc.wait(timeout=10), 0)

    def test_term_is_replayed_after_registration(self):
        events = []

        class Resources:
            def launch(self, args, env, name):
                events.append('popen')
                signal.raise_signal(signal.SIGTERM)
                events.append('registered')
                return object()

            def reserve_ipc(self):
                pass

        def stop(*_):
            events.append('term')
            raise InterruptedError

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            install(SimpleNamespace(finish_resources=lambda: None), Resources)
            with self.assertRaises(InterruptedError):
                Resources().launch([], {}, 'server')
            self.assertEqual(events, ['popen', 'registered', 'term'])
            self.assertIs(signal.getsignal(signal.SIGTERM), stop)
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_term_is_replayed_when_registration_fails(self):
        events = []

        def operation():
            signal.raise_signal(signal.SIGTERM)
            raise OSError('registration failed')

        def stop(*_):
            events.append('term')
            raise InterruptedError

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            with self.assertRaises(InterruptedError):
                defer_term(operation)()
            self.assertEqual(events, ['term'])
            self.assertIs(signal.getsignal(signal.SIGTERM), stop)
        finally:
            signal.signal(signal.SIGTERM, previous)


class WindowTests(unittest.TestCase):
    def test_admits_pair_only_when_time_budget_fits(self):
        now = [500]
        window = Window(1000, 400, 100, lambda: now[0])
        index = {'cases': {}}
        row = dict(pair_id='pair-1', case_id='7')
        window.admit(row, index)
        now[0] = 501
        window.admit(dict(row, case_id='8'), index)
        with self.assertRaises(InterruptedError):
            window.admit(dict(row, pair_id='pair-2', case_id='9'), index)
        now[0] = 900
        with self.assertRaises(InterruptedError):
            window.admit(row, index)
        window.admit(row, {'cases': {'7': {'adopted': '/old'}}})


if __name__ == '__main__':
    unittest.main()
