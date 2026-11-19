"""真实V3消息偏移回归及缺失常量的真实Linux锁验证。"""
import errno
import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from data.sharegpt import candidates, join_fragments
from ofd_compat import enable


class RealDataTests(unittest.TestCase):
    def test_nonconsecutive_offsets_and_adjacent_fragments(self):
        values = [('human' if i % 2 == 0 else 'gpt', str(i)) for i in range(22)]
        mapping = []
        restored = join_fragments([(0, values[:12], 'x_0'), (11, values[11:18], 'x_11'),
                                  (18, values[18:], 'x_18')], 0, mapping)
        self.assertEqual(restored, values)
        self.assertEqual([r['merged_offset'] for r in mapping], [0, 11, 18])
        self.assertTrue(all(r['basis'] == 'publisher_message_offset' for r in mapping))

    def test_gaps_conflicts_and_duplicate_offsets(self):
        with self.assertRaisesRegex(ValueError, '缺口'):
            join_fragments([(0, [('human', 'a')]), (2, [('gpt', 'b')])], 0)
        with self.assertRaisesRegex(ValueError, '片段冲突'):
            join_fragments([(0, [('human', 'a')]), (0, [('human', 'b')])], 0)
        self.assertEqual(join_fragments([(0, []), (0, [])], 0), [])

    def test_missing_constants_use_kernel_checked_abi(self):
        # 删除字典项模拟目标3.12，不改变磁盘或任何共享Python安装。
        original = {n: getattr(fcntl, n) for n in ('F_OFD_GETLK', 'F_OFD_SETLK') if hasattr(fcntl, n)}
        try:
            for name in original:
                delattr(fcntl, name)
            enable()
            self.assertEqual(fcntl.F_OFD_GETLK, 36)
            self.assertEqual(fcntl.F_OFD_SETLK, 37)
        finally:
            for name in ('F_OFD_GETLK', 'F_OFD_SETLK'):
                if name in original:
                    setattr(fcntl, name, original[name])
                elif hasattr(fcntl, name):
                    delattr(fcntl, name)

    def test_unsupported_abi_does_not_publish_constants(self):
        with patch('ofd_compat.platform.machine', return_value='unknown'), patch('ofd_compat.hasattr', return_value=False, create=True):
            with self.assertRaises(OSError) as caught:
                enable()
        self.assertEqual(caught.exception.errno, errno.ENOTSUP)


class ExclusionTests(unittest.TestCase):
    def test_empty_and_invalid_suffix_have_fragment_evidence(self):
        groups = {'empty': [(0, [], 'empty_0')],
                  'suffix': [(0, [('human', 'q'), ('gpt', 'a'), ('system', 'bad')], 'suffix_0')]}
        excluded = []
        pool, counts = candidates(groups, 0, excluded)
        self.assertEqual(set(pool), {'suffix'})
        self.assertEqual(counts['no_complete_pair'], 1)
        self.assertEqual(counts['discarded_suffix_messages'], 1)
        reasons = {r['reason']: r for r in excluded}
        self.assertEqual(reasons['no_complete_pair']['fragments'][0]['fragment_id'], 'empty_0')
        self.assertEqual(reasons['discarded_suffix_messages']['retained_pairs'], 1)
