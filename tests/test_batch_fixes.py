"""冻结资产、恢复与数据来源的 CPU 边界测试。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from batch import config, runner
from batch.environment import snapshot
from batch.ownership import SharedFault
from batch.report import arrival_counts, outcome_summary, seed_range, occupancy, cross_seed
from batch.patch import verify_prepared_files
from data.sharegpt import candidates, join_fragments


def request(**values):
    fields = dict(outcome='success', error_kind=None, http_started_at_seconds=1., ended_at_seconds=2.)
    fields.update(values)
    return SimpleNamespace(**fields, model_dump=lambda: fields)


class StaticFixTests(unittest.TestCase):
    def test_completed_attempt_uses_original_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / 'attempt'
            attempt.mkdir()
            (attempt / 'measurement.json').write_text(json.dumps({'wall_clock_origin_seconds': 2.}))
            (attempt / 'repetition.json').write_text(json.dumps({'window_end_seconds': 3.}))
            for start, end in ((1, 10), (20, 30)):
                lifecycle = root / ('lifecycle-' + str(start * 10**9))
                lifecycle.mkdir()
                (lifecycle / 'execution.json').write_text(json.dumps(dict(started_epoch=start, finished_epoch=end, cleanup=True)))
                (lifecycle / 'client-command.json').write_text(json.dumps(['--case', 'case']))
            saved = runner.attempt_execution(root, root / 'run', {'case_id': 'case'}, attempt)
            self.assertEqual(Path(saved['lifecycle']).name, 'lifecycle-1000000000')

    def test_freeze_rejects_arbitrary_patch_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'kvrepro-prepared.patch').write_text('arbitrary')
            with self.assertRaisesRegex(ValueError, 'patch'):
                verify_prepared_files(root, 'original\n', 'patched\n')

    def test_unknown_can_be_reaudited_without_rebinding_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            prepared = run / 'cases/c'
            prepared.mkdir(parents=True)
            (prepared / 'kvrepro-frozen.json').write_text('{}')
            original = dict(path='attempt', external='original', research={'valid': None})
            audited = dict(original, research={'valid': True})
            index = dict(cases={'c': {'attempts': [original]}}, pairs={})
            with patch.object(runner, 'read', return_value={}):
                runner.publish_attempts(index, {'case_id': 'c', 'pair_id': 'pair'}, run,
                                        [audited], {'cleanup': True}, {})
            self.assertEqual(index['cases']['c']['adopted'], 'attempt')
            self.assertEqual(index['cases']['c']['attempts'][0]['external'], 'original')

    def test_local_exit_continues_only_after_shared_rechecks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            run.mkdir()
            (root / 'configuration.json').write_text('{}')
            entry = {'cleanup': True}
            with patch.object(runner, 'verify_frozen') as frozen, patch.object(runner, 'shared_preflight') as shared, \
                    patch.object(runner, 'recheck_devices') as devices:
                runner.classify_native_fault({}, entry, {'infrastructure_error': 'benchmark worker exited with code 9'}, run)
            frozen.assert_called_once()
            shared.assert_called_once()
            devices.assert_called_once()
            self.assertEqual(entry['native_fault_scope'], 'local_worker_exit_after_verified_cleanup')

    def test_repeated_messages_use_publisher_offset(self):
        pair = [('human', 'q'), ('gpt', 'a')]
        self.assertEqual(join_fragments([(0, pair * 2), (2, pair * 2)], 0), pair * 3)

    def test_exclusion_keeps_original_fragment_mapping(self):
        pair = [('human', 'q'), ('gpt', 'a')]
        excluded = []
        pool, reasons = candidates({'c': [(0, pair * 2, 'c_0'), (1, pair * 3, 'c_1')]}, 0, excluded)
        self.assertEqual(pool, {})
        self.assertEqual(reasons['消息偏移内容冲突'], 1)
        self.assertEqual(excluded[0]['fragments'][1]['fragment_id'], 'c_1')

    def test_unique_overlap_keeps_offsets_and_source_positions(self):
        first = [('human', 'q'), ('gpt', 'a')]
        following = [('gpt', 'a'), ('human', 'r'), ('gpt', 'b')]
        mapping = []
        self.assertEqual(join_fragments([(0, first, 'c_0'), (1, following, 'c_1')], 0, mapping), first + following[1:])
        self.assertEqual(mapping[1]['merged_offset'], 1)
        self.assertEqual(mapping[1]['source_message_positions'], [0, 1, 2])

    def test_capacity_environment_distinguishes_unset_empty_changed(self):
        with patch.dict('os.environ', {}, clear=True):
            before = snapshot()
            with patch.dict('os.environ', {'KVCACHED_MAX_CACHED_BLOCKS': ''}):
                empty = snapshot()
            with patch.dict('os.environ', {'KVCACHED_MAX_CACHED_BLOCKS': '2000'}):
                changed = snapshot()
        self.assertNotEqual(before, empty)
        self.assertNotEqual(empty, changed)

    def test_output_limit_freeze_rejects_every_environment_transition(self):
        self.assert_environment_transitions('SGLANG_MAX_NEW_TOKENS_LIMIT', ('', '1024', '128'))

    def test_memory_balance_freeze_rejects_every_environment_transition(self):
        self.assert_environment_transitions('SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK', ('', '0', '1'))

    def assert_environment_transitions(self, key, values):
        states = ({}, *({key: value} for value in values))
        for initial in states:
            with patch.dict('os.environ', initial, clear=True):
                frozen = dict(values={'process_environment': snapshot()}, assets={}, patch={},
                              code={})
            for current in states:
                with self.subTest(initial=initial, current=current), \
                        patch.dict('os.environ', current, clear=True), \
                        patch.object(runner, 'verify_assets'):
                    if initial == current:
                        runner.verify_frozen(frozen, frozen['values'])
                    else:
                        with self.assertRaisesRegex(SharedFault, '服务运行环境变化'):
                            runner.verify_frozen(frozen, frozen['values'])

    def test_zero_failures_and_timeout_denominator(self):
        successful = outcome_summary([request()], 1)
        failed = outcome_summary([request(outcome='failed', error_kind='timeout')], 2)
        self.assertEqual(successful['counts']['failed'], 0)
        self.assertEqual(failed['failure_fractions']['timeout'], .5)
        self.assertEqual(failed['counts']['unknown'], 1)
        self.assertEqual(seed_range([0, 1, None])['seeds'], 2)

    def test_arrival_rates_use_left_closed_right_open_window(self):
        rows = [request(), request(http_started_at_seconds=10., ended_at_seconds=12.)]
        summary = arrival_counts(rows, 10., 2)
        self.assertEqual(summary['counts']['sent'], 1)
        self.assertEqual(summary['rates']['planned'], .2)
        self.assertEqual(summary['sent_fraction'], .5)

    def test_kv_known_missing_sample_breaks_integral(self):
        values = [(0., {'physical_bytes': 2}), (1., {}), (2., {'physical_bytes': 4})]
        self.assertEqual(occupancy(values, 2., 3.)['physical_bytes']['observed_span'], 0)
        self.assertEqual(occupancy([values[0], values[2]], 2., 3., [1.])['physical_bytes']['observed_span'], 0)

    def test_cross_seed_keeps_zero_and_missing_seed(self):
        manifest = {'cases': [dict(case_id=str(i), conditions={'rate': .2}, mps=False) for i in range(3)]}
        reports = {str(i): dict(models={'m': {'outcomes': {'failed': i}}}, native={'targets': {}},
                   kv={'ranks': {'m:0:g': {'physical_bytes': {'peak': 10 + i}}}}) for i in range(2)}
        result = cross_seed(manifest, reports)['0.2:False']['metrics']
        self.assertEqual(result['models.m.outcomes.failed']['median'], .5)
        self.assertEqual(result['kv.m:0:g.physical_bytes.peak']['missing'], 1)

    def test_unclassified_native_fault_remains_blocked(self):
        with self.assertRaises(SharedFault), patch.object(runner, 'shared_preflight') as checking:
            runner.classify_native_fault({}, {'cleanup': True}, {'infrastructure_error': 'disk broken'}, Path('/unused'))
        checking.assert_not_called()

    def test_local_worker_fault_requires_cleanup(self):
        with self.assertRaises(SharedFault):
            runner.classify_native_fault({}, {'cleanup': False},
                                         {'infrastructure_error': 'benchmark worker exited with code 9'}, Path('/unused'))

    def test_probe_same_path_content_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = root / 'probe'
            program.write_text('original')
            manifest = root / 'assets.json'
            manifest.write_text(json.dumps({'probe': config.file_hash(program)}))
            values = dict(probe_assets_manifest=str(manifest), rank_probe=[str(program)], kv_probe=[str(program)])
            config.probe_assets(values)
            program.write_text('changed')
            with self.assertRaisesRegex(ValueError, '摘要不符'):
                config.probe_assets(values)


if __name__ == '__main__':
    unittest.main()
