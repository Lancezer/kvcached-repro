"""批次配置、配对、资源与研究统计的 CPU 边界测试。"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from batch import config, native, prepared, runner
from batch.audit import adopt, kv_bytes, outcome, weighted_quantile
from batch.ownership import SharedFault, matches
from batch.patch import patched_runner
from data.sharegpt import join_fragments, prompt, valid_pairs, select


def configuration():
    return dict(rates=[.2, .5, .8, 1.], seeds=[11, 12, 13], mps_states=[False, True],
                duration=600., bucket_seconds=1., warmup=10, output_tokens=1024,
                sampling=dict(temperature=0., ignore_eos=False), max_inflight=128,
                models=[dict(id='local/test', tp=1, dp=1, port=31000)],
                startup_seconds=1800.)


class MatrixTests(unittest.TestCase):
    def test_patch_only_changes_preparation_asset_block(self):
        head = 'class Runner:\n    def prepare_case(self, case, program):\n'
        body = ('        workload = prepare_workload(case, config=config)\n'
                '        directory = self.run_entry.directory / "cases" / str(case.id)\n'
                '        directory.mkdir(parents=True)\n'
                '        write_json(\n            directory / "workload.json", {})\n')
        tail = '        manifest = BenchCaseManifest(case=case)\n        return manifest\n'
        changed = patched_runner(head + body + tail)
        self.assertTrue(changed.startswith(head))
        self.assertTrue(changed.endswith(tail))
        self.assertIn('materialize(directory, case, workload, PreparedWorkload)', changed)
        self.assertIn('freeze(directory, case, workload)', changed)

    def test_order_and_identity(self):
        value = configuration()
        rows = config.matrix(value, {'input': 'a'})
        self.assertEqual(len(rows), 24)
        self.assertEqual([r['mps'] for r in rows[:6]], [False, True, True, False, False, True])
        for first, second in zip(rows[::2], rows[1::2]):
            self.assertEqual(first['pair_id'], second['pair_id'])
            self.assertNotEqual(first['case_id'], second['case_id'])
        changed = copy.deepcopy(value)
        changed['models'][0]['port'] += 1
        other = config.matrix(changed, {'input': 'a'})
        self.assertEqual(rows[0]['pair_id'], other[0]['pair_id'])
        self.assertEqual(rows[0]['fingerprint'], other[0]['fingerprint'])

    def test_single_native_case_and_no_deadline(self):
        value, row = configuration(), config.matrix(configuration(), {})[0]
        args = native.command(dict(client_python='/python'), row, Path('/out'), Path('/catalog'))
        self.assertIn('--case', args)
        self.assertNotIn('--all', args)
        self.assertEqual(args[args.index('--repetitions') + 1], '1')
        continued = native.command(dict(client_python='/python'), row, Path('/out'), continuing=Path('/run'))
        self.assertEqual(continued[-3:], ['--all', '--continue', '/run'])
        self.assertNotIn('--rerun-failure', continued)

    def test_environment_not_mutated(self):
        value = dict(source='/independent')
        row = dict(pair_id='pair', conditions={})
        for inherited in ('', 'GPU-a,GPU-b'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': inherited}, clear=True):
                env = native.environment(value, row)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], inherited)
                self.assertEqual(dict(os.environ), {'CUDA_VISIBLE_DEVICES': inherited})


class PreparedCase:
    def __init__(self, identity='case-a', models=('local/a', 'local/b')):
        self.identity, self.models = identity, models

    def model_dump(self, **kwargs):
        return dict(id=self.identity, description='state', module='serving.multi_model', seed=7,
                    arrivals=dict(kind='poisson', rates={m: .2 for m in self.models}),
                    targets=[dict(model_id=m, base_url='http://localhost', prompts=dict(kind='random')) for m in self.models])


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source, self.destination = self.directory / 'source', self.directory / 'destination'
        self.source.mkdir()
        self.destination.mkdir()
        for name in prepared.FILES:
            (self.source / name).write_text('{}\n')
        (self.source / 'workload.json').write_text(json.dumps(dict(inputs=prepared.INPUTS)))
        self.workload = type('Workload', (), dict(prompt_sha256='p', trace_sha256='t', warmup_sha256='w'))()
        self.environment = patch.dict(os.environ, {'KVREPRO_PREPARED_CONTRACT': json.dumps(dict(pair_id='pair', conditions={}))})
        self.environment.start()
        prepared.freeze(self.source, PreparedCase(), self.workload)
        self.loader = type('Loader', (), {'load': lambda _, directory, case: self.workload})()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_rebind_and_materialize_bytes(self):
        with patch.dict(os.environ, {'KVREPRO_PREPARED_SOURCE': str(self.source)}):
            case = PreparedCase('case-b')
            loaded = prepared.load_source(case, self.loader)
            prepared.materialize(self.destination, case, loaded, self.loader)
        self.assertEqual(prepared.validate_files(self.source), prepared.validate_files(self.destination))
        self.assertFalse((self.destination / 'case.json').exists())

    def test_tamper_order_and_path_escape_rejected(self):
        with patch.dict(os.environ, {'KVREPRO_PREPARED_SOURCE': str(self.source)}):
            with self.assertRaises(ValueError):
                prepared.load_source(PreparedCase(models=('local/b', 'local/a')), self.loader)
            (self.source / 'trace.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                prepared.load_source(PreparedCase(), self.loader)
        (self.source / 'workload.json').write_text(json.dumps(dict(inputs=dict(prepared.INPUTS, prompts='../outside'))))
        with self.assertRaises(ValueError):
            prepared.validate_files(self.source)

    def test_link_rejected(self):
        (self.source / 'trace.jsonl').unlink()
        (self.source / 'trace.jsonl').symlink_to(self.directory / 'outside')
        with self.assertRaises(ValueError):
            prepared.validate_files(self.source)


class RecoveryTests(unittest.TestCase):
    def test_native_nonzero_can_be_adopted(self):
        original = dict(path='native/attempt-1', research=dict(valid=True), native_code=1)
        self.assertEqual(adopt(original), 'native/attempt-1')
        self.assertEqual(outcome(dict(outcome='failed', error_kind='evidence_missing')), 'unknown')

    def test_case_fault_continues_but_shared_blocks(self):
        value, manifest = configuration(), dict(cases=config.matrix(configuration(), {})[:3])
        index = dict(cases={}, pairs={}, invocations=[])
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runner, 'run_case', side_effect=[None, None, None]) as call:
                self.assertEqual(runner.drive(value, manifest, Path(temporary), index, (), None), 0)
                self.assertEqual(call.call_count, 3)
            with patch.object(runner, 'run_case', side_effect=SharedFault('shared')) as call:
                self.assertEqual(runner.drive(value, manifest, Path(temporary), index, (), None), 2)
                self.assertEqual(call.call_count, 1)

    def test_live_driver_and_unknown_intent_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'driver.json').write_text('{}')
            with patch.object(runner, 'matches', return_value=True):
                with self.assertRaises(SharedFault):
                    runner.resume_audit(root, False, {})

    def test_pid_reuse_refused(self):
        record = dict(pid=123, pgid=123, session=123, uid=os.getuid(),
                      host='host', boot='boot', start='old')
        with patch('batch.ownership.socket.gethostname', return_value='host'), patch('batch.ownership.boot', return_value='boot'):
            with patch('batch.ownership.identity', return_value=dict(record, start='new')):
                with self.assertRaises(SharedFault):
                    matches(record)

    def test_no_automatic_retry_and_at_most_one(self):
        row = dict(case_id='c')
        original = dict(path='one', research=dict(valid=True))
        index = dict(cases={'c': dict(attempts=[original])})
        self.assertFalse(runner.check_rerun(index, row, (), None))
        with self.assertRaises(ValueError):
            runner.check_rerun(index, row, ('c',), 'reason')
        original['research']['valid'] = False
        self.assertTrue(runner.check_rerun(index, row, ('c',), 'reason'))


class DataAndStatisticsTests(unittest.TestCase):
    def test_prefix_and_exact_template(self):
        pairs = valid_pairs([('human', ' Q\n'), ('gpt', ' A '), ('human', 'next'), ('gpt', 'ok'), ('human', '')])
        self.assertEqual(prompt(pairs, 1), 'Human:  Q\n\nAssistant:  A \nHuman: next\nAssistant:')
        self.assertEqual(join_fragments([(0, [('human', 'q'), ('gpt', 'a')]),
                         (1, [('gpt', 'a'), ('human', 'next')])], 0)[-1], ('human', 'next'))
        with self.assertRaises(ValueError):
            join_fragments([(1, [('human', 'q')])], 0)

    def test_tokenizer_intersection_and_equal_conversation_selection(self):
        tokenizer = type('Tokenizer', (), {'encode': lambda _, text: list(range(len(text)))})()
        candidates = {'a': [dict(text='short', conversation='a', turn=0)],
                      'b': [dict(text='long' * 100, conversation='b', turn=0)]}
        models = [dict(id='one', context_length=20, input_max_exclusive=20, request_total_limit=20)]
        selected, counts = select(candidates, {'one': tokenizer}, dict(data_seed=1, output_tokens=2, models=models))
        self.assertEqual([r['conversation'] for r in selected], ['a'])
        self.assertEqual(counts['no_intersection_conversation'], 1)

    def test_weighted_linear_quantiles_and_rank_geometry(self):
        self.assertEqual(weighted_quantile([(1, 2), (5, 2)], .5), 3)
        self.assertEqual(kv_bytes(2, dict(kind='mla', layers=3, element_bytes=2,
                                        kv_lora_rank=4, qk_rope_head_dim=2)), 72)
        self.assertEqual(kv_bytes(2, dict(kind='mha', layers=3, element_bytes=2,
                                        local_kv_heads=4, head_dim=2)), 192)


class FailureBoundaryTests(unittest.TestCase):
    def test_case_start_fault_is_local_and_cleanup_runs(self):
        config_value = configuration()
        row = config.matrix(config_value, {})[0]
        with tempfile.TemporaryDirectory() as temporary:
            resources = unittest.mock.Mock()
            resources.service_death = []
            resources.client_code.return_value = None
            resources.start_mps.side_effect = RuntimeError('local service startup fault')
            with patch.object(runner, 'Resources', return_value=resources):
                result = runner.execute(config_value, row, Path(temporary), None)
            self.assertTrue(result['cleanup'])
            self.assertEqual(result['fault'], 'local service startup fault')
            resources.close.assert_called_once()

    def test_native_request_failure_code_does_not_raise_case_fault(self):
        value = configuration()
        row = config.matrix(value, {})[0]
        with tempfile.TemporaryDirectory() as temporary:
            resources = unittest.mock.Mock()
            resources.service_death = []
            resources.client_code.return_value = 1
            with patch.object(runner, 'Resources', return_value=resources), patch.object(runner, 'start_resources'):
                with patch.object(runner, 'invoke', return_value=1):
                    result = runner.execute(value, row, Path(temporary), None)
            self.assertEqual(result['native_code'], 1)
            self.assertIsNone(result['fault'])

    def test_ipc_same_name_replacement_is_not_deleted(self):
        from batch.ipc import same
        import stat
        with tempfile.TemporaryDirectory() as temporary:
            path, replacement = Path(temporary) / 'owned', Path(temporary) / 'replacement'
            path.write_text('old')
            replacement.write_text('new')
            value = path.stat()
            record = dict(path=str(path), dev=value.st_dev, ino=value.st_ino,
                          uid=value.st_uid, mode=stat.S_IFREG)
            replacement.replace(path)
            with self.assertRaises(ValueError):
                same(record)
            self.assertEqual(path.read_text(), 'new')

    def test_living_client_without_progress_has_no_deadline(self):
        proc, resources = unittest.mock.Mock(), unittest.mock.Mock()
        proc.poll.return_value = None
        resources.dead.return_value = False
        resources.config = dict(poll_seconds=.1, sample_seconds=1.)
        with patch.object(runner.time, 'sleep', side_effect=InterruptedError('test harness stop')):
            with self.assertRaises(InterruptedError):
                runner.monitor(proc, resources)
        proc.send_signal.assert_not_called()
        proc.wait.assert_not_called()

    def test_confirmed_service_death_requests_native_cancel(self):
        proc, resources = unittest.mock.Mock(), unittest.mock.Mock()
        proc.poll.return_value = None
        resources.dead.return_value = True
        with self.assertRaises(RuntimeError):
            runner.monitor(proc, resources)
        proc.send_signal.assert_called_once()
        proc.wait.assert_called_once_with()

    def test_unknown_launch_gap_blocks_without_killing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lifecycle = root / 'cases/c/lifecycle-1'
            lifecycle.mkdir(parents=True)
            (lifecycle / 'resources.json').write_text(json.dumps(dict(intent=dict(owner={}), records=[])))
            with patch.object(runner, 'terminate') as terminate:
                with self.assertRaises(SharedFault):
                    runner.resume_audit(root, True, dict(cleanup_seconds=1))
                terminate.assert_not_called()

    def test_cleanup_foreign_uid_rejected_before_signal(self):
        from batch.ownership import signal_member
        with patch('batch.ownership.os.getuid', return_value=1), patch('batch.ownership.os.pidfd_open') as opening:
            with self.assertRaises(SharedFault):
                signal_member(dict(uid=2, pid=42), 15)
            opening.assert_not_called()

    def test_service_subsets_use_runtime_uuid_order_and_mps_keeps_all_devices(self):
        from batch.resources import Resources
        parent_devices = '4,2,0,3,1'
        uuids = ['GPU-e', 'GPU-c', 'GPU-a', 'GPU-d', 'GPU-b']
        models = [dict(id='first', port=31000, placement=[1, 3]),
                  dict(id='second', port=31001, placement=[0, 2, 4])]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=parent_devices), \
                patch.object(runner, 'cuda_observation', return_value=(uuids, {})), \
                patch('batch.probe.gpu_inventory', return_value=([], 'driver', {})), \
                patch('batch.probe.topology', return_value=[]), \
                patch.object(runner, 'validate_hardware'), patch.object(runner, 'hardware_identity'):
            config = dict(mps_control='/unused/mps', probe_seconds=1, nvidia_smi='/unused/smi')
            mapping = runner.observe_hardware(config)['mapping']
            resource = Resources(dict(config, logical_devices=mapping), dict(mps=True), Path(temporary))
            with (patch.object(resource, 'launch') as launch, patch.object(resource, 'wait_controller'),
                  patch.object(resource, 'mps_control'), patch.object(resource, 'wait_servers'),
                  patch.object(resource, 'reserve_ipc', return_value=('ipc', temporary))):
                resource.start_mps()
                self.assertEqual(launch.call_args.args[1]['CUDA_VISIBLE_DEVICES'], parent_devices)
                envs = [resource.service_env(model) for model in models]
                self.assertEqual([env['CUDA_VISIBLE_DEVICES'] for env in envs], ['GPU-c,GPU-d', 'GPU-e,GPU-a,GPU-b'])
                self.assertEqual(envs[0]['CUDA_MPS_PIPE_DIRECTORY'], resource.mps_env['CUDA_MPS_PIPE_DIRECTORY'])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], parent_devices)

    def test_rank_validation_uses_device_uuid_even_when_local_index_is_zero(self):
        from batch.resources import validate_ranks
        model = dict(id='model', tp=1, placement=[3], dp_attention=False,
                     geometry_by_rank={'0': {}}, max_total_tokens='auto')
        config = dict(models=[model], logical_devices={'0': 'GPU-other', '3': 'GPU-target'})
        rank = dict(model='model', rank=0, gpu='GPU-target', mps_server_pid=42,
                    prefix_cache_enabled=False, dp_attention=False, kvcached_enabled=True,
                    geometry={}, capacity_tokens=1)
        validate_ranks(config, dict(mps=True), dict(ranks=[rank]), [42])
        for wrong_gpu in ('GPU-other', '0'):
            with self.subTest(gpu=wrong_gpu), self.assertRaisesRegex(SharedFault, 'rank设备映射不符'):
                validate_ranks(config, dict(mps=True), dict(ranks=[dict(rank, gpu=wrong_gpu)]), [42])


if __name__ == '__main__':
    unittest.main()
