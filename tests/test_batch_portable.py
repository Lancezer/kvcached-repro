import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from batch import adoption, config, runner
from batch.ownership import SharedFault
from batch.storage import read, write


def observation(prefix):
    uuids = [f'GPU-{prefix}{i}' for i in range(5)]
    return dict(mapping={str(i): uuid for i, uuid in enumerate(uuids)},
                packages=dict(sglang='1', torch='2', kvcached='3'),
                devices=[dict(uuid=uuid, name='A100', total_memory_bytes=100) for uuid in uuids],
                driver_version='driver',
                links=[dict(source_uuid=a, destination_uuid=b, link='NV4')
                       for a in uuids for b in uuids if a != b])


class PortableBatchTests(unittest.TestCase):
    def test_explicit_path_precedence_and_empty_is_error(self):
        with patch.dict(os.environ, {'KVREPRO_SOURCE': '/environment'}):
            self.assertEqual(config.input_value('source', {'source': '/cli'}), ('/cli', 'cli'))
            self.assertEqual(config.input_value('source', {}), ('/environment', 'environment'))
            with self.assertRaisesRegex(ValueError, '非空本地路径'):
                config.local_path(config.input_value('source', {'source': ''})[0], 'source')
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'KVREPRO_SOURCE'):
            config.input_value('source', {})

    def test_missing_runtime_path_fails_before_observation_and_index(self):
        frozen = dict(values={'models': []})
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True), \
                patch.object(runner, 'batch_documents', return_value=(frozen, {})), \
                patch.object(runner, 'observe_hardware') as observe:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'KVREPRO_SOURCE'):
                runner.run(root)
            self.assertEqual(list(root.iterdir()), [])
            observe.assert_not_called()

    def test_json_paths_and_old_selection_are_rejected(self):
        for extra in ({'source': None}, {'device_map': {}}, {'models': [{'devices': None}]}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, '已移除'):
                config.reject_retired(dict({'models': []}, **extra))
        with self.assertRaisesRegex(ValueError, '绝对路径'):
            config.probe_command('["python", "/tmp/probe.py"]', 'rank_probe')

    def test_real_runtime_count_drives_gate(self):
        labels = ['GPU-' + str(i) for i in range(5)]
        options = dict(server_python='/explicit/python', probe_seconds=2)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6,5,4,3,2'}, clear=True), \
                patch.object(runner.subprocess, 'run') as process:
            for values in (labels[:4], labels + ['GPU-other'], labels[:-1] + labels[:1]):
                process.return_value = SimpleNamespace(stdout=json.dumps(dict(uuids=values, packages={})))
                with self.assertRaisesRegex(SharedFault, '实际可见五张'):
                    runner.cuda_observation(options)
            process.return_value = SimpleNamespace(stdout=json.dumps(dict(uuids=labels, packages={})))
            self.assertEqual(runner.cuda_observation(options)[0], labels)
            self.assertEqual(process.call_args.args[0][0], '/explicit/python')
        with patch.dict(os.environ, {}, clear=True), patch.object(runner.subprocess, 'run') as process:
            with self.assertRaisesRegex(SharedFault, 'CUDA_VISIBLE_DEVICES'):
                runner.cuda_observation(options)
            process.assert_not_called()

    def test_uuid_replacement_and_missing_or_changed_topology(self):
        old, current = observation('old'), observation('new')
        current['devices'].reverse()
        current['links'].reverse()
        runner.verify_hardware(old, current)
        current['links'][0]['link'] = 'PHB'
        with self.assertRaisesRegex(SharedFault, '不等价'):
            runner.verify_hardware(old, current)
        current['links'].pop()
        with self.assertRaisesRegex(SharedFault, '拓扑证据不完整'):
            runner.verify_hardware(old, current)

    def test_resume_uses_current_logical_devices_and_keeps_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = dict(cases={'kept': {'adopted': '/old/attempt'}}, pairs={}, invocations=[])
            write(root / 'index.json', index)
            old, current = observation('old'), observation('new')
            frozen = dict(hardware_observation=old)
            manifest = dict(cases=[dict(case_id='kept', pair_id='p', mps=False),
                                   dict(case_id='pending', pair_id='q', mps=True)])
            with patch.object(runner, 'batch_documents', return_value=(frozen, manifest)), \
                    patch.object(runner, 'runtime_config', return_value={}), \
                    patch.object(runner, 'verify_frozen'), patch.object(runner, 'resume_audit'), \
                    patch.object(runner, 'observe_hardware', return_value=current), \
                    patch.object(runner, 'run_case') as run_case:
                self.assertEqual(runner.run(root), 0)
            run_case.assert_called_once()
            self.assertEqual(run_case.call_args.args[0]['logical_devices'], current['mapping'])
            self.assertEqual(run_case.call_args.args[1]['case_id'], 'pending')
            self.assertEqual(read(root / 'index.json')['cases'], index['cases'])

    def test_service_rank_allocation_mismatch_rejects_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, hardware = Path(temporary), observation('new')
            model = dict(id='local/test', tp=1, placement=[1])
            rank = dict(model=model['id'], rank=0, gpu=hardware['mapping']['0'])
            write(root / 'hardware.json', hardware)
            write(root / 'metadata-ranks.json', dict(ranks=[rank]))
            write(root / 'resources.json', dict(server_pids=[]))
            frozen = dict(values={'models': [model]}, hardware_observation=hardware)
            with self.assertRaisesRegex(SharedFault, 'rank设备映射不符'):
                adoption.check_allocation({'external': str(root)}, frozen, {'mps': False}, {})

    def test_preview_adopt_idempotence_and_no_overwrite(self):
        row = dict(case_id='target', pair_id='p', mps=True)
        other = dict(row, case_id='source')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'target'
            output.mkdir()
            source = Path(temporary) / 'isolated'
            path = source / 'cases/source/attempt'
            path.mkdir(parents=True)
            index = output / 'index.json'
            write(index, dict(cases={}, pairs={}, invocations=[]))
            original = index.read_bytes()
            evidence = dict(path=str(path), research=dict(outcomes=dict(success=1)))
            with patch.object(runner, 'batch_documents', side_effect=lambda p: ({}, {'cases': [row if p == output else other]})), \
                    patch.object(runner, 'runtime_config'), patch.object(runner, 'verify_assets'), \
                    patch.object(adoption, 'compare', return_value=evidence) as compare:
                args = (output, source, 'target', 'source', path, {})
                self.assertEqual(adoption.import_attempt(*args)['status'], 'eligible')
                self.assertEqual(index.read_bytes(), original)
                self.assertFalse((output / 'batch.lock').exists())
                self.assertEqual(adoption.import_attempt(*args, commit=True)['status'], 'adopted')
                adopted = index.read_bytes()
                adoption.import_attempt(*args, commit=True)
                self.assertEqual(index.read_bytes(), adopted)
                entry = read(index)['cases']['target']
                self.assertEqual(entry['attempts'][0]['imported_from']['case_id'], 'source')
                with self.assertRaisesRegex(SharedFault, '已有采纳'):
                    adoption.import_attempt(output, source, 'target', 'source', path.parent, {}, commit=True)
                self.assertEqual(compare.call_count, 3)
                self.assertEqual(index.read_bytes(), adopted)

    def test_legacy_batch_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'configuration.json', {'values': {}, 'cuda_visible_devices': {}})
            write(root / 'manifest.json', {})
            with self.assertRaisesRegex(SharedFault, '当前格式'):
                runner.batch_documents(root)

    def test_metadata_uses_logical_rank_placement_with_native_optional_fields(self):
        old, new = observation('old'), observation('new')
        first = dict(old, target_device_uuids={'local/test': [old['mapping']['1']]})
        current = dict(new, target_device_uuids={'local/test': [new['mapping']['1']]})
        first['devices'] = [dict(d, pci_bus_id=None, cpu_affinity=None) for d in old['devices']]
        self.assertEqual(runner.metadata_identity(first, old), runner.metadata_identity(current, new))
        current['target_device_uuids']['local/test'] = [new['mapping']['0']]
        self.assertNotEqual(runner.metadata_identity(first, old), runner.metadata_identity(current, new))

    def test_workload_bytes_must_match_even_when_both_contracts_are_valid(self):
        old, new = observation('old'), observation('new')
        values = {k: 1 for k in ('process_environment', 'hardware', 'startup_seconds',
                                 'poll_seconds', 'cleanup_seconds', 'sample_seconds', 'probe_seconds')}
        target = dict(hardware_observation=old, code={}, patch={}, values=values, probe_protocol={})
        source = dict(target, hardware_observation=new)
        row = dict(case_id='target', pair_id='p', mps=True, conditions={'seed': 1})
        other = dict(row, case_id='source')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'target'
            origin = Path(temporary) / 'origin'
            pair = output / 'cases/partner/native/bench-runs/run/cases/partner'
            candidate = origin / 'cases/source/native/bench-runs/run/cases/source'
            for directory, payload in ((pair, '1'), (candidate, '2')):
                directory.mkdir(parents=True)
                write(directory / 'workload.json', {'inputs': adoption.prepared.INPUTS})
                for name in ('prompts.jsonl', 'trace.jsonl', 'warmup.jsonl'):
                    (directory / name).write_text(payload)
                files = adoption.prepared.validate_files(directory)
                write(directory / 'kvrepro-frozen.json',
                      dict(identity=dict(contract=dict(pair_id='p', conditions=row['conditions'])),
                           files=files, content={}))
            index = dict(pairs={'p': str(pair)})
            with self.assertRaisesRegex(SharedFault, 'workload 不一致'), \
                    patch.object(adoption, 'original_attempt') as original:
                adoption.compare(index, output, origin, candidate / 'attempt', row, other, target, source, {})
            original.assert_not_called()

    def test_service_uses_subset_local_start_and_stride_independently_of_tp_dp(self):
        from batch.resources import service_command
        model = dict(placement=[0, 4], tp=2, dp=2, dp_attention=True, checkpoint='/model',
                     port=31000, context_length=8192, mem_fraction_static=.5, max_total_tokens='auto')
        args = service_command({'server_python': '/python'}, model)
        self.assertEqual(args[args.index('--base-gpu-id') + 1], '0')
        self.assertEqual(args[args.index('--gpu-id-step') + 1], '1')
        self.assertEqual(args[args.index('--tp-size') + 1], '2')
        self.assertEqual(args[args.index('--dp-size') + 1], '2')

    def test_resume_requires_same_state_capacity_evidence(self):
        row = {'mps': True}
        model = dict(id='local/test', tp=1)
        observed = dict(ranks=[dict(model=model['id'], rank=0, capacity_tokens=500)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = dict(models=[model], capacity_directory=root, changed_devices=True)
            with self.assertRaisesRegex(SharedFault, '缺少目标同态'):
                runner.seal_capacity(values, row, observed)
            runner.seal_capacity(dict(values, changed_devices=False), row, observed)
            runner.seal_capacity(values, row, observed)
            observed['ranks'][0]['capacity_tokens'] = 501
            with self.assertRaisesRegex(SharedFault, '容量不等价'):
                runner.seal_capacity(values, row, observed)
            del observed['ranks'][0]['capacity_tokens']
            with self.assertRaisesRegex(SharedFault, '缺少实际 rank 容量'):
                runner.seal_capacity(values, row, observed)

    def test_probe_interpreter_requires_declared_absolute_script(self):
        import sys
        program = sys.executable
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / 'probe.py'
            script.touch()
            assets = {str(Path(program).resolve()): 'python', str(script): 'script'}
            for arg in ('probe.py', '/not/in/inventory.py'):
                with self.assertRaisesRegex(ValueError, '绝对脚本路径'):
                    config.verify_probe_script([program, '-s', arg], assets)
            config.verify_probe_script([program, '-s', str(script)], assets)

    def test_commit_refuses_locked_source_and_preserves_target(self):
        from batch.storage import lock
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, source = root / 'a-target', root / 'z-source'
            output.mkdir()
            path = source / 'cases/source/attempt'
            path.mkdir(parents=True)
            with lock(source), patch.object(runner, 'batch_documents') as documents:
                with self.assertRaises(BlockingIOError):
                    adoption.import_attempt(output, source, 'target', 'source', path, {}, commit=True)
                documents.assert_not_called()
            self.assertFalse((output / 'index.json').exists())

    def test_evidence_comparison_reaches_atomic_registration(self):
        row = dict(case_id='target', pair_id='pair', mps=False, conditions={'seed': 1})
        other = dict(row, case_id='source')
        model = dict(id='local/test', tp=1, placement=[1], dp_attention=False, max_total_tokens='auto',
                     geometry_by_rank={'0': {'kind': 'mha'}})
        values = {key: 1 for key in ('process_environment', 'hardware', 'startup_seconds', 'poll_seconds',
                                     'cleanup_seconds', 'sample_seconds', 'probe_seconds')}
        values['models'] = [model]
        with tempfile.TemporaryDirectory() as temporary:
            output, source = Path(temporary) / 'target', Path(temporary) / 'source'
            left = self.prepared_fixture(output, row, observation('old'))
            right = self.prepared_fixture(source, other, observation('new'))
            write(output / 'index.json', dict(cases={}, pairs={'pair': str(left)}, invocations=[]))
            path = right / 'repetition-0001.attempt-0001'
            path.mkdir()
            run = right.parents[1]
            write(run / 'run.json', {'selected_cases': ['source']})
            lifecycle = run.parents[2] / 'lifecycle-1'
            lifecycle.mkdir()
            self.attempt_fixture(path, lifecycle, model, observation('new'))
            write(source / 'index.json',
                  {'cases': {'source': {'attempts': [{'path': str(path), 'external': str(lifecycle)}]}}})
            frozen = {root: dict(values=values, hardware_observation=hardware, code={}, patch={}, probe_protocol={})
                      for root, hardware in ((output, observation('old')), (source, observation('new')))}
            with patch.object(runner, 'batch_documents',
                              side_effect=lambda root: (frozen[root], {'cases': [row if root == output else other]})), \
                    patch.object(runner, 'runtime_config', return_value=values), patch.object(runner, 'verify_assets'), \
                    patch.object(runner.native, 'attempts', return_value=[{'path': str(path)}]), \
                    patch.object(runner, 'research_verdict', return_value={'valid': True, 'outcomes': {'success': 1}}):
                args = (output, source, 'target', 'source', path, {})
                self.assertEqual(adoption.import_attempt(*args)['status'], 'eligible')
                write(runner.capacity_path(output, row), {model['id']: [501]})
                with self.assertRaisesRegex(SharedFault, '容量不等价'):
                    adoption.import_attempt(*args, commit=True)
                self.assertEqual(read(output / 'index.json')['cases'], {})
                write(runner.capacity_path(output, row), {model['id']: [500]})
                self.assertEqual(adoption.import_attempt(*args, commit=True)['status'], 'adopted')
            self.assertEqual(read(output / 'index.json')['cases']['target']['adopted'], str(path))

    def prepared_fixture(self, root, row, hardware):
        case_id = row['case_id']
        directory = root / 'cases' / case_id / 'native/bench-runs/run/cases' / case_id
        directory.mkdir(parents=True)
        write(directory / 'workload.json', {'inputs': adoption.prepared.INPUTS})
        for name in ('prompts.jsonl', 'trace.jsonl', 'warmup.jsonl'):
            (directory / name).write_text('same input\n')
        contract = dict(pair_id=row['pair_id'], conditions=row['conditions'])
        write(directory / 'kvrepro-frozen.json',
              dict(identity=dict(contract=contract), files=adoption.prepared.validate_files(directory), content={}))
        metadata = dict(hardware, target_device_uuids={'local/test': [hardware['mapping']['1']]})
        write(directory / 'case.json', {'serving_metadata': metadata})
        write(directory.parents[4] / 'prepared-hardware.json', hardware)
        write(runner.capacity_path(root, row), {'local/test': [500]})
        return directory

    def attempt_fixture(self, path, lifecycle, model, hardware):
        write(path / 'measurement.json', {'wall_clock_origin_seconds': 2})
        write(path / 'repetition.json', {'window_end_seconds': 3})
        write(lifecycle / 'client-command.json', ['client', '--case', 'source'])
        write(lifecycle / 'execution.json', {'started_epoch': 1, 'finished_epoch': 6})
        write(lifecycle / 'hardware.json', hardware)
        write(lifecycle / 'resources.json', {'server_pids': []})
        gpu = hardware['mapping']['1']
        rank = dict(model=model['id'], rank=0, gpu=gpu, mps_server_pid=None, prefix_cache_enabled=False,
                    dp_attention=False, kvcached_enabled=True, geometry=model['geometry_by_rank']['0'],
                    capacity_tokens=500)
        metadata = dict(hardware, target_device_uuids={model['id']: [gpu]})
        write(lifecycle / 'metadata-ranks.json', {'ranks': [rank], 'serving_metadata': metadata})
