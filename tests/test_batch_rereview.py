"""探针、局部故障与索引间隙的 CPU 回归测试。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from batch import config, runner
from batch.ownership import SharedFault
from batch.resources import Resources


class RereviewTests(unittest.TestCase):
    def test_completed_recovery_keeps_case_audit_and_does_not_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = dict(case_id='c', pair_id='p')
            directory = root / 'cases/c'
            run = directory / 'native/bench-runs/r'
            prepared = run / 'cases/c'
            prepared.mkdir(parents=True)
            (prepared / 'kvrepro-frozen.json').write_text('{}')
            (run / 'run.json').write_text('{}')
            old = dict(path='attempt', external='original', research={'valid': None})
            audit = dict(rerun_used=True, rerun_reason='startup failed', preparation_failures=['startup'])
            index = dict(cases={'c': dict(run=str(run), attempts=[old], **audit)}, pairs={})
            attempt = dict(old, manifest={'finished': True})
            execution = dict(native_code=0, cleanup=True, fault=None, lifecycle='original')
            with patch.object(runner, 'case_location', return_value=(directory, run)), \
                    patch.object(runner.native, 'attempts', return_value=[attempt]), \
                    patch.object(runner.native, 'locate', return_value=run), \
                    patch.object(runner, 'attempt_execution', return_value=execution), \
                    patch.object(runner, 'research_verdict', return_value={'valid': False}), \
                    patch.object(runner, 'launch_case') as launch:
                runner.run_case({}, row, root, index, (), None)
            launch.assert_not_called()
            entry = index['cases']['c']
            for key, value in audit.items():
                self.assertEqual(entry[key], value)
            self.assertEqual(entry['attempts'][0]['external'], 'original')
            self.assertIs(entry['attempts'][0]['research']['valid'], False)
            with self.assertRaisesRegex(ValueError, '一次'):
                runner.check_rerun(index, row, ['c'], 'again', directory)

    def test_rerun_marker_or_role_blocks_with_lost_index_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = dict(attempts=[{'research': {'valid': False}}])
            index = dict(cases={'c': entry})
            lifecycle = root / 'lifecycle-1'
            lifecycle.mkdir()
            for name, value in (('rerun.json', {'reason': 'first'}),
                                ('execution.json', {'role': 'rerun'})):
                path = lifecycle / name
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, '一次'):
                    runner.check_rerun(index, {'case_id': 'c'}, ['c'], 'again', root)
                path.unlink()
            (root / 'research-rerun').mkdir()
            self.assertTrue(runner.persisted_rerun(root))

    def test_marker_restores_audit_without_copying_other_execution_facts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'rerun.json').write_text(json.dumps({'reason': 'first'}))
            saved = runner.case_audit({'service_death': ['stale'], 'preparation_failures': ['original']},
                                      {'lifecycle': 'real'}, root)
            self.assertEqual(saved['rerun_reason'], 'first')
            self.assertTrue(saved['rerun_used'])
            self.assertNotIn('service_death', saved)
            self.assertEqual(saved['preparation_failures'], ['original'])

    def test_rules_forms_seal_same_explicit_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rules.json'
            path.write_text('{}')
            program = Path(temporary) / 'python3'
            program.write_text('declared interpreter asset')
            script = str(Path(config.__file__).parents[1] / 'main.py')
            for suffix in (['--rules', str(path)], ['--rules=' + str(path)]):
                args = [str(program), script, 'batch-probe', *suffix, '--kind', 'rank']
                with patch.object(config, 'verify_rule_programs') as verify:
                    config.verify_probe_commands(dict(rank_probe=args, kv_probe=args),
                                                 {str(program): 'hash', script: 'hash', str(path): 'hash'})
                self.assertEqual(verify.call_args.args[1], path)

    def test_rules_reject_duplicate_empty_abbreviation_and_hidden_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / 'rules.json')
            Path(path).write_text('{}')
            for args in (['--rules', path, '--rules=' + path], ['--rules='], ['--rules'],
                         ['--rules', '--kind'], ['--rule=' + path], ['--', '--rules', path]):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    config.probe_rules(args)
            self.assertIsNone(config.probe_rules(['custom', 'batch-probe']))

    def test_confirmed_service_death_rechecks_native_cancellations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            run.mkdir()
            (root / 'configuration.json').write_text('{}')
            for error in ('benchmark measurement did not complete', 'worker lost before terminal request accounting',
                          'task cancelled; scope safely drained'):
                entry = dict(cleanup=True, native_code=130,
                             service_death=[dict(pid=123, returncode=9, observed_epoch=1.)])
                with patch.object(runner, 'verify_frozen') as frozen, \
                        patch.object(runner, 'shared_preflight') as ports, patch.object(runner, 'recheck_devices') as devices:
                    runner.classify_native_fault({}, entry, {'infrastructure_error': error}, run)
                frozen.assert_called_once()
                ports.assert_called_once()
                devices.assert_called_once()
                self.assertEqual(entry['native_code'], 130)
                self.assertEqual(entry['native_fault_scope'], 'local_service_death_after_verified_cleanup')

    def test_cancellation_requires_fact_cleanup_and_successful_shared_recheck(self):
        error = 'task cancelled; scope safely drained'
        for entry in ({'cleanup': True}, {'cleanup': True, 'service_death': True},
                      {'cleanup': False, 'service_death': [{'pid': 1, 'returncode': 9, 'observed_epoch': 1.}]}):
            with self.assertRaises(SharedFault), patch.object(runner, 'verify_frozen') as verify:
                runner.classify_native_fault({}, entry, {'infrastructure_error': error}, Path('/unused'))
            verify.assert_not_called()
        self.assertIsNone(runner.local_fault_scope({'service_death': [{'pid': 1, 'returncode': 9,
                                         'observed_epoch': 1.}]}, 'arbitrary failure'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configuration.json').write_text('{}')
            entry = dict(cleanup=True, service_death=[dict(pid=1, returncode=9, observed_epoch=1.)])
            with patch.object(runner, 'verify_frozen'), patch.object(runner, 'shared_preflight'), \
                    patch.object(runner, 'recheck_devices', side_effect=SharedFault('NVLink unknown')), \
                    self.assertRaises(SharedFault):
                runner.classify_native_fault({}, entry, {'infrastructure_error': error}, root / 'run')
            self.assertNotIn('native_fault_scope', entry)

    def test_service_death_fact_excludes_shared_controller_death(self):
        service = SimpleNamespace(pid=1, returncode=9, poll=lambda: 9)
        controller = SimpleNamespace(pid=2, returncode=1, poll=lambda: 1)
        resources = Resources({}, {}, Path('/unused'))
        resources.services = [service]
        resources.processes = [service]
        self.assertEqual(resources.dead_services()[0]['pid'], 1)
        resources.processes.append(controller)
        self.assertIsNone(resources.dead_services())
        with self.assertRaises(SharedFault):
            runner.classify_native_fault({}, {'cleanup': True, 'service_death': None},
                                         {'infrastructure_error': 'benchmark worker exited with code 9'}, Path('/unused'))


# 显式复跑的索引落盘间隙回归场景。
def rerun_gap_fixture(root, old_run=None):
    directory = root / 'cases/c/research-rerun'
    run = directory / 'native/bench-runs/new'
    attempt = run / 'cases/c/attempt-1'
    attempt.mkdir(parents=True)
    lifecycle = directory / 'lifecycle-1000000000'
    lifecycle.mkdir()
    for path, value in (
            (directory / 'rerun.json', dict(role='rerun', reason='first')),
            (run / 'run.json', dict(selected_cases=['c'])),
            (attempt.parent / 'kvrepro-frozen.json', {}),
            (attempt / 'measurement.json', dict(wall_clock_origin_seconds=2.)),
            (attempt / 'repetition.json', dict(window_end_seconds=1.)),
            (lifecycle / 'client-command.json', ['client', '--case', 'c']),
            (lifecycle / 'execution.json', dict(role='rerun', native_code=0, cleanup=True,
                     fault=None, started_epoch=1., finished_epoch=4.))):
        path.write_text(json.dumps(value))
    entry = dict(run=old_run, attempts=[], preparation_failures=['startup'])
    return run, attempt, dict(cases={'c': entry}, pairs={})


def recover_gap(root, attempt, index, finished=True):
    value = dict(path=str(attempt), manifest={'finished': finished})
    with patch.object(runner.native, 'attempts', return_value=[value]), \
            patch.object(runner, 'research_verdict', return_value={'valid': None}), \
            patch.object(runner, 'launch_case') as launch:
        runner.run_case({}, dict(case_id='c', pair_id='p'), root, index, (), None)
    launch.assert_not_called()


class RerunIndexGapTests(unittest.TestCase):
    def test_old_unstarted_index_recovers_new_role_before_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, attempt, index = rerun_gap_fixture(root)
            recover_gap(root, attempt, index)
            entry = index['cases']['c']
            self.assertEqual(entry['run'], str(run))
            self.assertEqual(entry['preparation_failures'], ['startup'])
            self.assertEqual(entry['attempts'][0]['external'], str(run.parents[2] / 'lifecycle-1000000000'))
            self.assertIsNone(entry['attempts'][0]['research']['valid'])
            self.assertTrue(entry['rerun_used'])
            self.assertEqual(entry['rerun_reason'], 'first')
            with self.assertRaisesRegex(ValueError, '一次'):
                runner.check_rerun(index, {'case_id': 'c'}, ['c'], 'again', root / 'cases/c')

    def test_old_original_index_yields_to_new_role_before_index_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_run = root / 'cases/c/native/bench-runs/old'
            old_run.mkdir(parents=True)
            (old_run / 'run.json').write_text(json.dumps(dict(selected_cases=['c'])))
            run, attempt, index = rerun_gap_fixture(root, str(old_run))
            old = dict(path='old-attempt', research={'valid': False})
            index['cases']['c']['attempts'] = [old]
            self.assertEqual(runner.continuing_run(index, {'case_id': 'c'}, root / 'cases/c'), run)
            recover_gap(root, attempt, index)
            self.assertEqual(index['cases']['c']['run'], str(run))
            self.assertEqual(index['cases']['c']['attempts'][0], old)
            self.assertEqual(index['cases']['c']['attempts'][1]['path'], str(attempt))

    def test_marker_without_unique_run_blocks_without_launch(self):
        for missing in ('run.json', 'rerun.json', 'execution.json', None):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run, attempt, index = rerun_gap_fixture(root)
                if missing is None:
                    (run.parent / 'other').mkdir()
                else:
                    paths = {'run.json': run / 'run.json', 'rerun.json': run.parents[2] / 'rerun.json',
                             'execution.json': run.parents[2] / 'lifecycle-1000000000/execution.json'}
                    paths[missing].unlink()
                with patch.object(runner, 'launch_case') as launch:
                    with self.assertRaises(SharedFault):
                        recover_gap(root, attempt, index)
                    launch.assert_not_called()

    def test_unfinished_new_role_blocks_without_second_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, attempt, index = rerun_gap_fixture(root)
            with patch.object(runner, 'launch_case') as launch, self.assertRaises(SharedFault):
                recover_gap(root, attempt, index, finished=False)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
