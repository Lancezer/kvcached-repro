import json
import os
import runpy
import sys
from subprocess import CompletedProcess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import kv_export
import kv_probe
import main
from batch import config, runner
from batch.ownership import identity
from batch.resources import Resources, service_command


class BatchPortableFixTests(unittest.TestCase):
    def test_launcher_exports_only_with_service_opt_in(self):
        class Scheduler:
            def process_input_requests(self, requests):
                return requests

        scheduler_module = ModuleType('sglang.srt.managers.scheduler')
        scheduler_module.Scheduler = Scheduler
        model = dict(id='model', checkpoint='checkpoint', tp=1, dp=1, placement=[0],
                     port=12345, context_length=1024, mem_fraction_static=0.5,
                     dp_attention=False, max_total_tokens='auto')
        record = identity(os.getpid())

        def capture(_scheduler):
            return dict(pid=record['pid'], rank=0, checkpoint=model['checkpoint'],
                        process_identity=record, observed_epoch=kv_export.time.time(), gpu='GPU-test')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / 'observations'
            config_path = root / 'config.json'
            settings = dict(server_python=sys.executable, models=[model],
                            mps_control='/test/mps', nvidia_smi='/test/smi', logical_devices={'0': 'GPU-test'})
            config_path.write_text(json.dumps(settings))
            (root / 'resources.json').write_text(json.dumps(
                dict(records=[record], server_pids=[42], mps_env={})))
            resources = Resources(settings, {}, root)
            launcher = service_command(resources.config, model)[2]
            original = Scheduler.process_input_requests
            with (patch.dict(os.environ, {}, clear=True),
                  patch.dict(sys.modules, {scheduler_module.__name__: scheduler_module}),
                  patch('ofd_compat.enable'), patch('batch.ipc_journal.install'),
                  patch.object(kv_export, 'capture', side_effect=capture) as sample,
                  patch.object(resources, 'reserve_ipc', return_value=('ipc', directory))):
                runpy.run_path(launcher, run_name='__mp_main__')
                self.assertIs(Scheduler.process_input_requests, original)
                sample.assert_not_called()
                with patch.dict(os.environ, KVREPRO_KV_EXPORT_DIRECTORY=str(observations)):
                    service_env = resources.service_env(model)
                self.assertNotIn('KVREPRO_KV_EXPORT_DIRECTORY', os.environ)
                with patch.dict(os.environ, service_env):
                    runpy.run_path(launcher, run_name='__mp_main__')
                    self.assertEqual(Scheduler().process_input_requests(['request']), ['request'])
            rules = dict(config=str(config_path), directory=str(observations), probe_seconds=1)
            with (patch.dict(os.environ, resources.probe_environment()),
                  patch('batch.probe.subprocess.run', side_effect=[
                      CompletedProcess([], 0, str(record['pid'])),
                      CompletedProcess([], 0, '0, GPU-test, Test GPU, 1, driver'),
                  ]) as run, patch('batch.probe.importlib.metadata.version', return_value='test')):
                observed = kv_probe.observe(rules, 'rank')
            rows = observed['ranks']
        self.assertEqual([(row['model'], row['rank'], row['mps_server_pid']) for row in rows],
                         [('model', 0, 42)])
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ['/test/mps', '/test/smi'])
        sample.assert_called_once()

    def test_batch_freeze_duration_reaches_existing_configuration_loader(self):
        class StopAfterConfig(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'config.json'
            source.write_text(json.dumps(dict(duration=600, models=[])))
            output = Path(directory) / 'output'
            base = ['main.py', 'batch-freeze', '--config', str(source), '--out', str(output)]
            for extra, expected in (([], 600), (['--duration', '42'], 42.0)):
                with (self.subTest(extra=extra), patch.object(sys, 'argv', base + extra),
                      patch.object(config, 'resolve_paths', return_value={}),
                      patch.object(config, 'validate'),
                      patch.object(runner.settings, 'assets', side_effect=StopAfterConfig) as assets):
                    with self.assertRaises(StopAfterConfig):
                        main.main()
                    self.assertEqual(assets.call_args.args[0]['duration'], expected)
