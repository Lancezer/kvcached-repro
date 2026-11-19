"""通用运行配置与独立 payload 的 CPU 集成检查。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipapp
from unittest.mock import patch

import main
from runtime import configuration, base_url
from service.settings import server_command
from xbench.catalog import runtime_catalog
from xbench.client import command

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_cli_priority(self):
        with patch.dict(os.environ, {'KVREPRO_PORT': '31000', 'KVREPRO_TP_SIZE': '3'}):
            config = configuration(main.parser().parse_args([
                'validate', '--out', '/tmp/unused', '--port', '32000', '--devices', '']))
        self.assertEqual((config.port, config.tp_size, config.devices), (32000, 3, ''))
        config.host = '::'
        self.assertEqual(base_url(config), 'http://[::1]:32000')

    def test_catalog_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'config/xbench-smoke', root / 'config')
            suite = root / 'config/suites/serving'
            suite.mkdir(parents=True)
            (suite / 'multi_model.py').write_text('pass\n')
            original = (root / 'config/catalog.toml').read_bytes()
            out = root / 'output'
            out.mkdir()
            config = configuration()
            config.model, config.model_id = '/other/models/Example', 'owner/Example'
            config.port, config.host, config.tp_size = 32345, 'localhost', 2
            config.max_new_tokens, config.max_inflight = 77, 5
            config.server_python, config.client_python = '/other/python', '/client/python'
            catalog = runtime_catalog(root, out, config)
            text = catalog.read_text()
            self.assertIn('http://localhost:32345', text)
            self.assertIn('owner/Example', text)
            self.assertIn('max_inflight = 5', text)
            row = json.loads((catalog.parent / 'arrivals.jsonl').read_text())
            self.assertEqual((row['model_id'], row['max_new_tokens']), ('owner/Example', 77))
            self.assertEqual((root / 'config/catalog.toml').read_bytes(), original)
            args = server_command(config)
            self.assertEqual(args[0], '/other/python')
            self.assertEqual(args[args.index('--model-path') + 1], config.model)
            self.assertEqual(args[args.index('--tp-size') + 1], '2')
            self.assertEqual(command(root, out, config, catalog)[0], '/client/python')

    def test_payload_real_child_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'payload'
            shutil.copytree(ROOT / 'src', source, ignore=shutil.ignore_patterns('__pycache__'))
            # 仅替换任务入口；保留真实 CLI、命令和环境构建模块。
            (source / 'service/validation.py').write_text(
                'import subprocess\nfrom service.settings import server_command, server_environment\n'
                'def main(out, config):\n'
                '    return subprocess.run(server_command(config), '
                'env=server_environment(True, "test", config)).returncode\n')
            package = root / 'sglang'
            package.mkdir()
            (package / '__init__.py').write_text('')
            (package / 'launch_server.py').write_text(
                'import os,sys,json\nprint(json.dumps(dict(argv=sys.argv, '
                'visible=os.environ.get("CUDA_VISIBLE_DEVICES"), '
                'order=os.environ.get("CUDA_DEVICE_ORDER"))))\n')
            payload = root / 'tools.pyz'
            zipapp.create_archive(source, payload, main='main:entrypoint')
            for visible in (None, '', '1', '2,0', 'GPU-one,MIG-two'):
                env = dict(os.environ, PYTHONPATH=str(root), CUDA_DEVICE_ORDER='PCI_BUS_ID')
                env.pop('KVREPRO_DEVICES', None)
                env.pop('CUDA_VISIBLE_DEVICES', None)
                if visible is not None:
                    env['CUDA_VISIBLE_DEVICES'] = visible
                self.check_payload(payload, env, visible)
            self.check_payload(payload, env, 'GPU-explicit', ['--devices', 'GPU-explicit'])

    def check_payload(self, payload, env, visible, extra=()):
        result = subprocess.run([sys.executable, '-I', '-S', str(payload), 'validate',
                                 '--out', '/tmp/unused', '--server-python', sys.executable,
                                 '--tp-size', '2', *extra], cwd='/', env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        child = json.loads(result.stdout)
        self.assertEqual(child['visible'], visible)
        self.assertEqual(child['order'], 'PCI_BUS_ID')
        self.assertEqual(child['argv'][child['argv'].index('--tp-size') + 1], '2')
