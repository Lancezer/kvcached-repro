"""真实 CPU 进程收尾及 catalog 搬迁后的文件消费检查。"""
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import MagicMock, patch

import process
import protocol
from runtime import configuration
from service import validation
from xbench.catalog import runtime_catalog

ROOT = Path(__file__).resolve().parents[1]


class ValidationEvidenceTests(unittest.TestCase):
    def run_case(self, outcome):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            out = Path(directory) / 'out'
            code = 'raise SystemExit(7)' if outcome == 'startup' else 'import time; time.sleep(60)'
            stack.enter_context(patch.object(validation, 'server_command', return_value=[sys.executable, '-c', code]))
            stack.enter_context(patch.object(validation, 'check_port'))
            # 启动失败走真实 wait_ready 的子进程退出分支，不做网络探测。
            ready = (lambda proc, **kw: (proc.wait(), protocol.wait_ready(proc, **kw)))
            stack.enter_context(patch.object(validation, 'wait_ready', side_effect=ready if outcome == 'startup' else None))
            response = MagicMock()
            response.getresponse.return_value.status = 200
            response.getresponse.return_value.getheaders.return_value = []
            raw = b'data: {"meta_info":{"finish_reason":{"type":"stop"},"prompt_tokens":5,"completion_tokens":1}}\n\ndata: [DONE]\n'
            response.getresponse.return_value.read.return_value = raw
            stack.enter_context(patch.object(protocol.http.client, 'HTTPConnection', return_value=response))
            generate = stack.enter_context(patch.object(validation, 'generate', wraps=protocol.generate))
            if outcome == 'request':
                generate.side_effect = OSError('request failed')
            stop = stack.enter_context(patch.object(process, 'stop_group', wraps=process.stop_group))
            stack.enter_context(patch.object(validation, 'stop_group', stop))
            rc = validation.main(out, configuration())
            result = json.loads((out / 'result.json').read_text())
            pids = [group['pid'] for group in result['groups']]
            self.assertEqual(rc, int(outcome != 'success'))
            self.assertEqual(len(pids), 2 if outcome == 'success' else 1)
            self.assertEqual(result['process_groups'], pids)
            self.assertEqual(json.loads((out / 'final.cpu.json').read_text()), [])
            self.assertEqual([call.args[0].pid for call in stop.call_args_list], pids)
            for call in stop.call_args_list:
                self.assertIsNotNone(call.args[0].returncode)
            self.assertNotIn('cleanup_errors', result)
            if outcome == 'success':
                self.assertEqual((out / 'on.sse').read_bytes(), raw)

    def test_success_records_both_cleaned_groups(self):
        self.run_case('success')

    def test_request_failure_records_cleaned_group(self):
        self.run_case('request')

    def test_real_child_startup_failure_records_cleaned_group(self):
        self.run_case('startup')

    def test_launch_failure_has_no_invented_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out'
            with patch.object(validation, 'check_port'), patch.object(validation.subprocess, 'Popen', side_effect=OSError('launch')):
                self.assertEqual(validation.main(out, configuration()), 1)
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['process_groups'], [])
            self.assertNotIn('pid', result['groups'][0])
            self.assertFalse((out / 'final.cpu.json').exists())


class CatalogMetadataTests(unittest.TestCase):
    def test_metadata_files_survive_catalog_relocation(self):
        for model_absolute in (False, True):
            for serving_absolute in (False, True):
                with self.subTest(model_absolute=model_absolute, serving_absolute=serving_absolute):
                    self.check_metadata_files(model_absolute, serving_absolute)

    def check_metadata_files(self, model_absolute, serving_absolute):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, cwd, out = (root / name for name in ('original', 'cwd', 'output'))
            shutil.copytree(ROOT / 'config/xbench-smoke', source_dir)
            (source_dir / 'suites').mkdir()
            cwd.mkdir()
            out.mkdir()
            model = source_dir / 'metadata'
            model.mkdir()
            (model / 'config.json').write_text('{"vocab_size": 101, "max_position_embeddings": 2048}')
            serving = source_dir / 'serving.json'
            serving.write_text('{"schema_version": 1, "target_device_uuids": {}}')
            source = source_dir / 'catalog.toml'
            model_value = str(model) if model_absolute else 'metadata'
            serving_value = str(serving) if serving_absolute else 'serving.json'
            source.write_text(source.read_text().replace('seed = 0', 'serving_metadata_path = ' + json.dumps(serving_value) + '\nseed = 0').replace('api = "sglang"', 'model_metadata_path = ' + json.dumps(model_value) + '\napi = "sglang"'))
            original = {path: path.read_bytes() for path in source_dir.rglob('*') if path.is_file()}
            config = configuration()
            config.catalog = os.path.relpath(source, cwd)
            # 原 catalog、调用 cwd 和输出目录各不相同。
            with contextlib.chdir(cwd):
                catalog = runtime_catalog(root / 'unused', out, config)
            case = next(iter(tomllib.loads(catalog.read_text())['serving_cases'].values()))
            target = case['targets'][0]
            # 按消费接口实际打开 checkpoint/config.json 和服务元数据，不比较 TOML 文本。
            checkpoint = Path(target['model_metadata_path'])
            self.assertEqual(checkpoint, model)
            self.assertEqual(json.loads((checkpoint / 'config.json').read_text())['vocab_size'], 101)
            metadata = Path(case['serving_metadata_path'])
            self.assertEqual(metadata, serving)
            self.assertEqual(json.loads(metadata.read_bytes())['schema_version'], 1)
            prompts = [json.loads(row) for row in Path(target['prompts']['path']).read_text().splitlines()]
            arrivals = [json.loads(row) for row in Path(case['arrivals']['path']).read_text().splitlines()]
            self.assertEqual(arrivals[0]['prompt_id'], prompts[0]['prompt_id'])
            self.assertEqual(arrivals[0]['model_id'], target['model_id'])
            self.assertEqual({path: path.read_bytes() for path in original}, original)
