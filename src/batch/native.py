"""单case catalog和CLI桥接；不接管原生调度/attempt/continue。"""
import json
import os
import shutil
import subprocess
from pathlib import Path
from .config import file_hash
from .storage import read


def target(config, model):
    return dict(model_id=model['id'], api='sglang', base_url=f'http://127.0.0.1:{model["port"]}',
                prompts=dict(kind='jsonl', path=config['prompts']), output_tokens=config['output_tokens'],
                model_metadata_path=model['metadata'], sampling=config['sampling'])


def case_value(config, row, metadata):
    conditions = row['conditions']
    return dict(description='kvcached paired case ' + row['fingerprint'], module='serving.multi_model',
                mode='client', seed=conditions['seed'], max_inflight=config['max_inflight'],
                warmup_requests_per_target=config['warmup'], bucket_seconds=config['bucket_seconds'],
                serving_metadata_path=str(metadata), targets=[target(config, m) for m in config['models']],
                arrivals=dict(kind='poisson', duration_seconds=config['duration'],
                              rates={m['id']: conditions['rate'] for m in config['models']}))


def toml_lines(value, prefix=()):
    # 小型确定性TOML写出器：配置只用标量/嵌套dict/字典数组，不接受None或任意对象。
    lines = []
    if prefix:
        lines.append('[' + '.'.join(json.dumps(p) for p in prefix) + ']')
    for key, item in value.items():
        if not isinstance(item, dict) and not (isinstance(item, list) and item and isinstance(item[0], dict)):
            lines.append(json.dumps(key) + ' = ' + json.dumps(item, ensure_ascii=False, allow_nan=False))
    for key, item in value.items():
        if isinstance(item, dict):
            lines.extend(toml_lines(item, (*prefix, key)))
        elif isinstance(item, list) and item and isinstance(item[0], dict):
            for child in item:
                lines.append('[[' + '.'.join(json.dumps(p) for p in (*prefix, key)) + ']]')
                lines.extend(inline_lines(child))
    return lines


def inline_value(value):
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + inline_value(v) for k, v in value.items()) + '}'
    if value is None:
        raise ValueError('TOML无null；请提供显式值')
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def inline_lines(value):
    return [json.dumps(k) + ' = ' + inline_value(v) for k, v in value.items()]


def catalog(config, row, directory):
    path = directory / 'catalog.toml'
    metadata = directory / 'serving-metadata.json'
    source = Path(config['suite'])
    destination = directory / 'suites/serving/multi_model.py'
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    value = dict(serving_cases={row['case_id']: case_value(config, row, metadata)})
    path.write_text('\n'.join(toml_lines(value)) + '\n')
    return path


def environment(config, row, source=None):
    env = dict(os.environ)
    root = Path(config['source'])
    env.update(ENABLE_KVCACHED='0', KVCACHED_AUTOPATCH='0', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=f'{root}/src:{root}/src/xpool-dev',
               KVREPRO_PREPARED_CONTRACT=json.dumps(dict(pair_id=row['pair_id'], conditions=row['conditions'])))
    env.pop('KVREPRO_PREPARED_SOURCE', None)
    if source is not None:
        env['KVREPRO_PREPARED_SOURCE'] = str(source)
    return env


def command(config, row, directory, catalog_path=None, continuing=None, rerun=False):
    args = [config['client_python'], '-s', '-c',
            'from xbench.cli import main; raise SystemExit(main())', 'run']
    if continuing is not None:
        args += ['--all', '--continue', str(continuing)]
        return args + (['--rerun-failure'] if rerun else [])
    return args + ['--case', row['case_id'], '--catalog', str(catalog_path), '--repetitions', '1',
                   '--cache-root', str(directory / 'native')]


def locate(directory, case_id):
    root = directory / 'native/bench-runs'
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()] if root.exists() else []
    if len(candidates) != 1:
        raise ValueError('需要唯一原生run，缺口保留人工审计')
    run = candidates[0]
    manifest = read(run / 'run.json')
    if manifest['selected_cases'] != [case_id]:
        raise ValueError('原生run选择与冻结case身份不符')
    return run


def attempts(run, row, config):
    directory = run / 'cases' / row['case_id']
    script = ('import sys,json\nfrom pathlib import Path\n'
              'from xbench.harness.serving.measure import BenchCaseManifest,RepetitionManifest\n'
              'd=Path(sys.argv[1]); m=BenchCaseManifest.model_validate_json((d/"case.json").read_bytes())\n'
              'if str(m.case.id) != sys.argv[2]:\n    raise ValueError("case identity mismatch")\n'
              'paths=m.effective_attempts(d)\n'
              'print(json.dumps([dict(path=str(p),manifest=RepetitionManifest.from_directory(p).model_dump(mode="json")) '
              'for p in paths.values()]))')
    result = subprocess.run([config['client_python'], '-s', '-c', script, str(directory), row['case_id']],
                            env=environment(config, row), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def patch_identity(config):
    from .patch import verify_prepared
    verify_prepared(config['source'])
    root = Path(config['source'])
    paths = [root / 'kvrepro-prepared.patch', *sorted((root / 'src').rglob('*.py'))]
    identities = {str(path.relative_to(root)): file_hash(path) for path in paths}
    identities['suite'] = file_hash(config['suite'])
    return identities


def validate_attempt(config, row, path):
    script = ('import sys,json; from pathlib import Path; '
              'from xbench.harness.serving.measure import BenchCaseManifest,load_measurement; '
              'p=Path(sys.argv[1]); m=BenchCaseManifest.model_validate_json((p.parent/"case.json").read_bytes()); '
              'w=m.load_workload(p.parent); seal,s=load_measurement(p,w); '
              'print(json.dumps(dict(evidence_complete=s.evidence_complete, '
              'execution_complete=s.execution_complete, measurement_available=s.measurement_available)))')
    result = subprocess.run([config['client_python'], '-s', '-c', script, str(path)],
                            env=environment(config, row), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)
