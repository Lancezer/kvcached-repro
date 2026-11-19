"""固定版本局部文本适配；拒绝对子模块或共享源码调用。"""
import difflib
import hashlib
import shutil
from pathlib import Path
from .config import file_hash
from localpaths import allowed

ORIGINAL_SHA256 = '5708b2d13355a1e1dc38a604dd125f044f44e8372e554749233a4cbc31eb44e1'


def verify_prepared(source):
    root = allowed(source).resolve(strict=True)
    runner = root / 'src/xpool-dev/xbench/harness/serving/runner.py'
    verify_copy_paths(root, runner)
    text = runner.read_text()
    start = text.index('        from kvrepro_prepared import load_source, materialize, freeze\n')
    directory = text.index('        directory =', start)
    branch = text.index('        if imported:\n', directory)
    writing = text.index('            write_json(\n', branch)
    end = text.index('        freeze(directory, case, workload)\n', writing)
    old = ('        workload = prepare_workload(case, config=config)\n' + text[directory:branch]
           + ''.join(line[4:] for line in text[writing:end].splitlines(True)))
    original = text[:start] + old + text[end + len('        freeze(directory, case, workload)\n'):]
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256 or patched_runner(original) != text:
        raise ValueError('prepared runner不匹配固定原版及预期变换')
    verify_prepared_files(root, original, text)


def verify_prepared_files(root, original, text):
    expected = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/runner.py', tofile='b/runner.py'))
    if (root / 'kvrepro-prepared.patch').read_text() != expected:
        raise ValueError('prepared patch与预期变换不一致')
    if file_hash(root / 'src/xpool-dev/kvrepro_prepared.py') != file_hash(Path(__file__).with_name('prepared.py')):
        raise ValueError('prepared helper与当前版本不一致')


def verify_copy_paths(root, runner):
    repository = Path(__file__).resolve().parents[2]
    if root == repository or root.is_relative_to(repository / 'external') or (root / '.git').exists():
        raise ValueError('仅允许无.git的独立客户端副本')
    for path in (runner, root / 'src/xpool-dev/kvrepro_prepared.py', root / 'kvrepro-prepared.patch'):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('prepared组合必须为独立副本内普通文件')


def patched_runner(text):
    start = text.index('        workload = prepare_workload(case, config=config)\n')
    end = text.index('        manifest = BenchCaseManifest(', start)
    original = text[start:end]
    marker = '        write_json(\n'
    offset = original.index(marker)
    writing = original[offset:]
    prefix = ('        from kvrepro_prepared import load_source, materialize, freeze\n'
              '        workload = load_source(case, PreparedWorkload)\n'
              '        imported = workload is not None\n'
              '        workload = workload if imported else prepare_workload(case, config=config)\n')
    directory = original[original.index('        directory ='):offset]
    branch = ('        if imported:\n'
              '            materialize(directory, case, workload, PreparedWorkload)\n'
              '        else:\n' + ''.join('    ' + line for line in writing.splitlines(True)))
    return text[:start] + prefix + directory + branch + '        freeze(directory, case, workload)\n' + text[end:]


def prepare_frozen(source):
    runner = original_runner(source)
    source = allowed(source).resolve(strict=True)
    old, helper = runner.read_text(), source / 'src/xpool-dev/kvrepro_prepared.py'
    patch_file = source / 'kvrepro-prepared.patch'
    if helper.exists() or helper.is_symlink() or patch_file.exists() or patch_file.is_symlink():
        raise ValueError('不覆盖已有适配文件')
    new = patched_runner(old)
    shutil.copyfile(Path(__file__).with_name('prepared.py'), helper)
    runner.write_text(new)
    difference = difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/runner.py', tofile='b/runner.py')
    with patch_file.open('x') as output:
        output.write(''.join(difference))
    return dict(runner=file_hash(runner), helper=file_hash(helper), patch=file_hash(source / 'kvrepro-prepared.patch'))


def original_runner(source):
    source = allowed(source).resolve(strict=True)
    repository = Path(__file__).resolve().parents[2]
    if source == repository or source.is_relative_to(repository / 'external'):
        raise ValueError('只允许显式独立客户端副本，不允许仓库/子模块')
    if (source / '.git').exists():
        raise ValueError('要求无.git的独立archive副本')
    runner = source / 'src/xpool-dev/xbench/harness/serving/runner.py'
    if runner.is_symlink() or not runner.resolve().is_relative_to(source):
        raise ValueError('runner必须是独立副本内的普通文件')
    if file_hash(runner) != ORIGINAL_SHA256:
        raise ValueError('runner不是固定提交0d3267e的原始文件')
    return runner
