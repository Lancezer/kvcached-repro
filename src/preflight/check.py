"""禁用预检：先验证隔离条件，再执行已外部审核的 .pth。"""
import json
import os
import site
import sys
from importlib import metadata
from .dependencies import packages, dependency_issues
from .symbols import inspect_symbols

EXPECTED = {'CUDA_VISIBLE_DEVICES': '', 'ENABLE_KVCACHED': '0', 'KVCACHED_AUTOPATCH': '0'}


def guard():
    disabled = all(os.environ.get(key) == value for key, value in EXPECTED.items())
    if not sys.flags.isolated or not sys.flags.no_site or not disabled:
        raise SystemExit("使用 -I -S，并显式设置 CUDA_VISIBLE_DEVICES='' ENABLE_KVCACHED=0 KVCACHED_AUTOPATCH=0")


def imports():
    import torch
    import kvcached
    import kvcached._C
    return {
        'torch': torch.__file__, 'torch_version': torch.__version__,
        'kvcached': kvcached.__file__, 'extension': kvcached._C.__file__,
        'cuda_initialized': torch.cuda.is_initialized(),
        'sglang_imported': 'sglang' in sys.modules,
        'vllm_imported': 'vllm' in sys.modules,
        'sglang_autopatch_imported': 'kvcached.integration.sglang.autopatch' in sys.modules,
    }


def report():
    from packaging.specifiers import SpecifierSet
    installed = packages()
    root, symbols = inspect_symbols()
    supported = installed['sglang']['version'] in SpecifierSet('>=0.5.11,<0.5.21.dev0')
    plugins = [{'name': ep.name, 'value': ep.value}
               for ep in metadata.entry_points(group='sglang.srt.plugins')]
    return dict(python=sys.version, executable=sys.executable, environment=EXPECTED,
                packages=installed, sglang_in_supported_range=supported,
                direct_dependency_issues=dependency_issues(), sglang_source=root,
                static_symbols=symbols, sglang_plugins=plugins, imports=imports())


def invalid(result):
    flags = ('cuda_initialized', 'sglang_imported', 'vllm_imported',
             'sglang_autopatch_imported')
    return (bool(result['direct_dependency_issues'])
            or not result['sglang_in_supported_range']
            or not all(result['static_symbols'].values())
            or any(result['imports'][key] for key in flags))


def main():
    guard()
    site.main()
    result = report()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(invalid(result))
