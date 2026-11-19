"""通过 AST 核对 SGLang 接口，不导入引擎。"""
import ast
from importlib.machinery import PathFinder
from pathlib import Path

TARGETS = {
    'srt/mem_cache/memory_pool.py': {'MHATokenToKVPool': '_create_buffers_normal'},
    'srt/mem_cache/allocator/token.py': {'TokenToKVPoolAllocator': 'alloc'},
    'srt/mem_cache/allocator/paged.py': {'PagedTokenToKVPoolAllocator': 'alloc_extend'},
    'srt/mem_cache/kv_cache_configurator.py': {'KVCacheConfigurator': '_profile_available_bytes'},
    'srt/managers/schedule_batch.py': {'ScheduleBatch': 'prepare_for_extend'},
    'srt/managers/scheduler.py': {'Scheduler': '__init__'},
}


def has_method(tree, cls, method):
    return any(isinstance(node, ast.ClassDef) and node.name == cls
               and any(isinstance(item, ast.FunctionDef) and item.name == method
                       for item in node.body) for node in tree.body)


def inspect_symbols():
    root = Path(PathFinder.find_spec('sglang').origin).parent
    symbols = {}
    for relative, classes in TARGETS.items():
        tree = ast.parse((root / relative).read_text())
        for cls, method in classes.items():
            symbols[f'{relative}:{cls}.{method}'] = has_method(tree, cls, method)
    return str(root), symbols
