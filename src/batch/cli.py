"""批次和数据工具的显式命令入口；无自动下载或自动启动。"""
from pathlib import Path
from .config import MODEL_PATHS, PATHS


def configure(commands):
    patch = commands.add_parser('prepare-frozen', help='仅准备独立客户端prepared适配')
    patch.add_argument('source', type=Path)
    freeze = commands.add_parser('batch-freeze', help='验证已有资产与实际CUDA设备并冻结顺序批次；不启动服务')
    freeze.add_argument('--config', required=True, type=Path)
    freeze.add_argument('--out', required=True, type=Path)
    freeze.add_argument('--duration', type=float)
    runtime_options(freeze)
    run = commands.add_parser('batch-run', help='显式执行/恢复已冻结批次；需另行协调GPU')
    run.add_argument('--out', required=True, type=Path)
    run.add_argument('--clean-orphans', action='store_true')
    run.add_argument('--rerun-case', action='append', default=[])
    run.add_argument('--reason')
    runtime_options(run)
    configure_offline(commands)


def runtime_options(parser):
    for name in PATHS:
        parser.add_argument('--' + name.replace('_', '-'))
    for name in MODEL_PATHS:
        parser.add_argument('--model-' + name.replace('_', '-'), action='append')


def overrides(args):
    return {name: getattr(args, name) for name in (*PATHS, *('model_' + n for n in MODEL_PATHS))}


def configure_offline(commands):
    adopt = commands.add_parser('batch-adopt', help='离线预览独立 attempt，显式 --commit 才写目标索引')
    for name in ('out', 'source-batch', 'attempt'):
        adopt.add_argument('--' + name, required=True, type=Path)
    for name in ('case', 'source-case'):
        adopt.add_argument('--' + name, required=True)
    adopt.add_argument('--commit', action='store_true')
    runtime_options(adopt)
    ipc = commands.add_parser('batch-ipc-clean', help='内部：仅清理已登记且已停进程的私有IPC')
    ipc.add_argument('--registry', required=True, type=Path)
    tmux = commands.add_parser('batch-tmux-command', help='仅打印tmux启动命令，不开启会话')
    tmux.add_argument('--out', required=True, type=Path)
    tmux.add_argument('--session', required=True)
    probe = commands.add_parser('batch-probe', help='显式读取自有日志/实际MPS接入证据')
    probe.add_argument('--rules', required=True, type=Path)
    probe.add_argument('--kind', choices=('rank', 'kv'), required=True)
    report = commands.add_parser('batch-report', help='原生证据校验与离线研究派生')
    report.add_argument('--out', required=True, type=Path)
    report.add_argument('--destination', required=True, type=Path)
    report.add_argument('--source')
    download = commands.add_parser('data-download', help='显式下载固定普通V3；不会由批次调用')
    download.add_argument('--out', required=True, type=Path)
    verify = commands.add_parser('data-verify', help='仅校验已有普通V3文件')
    verify.add_argument('source', type=Path)
    data = commands.add_parser('data-prepare', help='已有普通V3与本地tokenizer生成原生prompts池')
    data.add_argument('--source', required=True, type=Path)
    data.add_argument('--config', required=True, type=Path)
    data.add_argument('--out', required=True, type=Path)


def dispatch(args):
    from . import runner
    if args.action == 'prepare-frozen':
        from .patch import prepare_frozen
        prepare_frozen(args.source)
    elif args.action == 'batch-freeze':
        runner.freeze(args.config, args.out, dict(overrides(args), duration=args.duration))
    elif args.action == 'batch-run':
        return runner.run(args.out, args.clean_orphans, args.rerun_case, args.reason, overrides(args))
    else:
        return offline_command(args)
    return 0


def offline_command(args):
    if args.action == 'batch-ipc-clean':
        from .ipc import clean
        from .storage import read
        from .ownership import audit
        registry = read(args.registry)
        if audit(registry['records'] + registry.get('descendants', [])):
            raise ValueError('自有进程仍活，禁止IPC清理')
        from cleanup_diagnostics import emit_cleanup_diagnostic
        try:
            clean(registry, args.registry.parent)
        except Exception as error:
            emit_cleanup_diagnostic(error, operation='clean_ipc')
            raise
    elif args.action == 'batch-adopt':
        return adopt_attempt(args)
    elif args.action == 'batch-report':
        report(args.out, args.destination, args.source)
    elif args.action == 'batch-probe':
        import json
        from .probe import observe
        print(json.dumps(observe(args.rules, args.kind), allow_nan=False))
    elif args.action == 'batch-tmux-command':
        from .tmux import command
        print(command(args.out, args.session))
    else:
        data_command(args)
    return 0


def adopt_attempt(args):
    import json
    from subprocess import SubprocessError
    from .adoption import import_attempt
    from .ownership import SharedFault
    try:
        result = import_attempt(args.out, args.source_batch, args.case, args.source_case,
                                args.attempt, overrides(args), args.commit)
    except (SharedFault, OSError, ValueError, KeyError, TypeError, IndexError, SubprocessError) as error:
        print(json.dumps(dict(status='rejected', reason=str(error)), ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


def report(output, destination, source):
    import sys
    from .config import input_value, local_path
    from .runner import batch_documents
    root = Path(local_path(input_value('source', {'source': source})[0], 'source', directory=True))
    batch_documents(output)
    # 本项目也有xbench包；离线原生导入必须明确优先独立客户端。
    sys.path[:0] = [str(root / 'src'), str(root / 'src/xpool-dev')]
    from .report import report as generate
    generate(output, destination)


def data_command(args):
    from data import source
    from .storage import read
    from localpaths import allowed
    if args.action == 'data-download':
        source.download(args.out)
    elif args.action == 'data-verify':
        source.verify(args.source)
    else:
        from data.sharegpt import prepare
        config = read(args.config)
        for model in config['models']:
            if not isinstance(model.get('tokenizer'), str):
                raise ValueError('必须提供已有本地tokenizer目录')
            model['tokenizer'] = str(allowed(args.config.resolve().parent / model['tokenizer']))
            if isinstance(model.get('assets_manifest'), str):
                model['assets_manifest'] = str(allowed(args.config.resolve().parent / model['assets_manifest']))
        prepare(args.source, args.out, config)
